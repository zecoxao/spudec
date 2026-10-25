"""
Pseudocode generation from the structured AST.

Three things happen here, none of which touch the SSA IR -- this is a
*rendering*, so `spudec.dump()` still shows the verifiable three-address form
underneath and the SSA verifier still applies to it.

**Naming.** Structured code has no place for phi nodes: the control flow that
selects between their arguments is now explicit. So a phi web (the transitive
closure of a phi's result and arguments) prints as one name, and
`r33#2 = phi(r33#1, r33#7)` disappears -- but only as far as the members'
live ranges allow. :mod:`outssa` colours each web against the interference
graph and turns the merges that no longer hold into copies; this module names
the colours it produced. Distinct webs, and distinct colours within one web,
get `_2`, `_3` suffixes rather than being conflated, so two values that are
live at the same time never share a name.

**Expression inlining.** A definition with exactly one use, in the same block,
with no side effect in between, is folded into its use site. This is what
turns

    t17 = r33 + 0x400
    store.b mem, t17, r34

into `*(u8 *)(r33 + 0x400) = r34;`. Loads are only inlined when no memory
write sits between the load and its use, so reordering across a store cannot
happen.

**Rendering.** Scalar operations (those the scalarisation pass proved live
only in the preferred slot) print as ordinary C. Vector operations print as
calls with an element-width suffix, because that is what they are -- pretending
`shufb` is an expression would be a lie.

No IDA imports; pass ``name_of`` to resolve call targets to real symbol names.
"""

from .ir import (Op, EW, OP_NAME, INFIX, MEM_READS, MEM_WRITES, ABI_OPS,
                 EW_SUFFIX, Var)
from . import regs, channels, types, frame, outssa
from .structure import (Basic, If, Loop, Break, Continue, Goto, Label,
                        Return, Tail)

CTYPE = {EW.B: "u8", EW.H: "u16", EW.W: "u32", EW.D: "u64", EW.Q: "qword"}
MAX_INLINE_DEPTH = 4

# Bitwise operations and their exact C spelling, with a flag saying whether
# the form already brackets itself.  `~` binds tighter than `&` and `|`, so
# `a & ~b` needs no inner parentheses.
# Compares produce an all-ones mask per lane, not 0 or 1, so `a > b` is only
# a fair rendering once scalarisation has *proved* the value lives in the
# preferred slot -- `insn.scalar`.  Type evidence alone is weaker, and a mask
# that reaches an `&` or a `selb` would then read as a boolean.  So the
# type-based rule below deliberately leaves these as lane calls.
_MASK_OPS = frozenset((Op.CMPEQ, Op.CMPGT, Op.CMPGTU))

_BITWISE = {
    Op.AND:  ("%s & %s", False),
    Op.OR:   ("%s | %s", False),
    Op.XOR:  ("%s ^ %s", False),
    Op.NAND: ("~(%s & %s)", True),
    Op.NOR:  ("~(%s | %s)", True),
    Op.EQV:  ("~(%s ^ %s)", True),
    Op.ANDC: ("%s & ~%s", False),
    Op.ORC:  ("%s | ~%s", False),
}


# ---------------------------------------------------------------------------
# CodeLines container
# ---------------------------------------------------------------------------


class CodeLines(list):
    """List of generated C lines carrying parallel line_eas, line_to_eas, ea_to_line, header_count."""
    def __init__(self, iterable=None):
        super(CodeLines, self).__init__(iterable or [])
        self.line_eas = []
        self.line_to_eas = {}
        self.ea_to_line = {}
        self.header_count = 0


# ---------------------------------------------------------------------------
# naming (phi-web coalescing)
# ---------------------------------------------------------------------------


class Namer(object):

    def __init__(self, func, param_names=None, call_args=None):
        self._parent = {}
        self._name = {}
        self._colours = []
        self._fixed = dict(param_names or {})
        self._call_args = call_args
        self._build(func)

    def _find(self, k):
        root = k
        while self._parent.get(root, root) != root:
            root = self._parent[root]
        while self._parent.get(k, k) != k:
            self._parent[k], k = root, self._parent[k]
        return root

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra

    def webs(self):
        """
        The groups of SSA keys that print as one variable.

        The colours, not the phi webs: type inference unifies over these, and
        a web that had to be split carries two variables with two independent
        types.  Returning the web would have unified them again.
        """
        return [set(c) for c in self._colours]

    def _build(self, func):
        keys = []
        for insn in func.insns():
            d = insn.defines()
            if d is not None:
                self._parent.setdefault(d.key(), d.key())
                keys.append(d.key())
            for u in insn.uses():
                self._parent.setdefault(u.key(), u.key())

        for insn in func.insns():
            if insn.op != Op.PHI:
                continue
            d = insn.defines()
            for s in insn.srcs:
                if s.is_var:
                    self._union(d.key(), s.key())

        # One name per web; distinct webs of a register get numbered suffixes
        # rather than being merged, so unrelated values never share a name.
        webs = {}
        for k in sorted(self._parent, key=lambda x: (x[0], x[1])):
            webs.setdefault(self._find(k), []).append(k)

        # ...but a web is only safe to print as one variable if no two of its
        # members are live at the same time.  Where they are, one name would
        # mean two things, and the listing reads the wrong one:
        #
        #     r9#19  = selb r3#22, r9#18, r11#16
        #     t60#1  = r9#1 & 0x3FFF0        <- the incoming pointer
        #     mem#20 = storeq mem#19, t60#1, r9#19
        #
        # printed as `*(qword *)(r9 & 0x3FFF0) = r9;`, whose address is now
        # the value assigned on the line above.  So each web is coloured: a
        # member joins an existing colour only if it interferes with none of
        # that colour's members, and the phis that no longer hold within one
        # colour are printed rather than suppressed (see `_suppressed`).
        colour, _groups = outssa.colours(func, self._call_args)

        seen = {}
        for root in sorted(webs, key=lambda r: (r[0], r[1])):
            by_colour = {}
            for k in sorted(webs[root]):
                by_colour.setdefault(colour.get(k, ("?", k)), []).append(k)
            colours = [by_colour[c] for c in sorted(by_colour, key=str)]
            # The colour holding the function's incoming value for an argument
            # register is that parameter, and takes its positional name; a
            # later, interfering reuse of the same register is not the
            # parameter and must not borrow its name.
            self._colours.extend(colours)
            for members in colours:
                fixed = next((self._fixed[k] for k in sorted(members)
                              if k in self._fixed), None)
                if fixed is not None:
                    for k in members:
                        self._name[k] = fixed
                    continue
                base = regs.reg_name(root[0])
                n = seen.get(base, 0) + 1
                seen[base] = n
                nm = base if n == 1 else "%s_%d" % (base, n)
                for k in members:
                    self._name[k] = nm

    def name(self, var):
        return self._name.get(var.key(), str(var))


def _get_const_addr(v, defs):
    if v.is_const:
        val = v.val
        if (val >> 96) > 0:
            return (val >> 96) & 0xFFFF
        return val & 0xFFFF
    if v.is_var and v.key() in defs:
        d = defs[v.key()]
        if d.op in (Op.CONST, Op.MOV) and len(d.srcs) >= 1:
            return _get_const_addr(d.srcs[0], defs)
        if d.op in (Op.LOAD, Op.LOADQ) and len(d.srcs) >= 2:
            return _get_const_addr(d.srcs[1], defs)
    return None


# ---------------------------------------------------------------------------
# inlining
# ---------------------------------------------------------------------------


def _inlinable(func, call_args=None, debris=None):
    """
    Keys of definitions that should print at their use site instead.

    ``call_args`` maps ``id(call_insn)`` to the operands that call will really
    render as arguments.  Those may fold into the call even though its operand
    list is an ABI op, which is what turns three `rN = const` lines plus
    `memcpy()` into `memcpy(0, 0xC720, 0x20)`.

    Constants get a wider licence than anything else: a literal has no
    operands, no side effects and nothing that can change underneath it, and
    SSA dominance guarantees its definition reaches every use.  So it may be
    printed at the use site even when that sits in another block -- which is
    exactly where a format string set up before a label ends up:

        r3 = "ERROR: %s(%d) drift is set";
        loc_28BB8:
            printf(r3, "sceSblSecureClockSrtcWrite1", 0x241);

    The condition is that *every* use renders it as an argument.  A constant
    the return statement also names has to keep its assignment, or the listing
    would use a name it never assigns; so would one passed to a call whose
    arity is unknown, where the argument list is not printed at all and the
    setup line is the only evidence the value was ever produced.
    """
    call_args = call_args or {}
    debris = debris or set()
    rendered = {}                   # key -> the calls that will print it
    for cid, args in call_args.items():
        for a in args:
            rendered.setdefault(a.key(), set()).add(cid)

    defs, use_count, use_site, use_sites = {}, {}, {}, {}
    for insn in func.insns():
        d = insn.defines()
        if d is not None:
            defs[d.key()] = insn
        if d is not None and d.key() in debris:
            continue
        if insn.op == Op.RET:
            real_srcs = [s for s in insn.srcs if s.is_var and s.reg == regs.ARG_FIRST and s.ver != 0]
        elif insn.op in (Op.CALL, Op.ICALL, Op.TCALL):
            n_real = ABI_OPS.get(insn.op, 0)
            real_srcs = [s for s in insn.srcs[:n_real] if s.is_var] + list(call_args.get(id(insn), []))
        else:
            real_srcs = insn.uses()
        for u in real_srcs:
            if not u.is_var:
                continue
            k = u.key()
            use_count[k] = use_count.get(k, 0) + 1
            use_site[k] = insn
            use_sites.setdefault(k, []).append(insn)

    ok = set()
    for key, insn in defs.items():
        is_arg = key in rendered
        # UNDEF stays a statement of its own: folding "<clobbered by call>"
        # into an expression hides the very thing the reader needs to see.
        if insn.op in (Op.PHI, Op.UNDEF) or insn.op.has_side_effects:
            continue
        if insn.op == Op.CONST:
            if is_arg and all(id(s) in rendered[key]
                              for s in use_sites.get(key, ())):
                ok.add(key)
            continue
        if use_count.get(key, 0) != 1:
            continue
        site = use_site[key]
        if site.op == Op.PHI:
            continue
        if site.op in ABI_OPS and not (is_arg and
                                       site.op in (Op.CALL, Op.ICALL, Op.TCALL)):
            continue
        if site.block is not insn.block:
            continue
        if insn.op in MEM_READS:
            # A load may only move down to its use if nothing writes memory in
            # between, or the value printed would be read after the write.
            blk = insn.block
            i0, i1 = blk.insns.index(insn), blk.insns.index(site)
            if any(blk.insns[j].op in MEM_WRITES or
                   blk.insns[j].op.has_side_effects
                   for j in range(i0 + 1, i1)):
                continue
        ok.add(key)

    # -- apply the depth cap here, not while rendering --------------------
    #
    # `operand` only inlines while `depth < MAX_INLINE_DEPTH`; past that it
    # prints the definition's name instead, and `insn_stmt` has already
    # dropped the definition as a statement.  That combination names a
    # variable the listing never computes.  Position in the chain is
    # knowable now, so drop the candidates that would sit too deep and let
    # them stay statements.  Removing one shortens every chain through it,
    # hence the fixpoint.
    while True:
        pos = _chain_positions(ok, defs, use_site)
        too_deep = {k for k, p in pos.items() if p >= MAX_INLINE_DEPTH}
        if not too_deep:
            break
        ok -= too_deep
    return ok, defs


def _chain_positions(ok, defs, use_site):
    """
    How deep each inline candidate would be rendered.

    Zero when its use site is printed as a statement, one more than its
    consumer when the consumer is itself inlined.  SSA plus the single-use
    requirement make the chains acyclic, but the ``seen`` set keeps a cycle
    from recursing forever if either ever stops holding.
    """
    pos = {}

    def walk(key, seen):
        if key in pos:
            return pos[key]
        site = use_site.get(key)
        d = site.defines() if site is not None else None
        if d is None or d.key() not in ok or d.key() in seen:
            pos[key] = 0
        else:
            pos[key] = walk(d.key(), seen | {key}) + 1
        return pos[key]

    for key in ok:
        walk(key, {key})
    return pos


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _is_active_arg(v, defs, params, visited=None):
    if v.ver == 0:
        return v.reg in params
    if defs is None:
        return False
    d = defs.get(v.key())
    if d is None or d.op == Op.UNDEF:
        return False
    if d.op == Op.PHI:
        if visited is None:
            visited = set()
        if v.key() in visited:
            return False
        visited.add(v.key())
        return any(s.is_var and _is_active_arg(s, defs, params, visited) for s in d.srcs)
    return True


class CGen(object):

    def __init__(self, func, name_of=None, arity_of=None, str_of=None,
                 stk_of=None, this_of=None, sym_of=None,
                 user_renames=None, user_types=None, user_proto=None):
        self.func = func
        self.arity_of = arity_of
        self.str_of = str_of or (lambda ea: None)
        self.sym_of = sym_of or (lambda ea: None)
        self.user_renames = dict(user_renames or {})
        self.user_types = dict(user_types or {})
        self.user_proto = user_proto

        self.body_eas = []
        self.body_line_to_eas = {}
        self._current_inlined_eas = []
        self.raw_param_name = {}
        self.all_var_types = {}
        self.sig_params = []
        self.ret_type = "void"
        self.sig_line_idx = None

        # Parameters get positional names (a1, a2, ...) the way a real
        # decompiler shows them; the register each one came from goes in the
        # header comment so the mapping back to the disassembly is not lost.
        self.params = _params(func, arity_of)

        defs_map = {i.defines().key(): i for i in func.insns() if i.defines() is not None}

        # Which operands of each call are really its arguments.
        self.call_args = {}
        for insn in func.insns():
            if insn.op in (Op.CALL, Op.ICALL, Op.TCALL):
                args = self._compute_call_args(insn, arity_of, defs_map)
                if args:
                    self.call_args[id(insn)] = args

        self.call_return = {}
        self._suppressed_call_undefs = set()
        for b in func.blocks:
            for idx, insn in enumerate(b.insns):
                if insn.op in (Op.CALL, Op.ICALL):
                    for other in b.insns[idx + 1:]:
                        if other.ea != insn.ea:
                            break
                        if other.op == Op.UNDEF and other.dst is not None and other.dst.reg == regs.ARG_FIRST:
                            self.call_return[id(insn)] = other.dst
                            self._suppressed_call_undefs.add(id(other))
                            break
        # A non-static member function's r3 is the hidden `this`, named from
        # the class the function belongs to (see data.Members and `_this_reg`).
        self.member_class = this_of(func.start_ea) if this_of is not None \
            else None
        self.this_reg = self._this_reg()
        # A member that reads nothing still has a `this`; `_params` only counts
        # registers with a use, so give it one.  Otherwise r3 already leads the
        # contiguous parameter list.
        if self.this_reg is not None and not self.params:
            self.params = [self.this_reg]
        self.param_name = {}
        pnames = {}
        ai = 0
        if self.user_proto:
            from . import abi
            parsed = abi._parse_proto(self.user_proto, func.name or ("sub_%X" % func.start_ea))
            if parsed and parsed.params:
                for idx, p in enumerate(parsed.params):
                    reg_num = regs.ARG_FIRST + idx
                    if reg_num not in self.params:
                        self.params.append(reg_num)
                    self.raw_param_name[reg_num] = p.name
                    p_eff_name = self.user_renames.get(p.name, p.name)
                    self.param_name[reg_num] = p_eff_name
                    pnames[(reg_num, 0)] = p_eff_name
        for r in self.params:
            if r in self.param_name:
                continue
            if r == self.this_reg:
                nm = "this"
            else:
                ai += 1
                nm = "a%d" % ai
            self.raw_param_name[r] = nm
            eff_nm = self.user_renames.get(nm, nm)
            self.param_name[r] = eff_nm
            pnames[(r, 0)] = eff_nm
        self.namer = Namer(func, pnames, self.call_args)
        self.call_at = {i.ea: i for i in func.insns()
                        if i.op in (Op.CALL, Op.ICALL, Op.TCALL)}
        self.defs = defs_map
        self.debris = self._compute_debris()
        # Argument setup may fold into the call now that the call shows its
        # arguments -- but only for operands that are actually rendered.  A
        # register set before a call the callee never reads must stay a
        # visible statement, or the value would vanish from the listing.
        self.inlinable, self.defs = _inlinable(func, self.call_args, self.debris)
        self.name_of = name_of or (lambda ea: "sub_%X" % ea)
        # Names the rendered body actually mentions; `declarations` declares
        # exactly these, so the two can never disagree.
        self.printed = set()
        # Names that appear as an assignment target, which is what makes a
        # name a local rather than something the function only reads.
        self.assigned = set()
        self.mute = _muted_clobbers(func, self._rendered_keys())
        # The calling convention's own instructions -- saving lr and the
        # callee-saved registers, writing the back chain -- say nothing about
        # what this function does.  `generate` states them in the header
        # instead; see frame.py for what makes a store recognisable as one.
        self.frame = frame.analyse(func)
        # Stack slots by name, from IDA's frame analysis.  Off by default:
        # cgen stays runnable without a database, exactly as with `str_of`.
        self.stk_of = stk_of or (lambda ea: None)
        # Resolved on first use, not here: deciding a slot's C type needs
        # the recovered types, and type recovery runs later in this
        # constructor.
        self._slots = None
        # name -> C type, filled as the body prints them, so `declarations`
        # declares the slots the listing actually mentions and no others.
        self.stack_decls = {}
        # Stack-derived addresses that did not resolve to a fixed slot.  Any
        # of them could alias a named slot, so the header says how many
        # there were rather than the pass pretending there were none.
        self.stack_unresolved = 0
        # Names some visible definition really assigns.  A printed phi can
        # otherwise name a value whose every definition is suppressed -- a
        # colour made only of other phis and muted clobbers -- and the listing
        # then mentions a variable nothing declares or computes.
        self.visible = set()
        for insn in func.insns():
            d = insn.defines()
            if d is None or d.reg in regs.PSEUDO:
                continue
            if not self._suppressed(insn, d):
                self.visible.add(self.namer.name(d))
        # Recover types now rather than earlier in the pipeline: inference has
        # to unify across the namer's webs, since everything printed under one
        # variable name must end up with one type.
        self.types = getattr(func, "types", None)
        if self.types is None:
            from . import lanes
            demand = getattr(func, "demand", None)
            if demand is None:
                from . import param_demand
                demand = lanes.compute_demand(func, callee=param_demand)
            self.types, stats = types.infer(
                func, demand, self.namer.webs(),
                is_string=lambda val: self.str_of(val >> 96) is not None)
            stats["kinds"] = self.types.counts()
            func.types = self.types
            func.type_stats = stats

    def _compute_call_args(self, insn, arity_of, defs=None):
        by_reg = {}
        for s in insn.srcs:
            if s.is_var and regs.ARG_FIRST <= s.reg <= regs.ARG_LAST:
                by_reg[s.reg] = s
        if not by_reg:
            return []

        callee_var = False
        if insn.op in (Op.CALL, Op.TCALL) and insn.aux is not None:
            try:
                from . import is_variadic
                callee_var = is_variadic(insn.aux)
            except Exception:
                pass

        n = None
        if insn.op in (Op.CALL, Op.TCALL) and insn.aux is not None and arity_of is not None:
            n = arity_of(insn.aux)

        if not callee_var and n is not None and 0 < n <= len(by_reg):
            out = []
            for i in range(n):
                r = regs.ARG_FIRST + i
                if r in by_reg:
                    out.append(by_reg[r])
                else:
                    break
            return out
        if n == 0:
            return []
        if insn.op == Op.ICALL and len(insn.srcs) > 0 and insn.srcs[0].is_var and insn.srcs[0].reg == regs.ARG_FIRST:
            return []

        # Variadic, unknown arity, or ICALL: determine from active setup
        out = []
        for i in range(len(by_reg)):
            r = regs.ARG_FIRST + i
            v = by_reg.get(r)
            if v is None:
                break
            if _is_active_arg(v, defs, self.params):
                out.append(v)
            else:
                break
        return out

    # -- operands ----------------------------------------------------------

    def _rendered_keys(self):
        """
        Keys whose *name* the body will print even though nothing in this
        function reads them as code: a call's rendered arguments, and the
        value the return statement forwards.  A clobber among them must keep
        its assignment or the listing names a variable it never sets.
        """
        keys = set()
        for args in self.call_args.values():
            for a in args:
                if a.is_var:
                    keys.add(a.key())
        return keys

    def nm(self, v):
        """The printed name of a value, recording that it was printed."""
        name = self.namer.name(v)
        self.printed.add(name)
        return self.user_renames.get(name, name)

    def nma(self, v):
        """
        The printed name of an *assignment target*.

        Kept apart from :meth:`nm` because a phi web shares one name across
        many SSA values: a callee-saved register's name is printed by the
        prologue save, which reads the incoming value, while the definition
        that would assign it -- the epilogue restore -- prints nothing.  Only
        a name that really gets assigned somewhere is a local.
        """
        name = self.namer.name(v)
        self.printed.add(name)
        self.assigned.add(name)
        return self.user_renames.get(name, name)


    def const(self, c, scalar, strings=True):
        """
        A constant, as a string literal when its preferred slot points at one.

        ``strings=False`` is for a position where a literal would be actively
        misleading rather than helpful -- the destination of a store, where
        `*(char *)"text" = x` reads as writing to a string constant when the
        code is really writing into a buffer that currently holds text.
        """
        if strings:
            lit = self.str_of(c.val >> 96)
            if lit is not None:
                return lit
        sym = self.sym_of(c.val >> 96)
        if sym is not None:
            if " + " in sym:
                base, off = sym.split(" + ", 1)
                eff_base = self.user_renames.get(base, base)
                return "&" + eff_base + " + " + off
            eff_sym = self.user_renames.get(sym, sym)
            return "&" + eff_sym
        if c.val == 0:
            return "0"
        for cand in (EW.W, EW.H, EW.B, EW.D):
            bits = int(cand) * 8
            n = 16 // int(cand)
            lane = c.val >> (128 - bits)
            if all(((c.val >> (128 - bits * (i + 1))) & ((1 << bits) - 1)) == lane
                   for i in range(n)):
                if cand == EW.W:
                    if lane >= 0x80000000 and (0x100000000 - lane) <= 0xFFFF:
                        return "-0x%X" % (0x100000000 - lane)
                    return "0x%X" % lane
                elif cand == EW.D:
                    if lane >= 0x8000000000000000 and (0x10000000000000000 - lane) <= 0xFFFF:
                        return "-0x%X" % (0x10000000000000000 - lane)
                    return "0x%X" % lane
                else:
                    return "0x%X" % lane
        if scalar:
            return c.scalar_str()
        if (c.val & ((1 << 96) - 1)) == 0:
            return c.scalar_str()
        return "0x%032X" % c.val

    def operand(self, v, depth=0, scalar=True, top=False, want_scalar=False,
                strings=True):
        """
        ``top`` means the result already sits in a context that brackets it --
        the inside of a `*(u32 *)(...)`, or the whole right-hand side of an
        assignment -- so an infix expression needs no parentheses of its own.

        ``want_scalar`` says the consuming context reads the preferred slot,
        so a constant folded in here should be spelled as a scalar even if the
        lifter could not prove the definition was one.  Call arguments and
        addresses are both like that.
        """
        if v.is_const:
            return self.const(v, scalar, strings)
        if depth < MAX_INLINE_DEPTH and v.key() in self.inlinable:
            def_insn = self.defs[v.key()]
            if getattr(def_insn, "ea", None) is not None:
                self._current_inlined_eas.append(def_insn.ea)
            return self.rhs(def_insn, depth + 1, top=top,
                            want_scalar=want_scalar, strings=strings)
        return self.nm(v)

    def _stack_indexed_addr(self, v, depth=0):
        if self._slots is None:
            self._slots = self._slot_types()
        if not self._slots:
            return None
        from .scalarize import addr_expr, word0

        cur = v
        if cur.is_var and cur.key() in self.defs:
            d = self.defs[cur.key()]
            if (d.op == Op.AND and len(d.srcs) == 2 and d.srcs[1].is_const
                    and word0(d.srcs[1].val) in (0x3FFF0, 0x3FFFF)):
                cur = d.srcs[0]

        if not (cur.is_var and cur.key() in self.defs):
            return None
        d = self.defs[cur.key()]
        if d.op != Op.ADD or d.ew != EW.W or len(d.srcs) != 2:
            return None

        sp_in = (regs.SP, 0)
        s1, s2 = d.srcs
        for sp_term, idx_term in [(s1, s2), (s2, s1)]:
            b_sp, off_sp, _ = addr_expr(sp_term, self.defs)
            if b_sp == sp_in:
                b_idx, off_idx, _ = addr_expr(idx_term, self.defs)
                if b_idx is not None and b_idx != sp_in:
                    tot_off = (off_sp + off_idx) & 0xFFFFFFFF
                    slot_base = tot_off & ~0xF
                    byte_off = tot_off & 0xF
                    hit = self._slots.get((b_sp, slot_base))
                    if hit is None:
                        hit = self._slots.get((b_sp, tot_off))
                        byte_off = 0
                    if hit is None:
                        from . import frame
                        signed_off = frame.signed_offset(slot_base)
                        if signed_off != 0 and -0x4000 <= signed_off <= 0x200:
                            name = "var_%X" % abs(signed_off) if signed_off < 0 else "arg_%X" % signed_off
                            hit = (name, "qword", EW.Q)
                            self._slots[(b_sp, slot_base)] = hit
                    if hit is not None:
                        self.stack_decls[hit[0]] = hit[1]
                        slot_name = self.user_renames.get(hit[0], hit[0])
                        idx_var = Var(b_idx[0], b_idx[1])
                        idx_str = self.operand(idx_var, depth, True, want_scalar=True)
                        if byte_off == 0:
                            return "(char *)&%s + %s" % (slot_name, idx_str)
                        else:
                            return "(char *)&%s + %s + 0x%X" % (slot_name, idx_str, byte_off)
        return None

    def addr(self, v, depth=0, strings=True):
        stk_idx = self._stack_indexed_addr(v, depth)
        if stk_idx is not None:
            return stk_idx
        if v.is_const:
            if strings:
                lit = self.str_of(v.val >> 96)
                if lit is not None:
                    return lit
            sym = self.sym_of(v.val >> 96)
            if sym is not None:
                if " + " in sym:
                    base, off = sym.split(" + ", 1)
                    eff_base = self.user_renames.get(base, base)
                    return "&" + eff_base + " + " + off
                eff_sym = self.user_renames.get(sym, sym)
                return "&" + eff_sym
            return "0x%X" % (v.val >> 96)
        return self.operand(v, depth, top=True, want_scalar=True,
                            strings=strings)

    # -- right-hand sides --------------------------------------------------

    def rhs(self, insn, depth=0, top=True, want_scalar=False, strings=True):
        op = insn.op
        sc = insn.scalar

        d = insn.defines()
        if d is not None and hasattr(self, 'sbox_trees') and d.key() in self.sbox_trees:
            inp_var, is_inv, tree_insns, mask_insn = self.sbox_trees[d.key()]
            op_name = "aes_inv_sub_bytes" if is_inv else "aes_sub_bytes"
            inp_str = self.operand(inp_var, depth, False)
            return "%s(%s)" % (op_name, inp_str)

        # An address computed into a named slot is that slot's address.  This
        # is what connects the two halves of the common idiom: a pointer into
        # the frame handed to a callee, and the slot read back afterwards.
        # Without it the listing showed `r7 = (char *)sp_2 + 0x20;` and later
        # `r8 = shufb(r8, var_130, r10);` with nothing to say those are the
        # same sixteen bytes.
        if op == Op.ADD and insn.ew == EW.W:
            slot = self._addr_of_slot(insn)
            if slot is not None:
                return slot

        if op == Op.CONST:
            # A constant assigned to something the types say is a scalar
            # prints as a scalar, even when the lifter could not prove it was
            # one: `0x3e000` rather than `#0x3e000:w4`.
            if insn.dst is not None and insn.dst.reg == regs.ARG_FIRST:
                sc = True
            t = self.type_of(insn.dst) if insn.dst is not None else None
            if not sc and t is not None and t.kind in (types.INT, types.PTR,
                                                       types.UNK) \
                    and t.width in (1, 2, 4):
                sc = True
            # A consumer that reads only the preferred slot settles it too,
            # unless the type positively says this value is a vector.
            if not sc and want_scalar and (t is None or t.kind != types.VEC):
                sc = True
            return self.const(insn.srcs[0], sc, strings)
        if op == Op.MOV:
            return self.operand(insn.srcs[0], depth, sc, top=top,
                                strings=strings)
        if op == Op.UNDEF:
            return "<clobbered by call>"
        if op == Op.PHI:
            return "phi(%s)" % ", ".join(self._phi_arg(s, depth, sc)
                                         for s in insn.srcs)
        if op in (Op.LOAD, Op.LOADQ, Op.LOADU):
            return self._deref(insn, depth)
        if op in (Op.CALL, Op.ICALL, Op.TCALL):
            if op in (Op.CALL, Op.TCALL):
                raw_tgt = self.name_of(insn.aux)
                tgt = self.user_renames.get(raw_tgt, raw_tgt)
            else:
                tgt = "(*%s)" % self.operand(insn.srcs[0], depth)
            args = self.call_args.get(id(insn))
            if not args:
                return "%s()" % tgt
            return "%s(%s)" % (tgt, ", ".join(
                self.operand(a, depth, True, top=True, want_scalar=True)
                for a in args))
        if op == Op.RDCH:
            return "rdch(%s)" % channels.label(insn.aux)
        if op == Op.RCHCNT:
            return "rchcnt(%s)" % channels.label(insn.aux)
        if op == Op.MFSPR:
            return "mfspr(%s)" % insn.aux
        if op == Op.INTRINSIC:
            args = ", ".join(self.operand(s, depth, False) for s in insn.srcs)
            return "%s(%s)" % (insn.aux, args)

        # Bitwise operations are lane-independent: byte i of the result
        # depends only on byte i of the inputs.  So `&`, `|`, `^` mean exactly
        # the same thing whether the value is a scalar or a vector, and there
        # is no element width to lose by writing them infix -- unlike `add.w`
        # versus `add.b`, which are different operations on the same 128 bits.
        #
        # (`&&` would be wrong here: C's logical AND yields 0 or 1 and
        # short-circuits, where this masks bits.)
        if op in _BITWISE and len(insn.srcs) == 2:
            wide = self._scalarish(insn)
            fmt, wrapped = _BITWISE[op]
            text = fmt % (self.operand(insn.srcs[0], depth, wide),
                          self.operand(insn.srcs[1], depth, wide))
            return text if (top or wrapped) else "(%s)" % text
        if op == Op.NOT and len(insn.srcs) == 1:
            return "~%s" % self.operand(insn.srcs[0], depth,
                                        self._scalarish(insn))

        # Element-wise arithmetic only reads infix when the width is not in
        # doubt.  Three ways for it not to be:
        #
        #   * scalarisation proved the value lives in the preferred slot
        #     (`sc` is already set);
        #   * its type is a vector whose element width matches the operation,
        #     so `a + b` on a `vec_uint4` really is the 32-bit lane-wise add
        #     the opcode performs;
        #   * its type occupies no more than one element -- an address, or
        #     anything demand analysis narrowed to a word or less.  Then one
        #     lane is all there is, and `sp - 0x13B0` says what `add.w(sp,
        #     #0xffffec50:w4)` says, only legibly.  Stack-frame arithmetic is
        #     the commonest arithmetic in this code, and it all lands here.
        #
        # The third case is the weakest of the three and worth being explicit
        # about.  A PTR type comes from a real use as a load or store address,
        # which reads the preferred slot -- but the same register can *also*
        # have a genuinely wide use, and the SPU ABI hands it one: the
        # prologue stores the old stack pointer as a full-quadword back
        # chain.  Demand then stays ALL while the type still says address, and
        # no single spelling of the definition is right for both consumers.
        # The address reading is the one a reader of stack arithmetic wants,
        # so that is what this prints.  With the cross-procedural demand in
        # place the first case already covers most of it; this catches the
        # remainder (117 `add.w` calls in a 264-function module).
        if not sc and op in INFIX and len(insn.srcs) == 2 and insn.ew != EW.Q:
            d = insn.defines()
            t = self.type_of(d) if d is not None else None
            if t is not None and t.kind == types.VEC and t.width == int(insn.ew):
                sc = True
            elif (t is not None and op not in _MASK_OPS
                    and self._scalarish(insn)
                    and 0 < t.width <= int(insn.ew)):
                sc = True

        if sc and op in INFIX and len(insn.srcs) == 2:
            lhs = self.operand(insn.srcs[0], depth, True)
            rhs = self.operand(insn.srcs[1], depth, True)
            if op in (Op.ADD, Op.SUB):
                lhs = self._byte_ptr(insn.srcs[0], lhs)
                rhs = self._byte_ptr(insn.srcs[1], rhs)
            sym = INFIX[op]
            # `ai rX, rX, -1` is everywhere; print it as subtraction rather
            # than as an addition of a negative literal.
            if op == Op.ADD and insn.srcs[1].is_const and rhs.startswith("-"):
                sym, rhs = "-", rhs[1:]
            text = "%s %s %s" % (lhs, sym, rhs)
            return text if top else "(%s)" % text

        if op in (Op.ADDX, Op.SUBX) and len(insn.srcs) >= 2:
            lhs = self.operand(insn.srcs[0], depth, True)
            rhs = self.operand(insn.srcs[1], depth, True)
            lhs = self._byte_ptr(insn.srcs[0], lhs)
            rhs = self._byte_ptr(insn.srcs[1], rhs)
            sym = "+" if op == Op.ADDX else "-"
            if op == Op.ADDX and insn.srcs[1].is_const and rhs.startswith("-"):
                sym, rhs = "-", rhs[1:]
            text = "%s %s %s" % (lhs, sym, rhs)
            return text if top else "(%s)" % text

        if op == Op.EXTS and len(insn.srcs) == 1:
            arg = self.operand(insn.srcs[0], depth, sc)
            if not arg.isidentifier() and not (arg.startswith("(") and arg.endswith(")")):
                arg = "(%s)" % arg
            if insn.ew == EW.H:
                return "(short)(char)%s" % arg
            if insn.ew == EW.W:
                return "(int)(short)%s" % arg
            if insn.ew == EW.D:
                return "(long long)(int)%s" % arg

        if op == Op.SELB and len(insn.srcs) == 3:
            rc_var = insn.srcs[2]
            if rc_var.is_var and rc_var.key() in self.defs:
                d = self.defs[rc_var.key()]
                if d.op == Op.FSM and len(d.srcs) == 1:
                    rc_var = d.srcs[0]
            ra = self.operand(insn.srcs[0], depth, sc, want_scalar=True)
            rb = self.operand(insn.srcs[1], depth, sc, want_scalar=True)
            rc = self.operand(rc_var, depth, sc, want_scalar=True)
            cond = rc.strip()
            if not (cond.startswith("(") and cond.endswith(")")) and (" " in cond):
                cond = "(%s)" % cond
            text = "%s ? %s : %s" % (cond, rb, ra)
            return text if top else "(%s)" % text

        if op == Op.FSM and len(insn.srcs) == 1:
            arg = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            if not arg.isidentifier() and not (arg.startswith("(") and arg.endswith(")")):
                arg = "(%s)" % arg
            text = "%s ? -1 : 0" % arg
            return text if top else "(%s)" % text

        if op in (Op.CMPEQ, Op.CMPGT, Op.CMPGTU) and len(insn.srcs) == 2:
            if op in (Op.CMPGT, Op.CMPGTU) and insn.srcs[1].is_const and (insn.srcs[1].val >> 96) in (0xB, 0xF):
                lhs_op = insn.srcs[0]
                if lhs_op.is_var and lhs_op.key() in self.defs:
                    gb_insn = self.defs[lhs_op.key()]
                    if gb_insn.op == Op.GB and len(gb_insn.srcs) == 1:
                        cmp_op = gb_insn.srcs[0]
                        if cmp_op.is_var and cmp_op.key() in self.defs:
                            cmp_insn = self.defs[cmp_op.key()]
                            if cmp_insn.op in (Op.CMPEQ, Op.FCMPEQ) and len(cmp_insn.srcs) == 2:
                                lhs = self.operand(cmp_insn.srcs[0], depth, True, want_scalar=True)
                                rhs = self.operand(cmp_insn.srcs[1], depth, True, want_scalar=True)
                                text = "%s == %s" % (lhs, rhs)
                                return text if top else "(%s)" % text
            lhs = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            rhs = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            sym = "==" if op == Op.CMPEQ else ">"
            text = "%s %s %s" % (lhs, sym, rhs)
            return text if top else "(%s)" % text

        if op == Op.GB and len(insn.srcs) == 1:
            arg = insn.srcs[0]
            if arg.is_var and arg.key() in self.defs:
                d = self.defs[arg.key()]
                if d.op in (Op.CMPEQ, Op.FCMPEQ, Op.CMPGT, Op.CMPGTU):
                    return self.rhs(d, depth, top=top, want_scalar=True)

        if op in (Op.QSHLBY, Op.QROTMBY, Op.QSHLBI, Op.QROTMBI) and len(insn.srcs) == 2:
            val_str = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
            if insn.srcs[1].is_const:
                cnt = insn.srcs[1].val >> 96
                if op == Op.QSHLBY:
                    if cnt == 4:
                        if not val_str.isidentifier() and not (val_str.startswith("(") and val_str.endswith(")")):
                            val_str = "(%s)" % val_str
                        return "(u32)%s" % val_str
                    else:
                        bits = cnt * 8
                        text = "%s << %d" % (val_str, bits)
                        return text if top else "(%s)" % text
                elif op == Op.QROTMBY:
                    k = (-cnt) & 0x1F
                    bits = k * 8
                    text = "%s >> %d" % (val_str, bits)
                    return text if top else "(%s)" % text
                elif op == Op.QSHLBI:
                    text = "%s << %d" % (val_str, cnt)
                    return text if top else "(%s)" % text
                elif op == Op.QROTMBI:
                    text = "%s >> %d" % (val_str, cnt)
                    return text if top else "(%s)" % text
            else:
                cnt_str = self.operand(insn.srcs[1], depth, True, want_scalar=True)
                if op == Op.QSHLBY:
                    text = "%s << (%s * 8)" % (val_str, cnt_str)
                elif op == Op.QROTMBY:
                    text = "%s >> (%s * 8)" % (val_str, cnt_str)
                elif op == Op.QSHLBI:
                    text = "%s << %s" % (val_str, cnt_str)
                elif op == Op.QROTMBI:
                    text = "%s >> %s" % (val_str, cnt_str)
                return text if top else "(%s)" % text

        if op == Op.SHUFB and len(insn.srcs) == 3:
            mask_op = insn.srcs[2]
            addr = _get_const_addr(mask_op, self.defs)
            mask_val = mask_op.val if mask_op.is_const else None
            mask_w0 = (mask_val & 0xFFFFFFFF) if mask_val is not None else None
            mask_str = self.operand(mask_op, depth, False)

            # RotWord masks (32-bit rotate left by 8)
            if mask_w0 in (0x15161714, 0x1D1E1F1C):
                rb_str = self.operand(insn.srcs[1], depth, False, want_scalar=True)
                return "_rotl(%s, 8)" % rb_str
            if mask_w0 == 0x0D0E0F0C:
                ra_str = self.operand(insn.srcs[0], depth, False, want_scalar=True)
                return "_rotl(%s, 8)" % ra_str

            # Word extraction masks
            if mask_w0 == 0x1C1D1E1F:
                rb_str = self.operand(insn.srcs[1], depth, False, want_scalar=True)
                return "(u32)%s" % rb_str
            if mask_w0 in (0x00010203, ) or "10203" in mask_str:
                return self.operand(insn.srcs[0], depth, sc, want_scalar=True)
            if mask_w0 == 0x04050607 or "4050607" in mask_str:
                ra_str = self.operand(insn.srcs[0], depth, False)
                return "%s[1]" % ra_str
            if mask_w0 == 0x08090A0B or "8090a0b" in mask_str.lower():
                ra_str = self.operand(insn.srcs[0], depth, False)
                return "%s[2]" % ra_str
            if mask_w0 == 0x0C0D0E0F or "c0d0e0f" in mask_str.lower():
                ra_str = self.operand(insn.srcs[0], depth, False)
                return "%s[3]" % ra_str

            # Byte splat mask (e.g. memset)
            if mask_w0 == 0x03030303 or "3030303" in mask_str:
                ra_str = self.operand(insn.srcs[0], depth, True, want_scalar=True)
                return "spu_splats((u8)%s)" % ra_str

            # Known table addresses / crypto constants
            if addr is not None:
                a_str = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
                b_str = self.operand(insn.srcs[1], depth, sc, want_scalar=False)
                if addr == 0xCAE0:
                    if "DecKey" in getattr(self.func, "name", ""):
                        return "aes_inv_mix_columns(%s)" % a_str
                    return "aes_shift_rows(%s)" % a_str
                elif addr == 0xCAF0:
                    return "aes_mix_cols_rot(%s, 1)" % a_str
                elif addr == 0xCB00:
                    return "aes_mix_cols_rot(%s, 2)" % a_str
                elif addr == 0xCB10:
                    return "aes_mix_cols_rot(%s, 3)" % a_str
                elif addr == 0xCC30:
                    if "DecKey" in getattr(self.func, "name", ""):
                        return "aes_mix_columns(%s)" % a_str
                    return "aes_inv_shift_rows(%s)" % a_str
                elif addr == 0xCC40:
                    return "aes_inv_mix_cols_rot(%s, 1)" % a_str
                elif addr == 0xCC50:
                    return "aes_inv_mix_cols_rot(%s, 2)" % a_str
                elif addr == 0xCC60:
                    return "aes_inv_mix_cols_rot(%s, 3)" % a_str
                elif addr == 0xCB20:
                    m_str = self.operand(insn.srcs[2], depth, False)
                    return "aes_xtimes_table(%s)" % m_str
                elif addr == 0xCC70:
                    m_str = self.operand(insn.srcs[2], depth, False)
                    return "aes_xtimes2_table(%s)" % m_str
                elif addr == 0xCC80:
                    m_str = self.operand(insn.srcs[2], depth, False)
                    return "aes_xtimes3_table(%s)" % m_str
                elif addr == 0xCCF0:
                    text = "(%s << 32) | (%s >> 96)" % (a_str, b_str)
                    return text if top else "(%s)" % text
                elif addr == 0xCAD0:
                    text = "(%s << 64) | (%s >> 64)" % (a_str, b_str)
                    return text if top else "(%s)" % text
                elif addr == 0xCAC0:
                    text = "((u64)%s << 64) | (u64)%s" % (a_str, b_str)
                    return text if top else "(%s)" % text
                elif addr == 0xCA80:
                    text = "%s << 32" % a_str
                    return text if top else "(%s)" % text
                elif addr == 0xCA90:
                    text = "%s << 64" % a_str
                    return text if top else "(%s)" % text
                elif addr == 0xCAA0:
                    text = "%s >> 32" % a_str
                    return text if top else "(%s)" % text
                elif addr == 0xCAB0:
                    text = "%s >> 64" % a_str
                    return text if top else "(%s)" % text

            if ("xmmword_232A0" in mask_str or "xmmword_232B0" in mask_str) or (
                    insn.srcs[0].is_var and insn.srcs[1].is_var and insn.srcs[0].key() == insn.srcs[1].key()
                    and ("232A0" in mask_str or "232B0" in mask_str)):
                ra_str = self.operand(insn.srcs[0], depth, sc, want_scalar=True)
                if not ra_str.isidentifier() and not (ra_str.startswith("(") and ra_str.endswith(")")):
                    ra_str = "(%s)" % ra_str
                return "(u32)%s" % ra_str
            if "233B0" in mask_str:
                ra_str = self.operand(insn.srcs[0], depth, sc, want_scalar=True)
                return "__builtin_bswap32(%s)" % ra_str
            if "239D0" in mask_str:
                a_str = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
                b_str = self.operand(insn.srcs[1], depth, sc, want_scalar=False)
                text = "(%s << 32) | (%s >> 96)" % (a_str, b_str)
                return text if top else "(%s)" % text
            if "237B0" in mask_str or "239B0" in mask_str:
                a_str = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
                b_str = self.operand(insn.srcs[1], depth, sc, want_scalar=False)
                text = "(%s << 64) | (%s >> 64)" % (a_str, b_str)
                return text if top else "(%s)" % text
            if any(k in mask_str for k in ("237C0", "237D0", "237E0", "237F0")):
                a_str = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
                return "aes_shift_rows(%s)" % a_str
            if any(k in mask_str for k in ("23910", "23920", "23930", "23940")):
                a_str = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
                return "aes_inv_shift_rows(%s)" % a_str
            if any(k in mask_str for k in ("CAE0", "cae0")):
                a_str = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
                return "aes_inv_mix_columns(%s)" % a_str
            if insn.srcs[2].is_var and insn.srcs[2].key() in self.defs:
                c_insn = self.defs[insn.srcs[2].key()]
                if c_insn.op == Op.GENCTL:
                    val_s = self.operand(insn.srcs[0], depth, sc, want_scalar=True)
                    old_s = self.operand(insn.srcs[1], depth, sc, want_scalar=False)
                    from .scalarize import addr_expr, word0, _slot_of
                    ca = addr_expr(c_insn.srcs[0], self.defs)
                    slot = _slot_of(ca)
                    if slot is not None:
                        if c_insn.ew == EW.W:
                            idx = slot // 4
                            if old_s == "0":
                                if idx == 0:
                                    return val_s
                                return "(vec_uint4){ [3] = %s }" % val_s if idx == 3 else "spu_insert(%s, (vec_uint4){0}, %d)" % (val_s, idx)
                            return "spu_insert(%s, %s, %d)" % (val_s, old_s, idx)
                        elif c_insn.ew == EW.H:
                            idx = slot // 2
                            if old_s == "0" and idx == 0:
                                return val_s
                            return "spu_insert(%s, %s, %d)" % (val_s, old_s, idx)
                        else:
                            if old_s == "0" and slot == 0:
                                return val_s
                            return "spu_insert(%s, %s, %d)" % (val_s, old_s, slot)
                    c_addr = self.operand(c_insn.srcs[0], depth, True, want_scalar=True)
                    if c_insn.ew == EW.W:
                        return "spu_insert(%s, %s, (%s & 0xF) / 4)" % (val_s, old_s, c_addr)
                    elif c_insn.ew == EW.H:
                        return "spu_insert(%s, %s, (%s & 0xF) / 2)" % (val_s, old_s, c_addr)
                    else:
                        return "spu_insert(%s, %s, %s & 0xF)" % (val_s, old_s, c_addr)
            a_str = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
            b_str = self.operand(insn.srcs[1], depth, sc, want_scalar=False)
            m_str = self.operand(insn.srcs[2], depth, False)
            return "spu_shuffle(%s, %s, %s)" % (a_str, b_str, m_str)

        if op == Op.CLZ and len(insn.srcs) >= 1:
            arg = self.operand(insn.srcs[0], depth, sc, want_scalar=True)
            return "__builtin_clz(%s)" % arg

        if op == Op.CG and len(insn.srcs) == 2:
            lhs = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            rhs = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            text = "((u64)(u32)%s + (u32)%s > 0xFFFFFFFF)" % (lhs, rhs)
            return text if top else "(%s)" % text

        if op == Op.BG and len(insn.srcs) == 2:
            lhs = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            rhs = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            text = "(u32)%s < (u32)%s" % (lhs, rhs)
            return text if top else "(%s)" % text

        if op == Op.CGX and len(insn.srcs) == 3:
            lhs = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            rhs = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            c = self.operand(insn.srcs[2], depth, True, want_scalar=True)
            text = "((u64)(u32)%s + (u32)%s + (%s & 1) > 0xFFFFFFFF)" % (lhs, rhs, c)
            return text if top else "(%s)" % text

        if op == Op.BGX and len(insn.srcs) == 3:
            lhs = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            rhs = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            c = self.operand(insn.srcs[2], depth, True, want_scalar=True)
            text = "((u64)(u32)%s < (u32)%s + (%s & 1))" % (lhs, rhs, c)
            return text if top else "(%s)" % text

        if op in (Op.ROTM, Op.ROTMA, Op.ROL) and len(insn.srcs) == 2:
            val = self.operand(insn.srcs[0], depth, sc, want_scalar=True)
            if not val.isidentifier() and not (val.startswith("(") and val.endswith(")")):
                val = "(%s)" % val
            if insn.srcs[1].is_const:
                cnt_val = (insn.srcs[1].val >> 96) & 0xFFFFFFFF
                k = cnt_val if cnt_val < 0x80000000 else cnt_val - 0x100000000
                if k <= 0:
                    shift = (-k) & 31
                    if op == Op.ROTMA:
                        text = "(int)%s >> %d" % (val, shift)
                    else:
                        text = "%s >> %d" % (val, shift)
                else:
                    text = "_rotl(%s, %d)" % (val, k & 31)
                return text if top else "(%s)" % text
            else:
                cnt_str = self.operand(insn.srcs[1], depth, True, want_scalar=True)
                if cnt_str.startswith("(0 - ") and cnt_str.endswith(")"):
                    inner = cnt_str[5:-1].strip()
                    if op == Op.ROTMA:
                        text = "(int)%s >> %s" % (val, inner)
                    else:
                        text = "%s >> %s" % (val, inner)
                    return text if top else "(%s)" % text
                elif cnt_str.startswith("-"):
                    inner = cnt_str[1:].strip()
                    if op == Op.ROTMA:
                        text = "(int)%s >> %s" % (val, inner)
                    else:
                        text = "%s >> %s" % (val, inner)
                    return text if top else "(%s)" % text
                if op == Op.ROL:
                    text = "_rotl(%s, %s)" % (val, cnt_str)
                elif op == Op.ROTMA:
                    text = "(int)%s >> %s" % (val, cnt_str)
                else:
                    text = "%s >> %s" % (val, cnt_str)
                return text if top else "(%s)" % text

        if op == Op.QROTBI and len(insn.srcs) == 2:
            val = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
            if insn.srcs[1].is_const:
                cnt = (insn.srcs[1].val >> 96) & 0x7F
                text = "_rotl(%s, %d)" % (val, cnt)
            else:
                cnt_str = self.operand(insn.srcs[1], depth, True, want_scalar=True)
                text = "_rotl(%s, %s)" % (val, cnt_str)
            return text if top else "(%s)" % text

        if op == Op.QROTBY and len(insn.srcs) == 2:
            val = self.operand(insn.srcs[0], depth, sc, want_scalar=False)
            if insn.srcs[1].is_const:
                cnt = (insn.srcs[1].val >> 96) & 0xF
                bits = cnt * 8
                text = "_rotl(%s, %d)" % (val, bits)
            else:
                cnt_str = self.operand(insn.srcs[1], depth, True, want_scalar=True)
                text = "_rotl(%s, (%s) * 8)" % (val, cnt_str)
            return text if top else "(%s)" % text

        if op == Op.MPYH and len(insn.srcs) == 2:
            a = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            b = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            text = "((u32)(%s >> 16) * (u16)%s) << 16" % (a, b)
            return text if top else "(%s)" % text

        if op == Op.MPYHHU and len(insn.srcs) == 2:
            a = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            b = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            text = "((u32)%s >> 16) * ((u32)%s >> 16)" % (a, b)
            return text if top else "(%s)" % text

        if op == Op.MPYHH and len(insn.srcs) == 2:
            a = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            b = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            text = "((int)%s >> 16) * ((int)%s >> 16)" % (a, b)
            return text if top else "(%s)" % text

        if op == Op.MPYU and len(insn.srcs) == 2:
            a = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            b = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            text = "(u16)%s * (u16)%s" % (a, b)
            return text if top else "(%s)" % text

        if op in (Op.MPY, Op.MUL) and len(insn.srcs) == 2:
            a = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            b = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            if op == Op.MUL or insn.ew == EW.W:
                text = "%s * %s" % (a, b)
            else:
                text = "(short)%s * (short)%s" % (a, b)
            return text if top else "(%s)" % text

        if insn.ew == EW.H and op in (Op.ADD, Op.SUB, Op.SHL, Op.SHR, Op.SAR) and len(insn.srcs) == 2:
            a = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            b = self.operand(insn.srcs[1], depth, True, want_scalar=True)
            sym = {Op.ADD: "+", Op.SUB: "-", Op.SHL: "<<", Op.SHR: ">>", Op.SAR: ">>"}[op]
            cast = "(short)" if op == Op.SAR else "(u16)"
            text = "%s(%s %s %s)" % (cast, a, sym, b)
            return text if top else "(%s)" % text

        if op == Op.GENCTL and len(insn.srcs) >= 1:
            addr_str = self.operand(insn.srcs[0], depth, True, want_scalar=True)
            return "__builtin_spu_mask(%s)" % addr_str

        name = OP_NAME.get(op, "op%d" % int(op))
        if op in _EW_CALLS:
            name += "." + EW_SUFFIX[insn.ew]
        # A shift or rotate count is read from the preferred slot only, so it
        # is a number -- `qrotmby(r2, 4)`.  Printing the operand as the
        # quadword it technically is gave
        # `qrotmby(r2, #0x00000004000000000000000000000000)`, which buries a 4
        # in thirty-one zeroes and reads as though the whole register mattered.
        args = ", ".join(
            self.operand(s, depth, sc or (op in _COUNT_OPS and i == 1),
                         want_scalar=(op in _COUNT_OPS and i == 1))
            for i, s in enumerate(insn.srcs))
        if insn.aux is not None and op not in (Op.CALL,):
            args = args + (", " if args else "") + "/*%s*/" % (insn.aux,)
        return "%s(%s)" % (name, args)

    # -- typed rendering ---------------------------------------------------

    def type_of(self, v):
        return self.types.of_var(v) if self.types is not None else None

    def _byte_ptr(self, v, text):
        """
        Cast a pointer operand of arithmetic to `char *`.

        SPU address arithmetic is in bytes and C pointer arithmetic scales, so
        `sp - 0x13B0` on an `unsigned int *` would read as 0x4EC0 bytes -- not
        what the instruction does.  The cast is the only thing keeping the
        printed expression equal to the machine's.  An operand already
        rendered as a byte cast, or already pointing at bytes, is left alone.
        """
        t = self.type_of(v)
        if t is None or t.kind != types.PTR:
            return text
        if t.pointee is not None and t.pointee.width == 1:
            return text
        if text.startswith("(char *)"):
            return text
        return "(char *)%s" % (text if text.isidentifier()
                               else "(%s)" % text)

    def _scalarish(self, insn):
        """
        Should this result's constants be spelled as scalars?

        True when scalarisation proved it lives in the preferred slot, or when
        the recovered type is a scalar of at most word width.  A genuine
        vector keeps the lane spelling, because `#0x20000:w4` says something
        `0x20000` does not.
        """
        if insn.scalar:
            return True
        d = insn.defines()
        t = self.type_of(d) if d is not None else None
        return (t is not None
                and t.kind in (types.INT, types.PTR, types.UNK)
                and 0 < t.width <= 4)

    def ctype(self, ty, fallback="qword"):
        """
        The fallback is for types that say nothing at all.  A kind of UNK with
        a known width still says the width, and Ty renders that as an integer.
        """
        if ty is None:
            return fallback
        if ty.kind == types.UNK and ty.width not in (1, 2, 4, 8):
            return fallback
        return str(ty)

    def _access_type(self, insn):
        """The C type a load/store moves."""
        if insn.op in (Op.LOADQ, Op.STOREQ):
            return "qword"
        if insn.op == Op.LOADU:
            return "qword_unaligned"
        # Prefer the recovered element type over the bare width: it carries
        # signedness, and float where the ISA proved it.
        ty = None
        if insn.op == Op.LOAD:
            ty = self.type_of(insn.dst) if insn.dst is not None else None
        else:
            ty = self.type_of(insn.srcs[2])
        if ty is not None and ty.kind in (types.INT, types.FLT) \
                and ty.width == int(insn.ew):
            return str(ty)
        return CTYPE[insn.ew]

    def _slot_types(self):
        """
        Which stack slots may print as a name, what it is, and its C type.

        The division of labour matters.  *Identity* comes from this
        decompiler: :func:`scalarize.addr_expr` resolves an address to
        ``(base, offset)``, and two accesses are the same slot when they
        resolve to the same pair relative to the incoming stack pointer.  The
        *name* comes from IDA, because that is the name the disassembly shows
        and the one the user gets to choose.

        Doing it the other way round -- grouping by IDA's name alone -- named
        a slot that was read through a displacement operand IDA had labelled
        and written through a form it had not, so one location printed two
        ways and nothing said they were the same address.

        Three conditions, each of which really fires on this corpus:

        * some access to it carries an IDA frame member, to take the name
          from.
        * the slot is declared at its *widest* access, and a narrower access
          prints as ``*(u32 *)&var_30``.  Requiring one width throughout
          instead dropped two thirds of the slots, because a quadword buffer
          read back a word at a time is ordinary here -- and the cast form
          still shows the reader it is the same slot, which the explicit
          ``*(u32 *)((sp_2 + 0x50) & 0x3FFF0)`` does not.  Where two types
          share that widest access (a load typed `int` and a store typed
          `u32`), the plain integer spelling is used rather than asserting a
          signedness the code does not settle.
        A stack-derived address this pass cannot resolve to a fixed offset --
        a dynamic index into a stack array, or the ``(x & ~0xF) + k`` shape
        ``addr_expr`` deliberately refuses to simplify -- could reach any
        slot.  Switching naming off for the whole function on that basis cost
        nine tenths of the feature (892 named lines down to 78) and would be
        out of step with the rest of the tool, which assumes bounded aliasing
        and says so: ``recover_stores`` reports ``assumed_noalias`` rather
        than giving up.  So the count is recorded and stated in the header
        instead.
        """
        from .scalarize import addr_expr, word0

        sp_in = (regs.SP, 0)
        # Values built from the incoming stack pointer, so an address that
        # fails to resolve can be recognised as *stack*-derived rather than
        # merely unknown.
        # To a fixpoint, not in one pass: a phi can take an argument defined
        # later in iteration order, and a loop-carried stack pointer would
        # otherwise escape the taint entirely.
        #
        # Pseudo-registers are excluded at both ends.  Memory is one of them,
        # and a store's operands include both its address and the memory
        # chain, so letting taint through R_MEM would mark every subsequent
        # load -- and then every address computed from a loaded value -- as
        # stack-derived, which switches the whole feature off everywhere.
        tainted = {sp_in}
        changed = True
        while changed:
            changed = False
            for insn in self.func.insns():
                d = insn.defines()
                if d is None or d.reg in regs.PSEUDO or d.key() in tainted:
                    continue
                if any(s.is_var and s.reg not in regs.PSEUDO
                       and s.key() in tainted for s in insn.srcs):
                    tainted.add(d.key())
                    changed = True

        groups = {}
        for insn in self.func.insns():
            if insn.op in MEM_READS or insn.op in MEM_WRITES:
                if len(insn.srcs) < 2:
                    continue
                base, off, _ = addr_expr(insn.srcs[1], self.defs)
                if base != sp_in:
                    if base in tainted:
                        self.stack_unresolved += 1
                    continue
                g = groups.setdefault((base, off),
                                      {"ew": None, "types": {}, "name": None})
                ew = insn.ew
                if g["ew"] is None or int(ew) > int(g["ew"]):
                    g["ew"] = ew
                g["types"].setdefault(ew, set()).add(self._access_type(insn))
                if g["name"] is None and insn.ea:
                    hit = self.stk_of(insn.ea)
                    if hit is not None and hit[0]:
                        g["name"] = hit[0]
            elif insn.op == Op.ADD:
                d = insn.defines()
                if d is not None:
                    base, off, _ = addr_expr(d, self.defs)
                    if base == sp_in:
                        signed_off = frame.signed_offset(off)
                        if signed_off != 0 and -0x4000 <= signed_off <= 0x200:
                            g = groups.setdefault((base, off), {"ew": EW.Q, "types": {}, "name": None})
                            if g["ew"] is None:
                                g["ew"] = EW.Q
                            g["types"].setdefault(EW.Q, set()).add("qword")
                            if g["name"] is None and insn.ea:
                                hit = self.stk_of(insn.ea)
                                if hit is not None and hit[0]:
                                    g["name"] = hit[0]
                            if g["name"] is None:
                                if signed_off < 0:
                                    g["name"] = "var_%X" % abs(signed_off)
                                else:
                                    g["name"] = "arg_%X" % signed_off

        # The frame's own slots are not locals.  Naming them turned the frame
        # adjust into `sp_2 = &var_150;` -- true, since the new stack pointer
        # is the back chain's address, but it hides that sp_2 is the frame
        # pointer, and it declared a slot whose only write the prologue pass
        # had just hidden.
        reserved = {off for off, _ in self.frame.saves}
        if self.frame.back_chain is not None:
            reserved.add(self.frame.back_chain)

        out = {}
        for key, g in groups.items():
            if g["name"] is None or frame.signed_offset(key[1]) in reserved:
                continue
            widest = g["types"][g["ew"]]
            ct = widest.pop() if len(widest) == 1 else CTYPE[g["ew"]]
            out[key] = (g["name"], ct, g["ew"])
        return out

    def _addr_of_slot(self, insn):
        """
        ``&var_30`` when this instruction computes a named slot's address.

        Deliberately narrow: the value has to resolve to *exactly* a slot the
        listing already names, so this never invents a name and never prints
        an address that is merely near one.  The declaration is registered
        here too, because taking a slot's address is as much a mention of it
        as reading it.
        """
        if self._slots is None:
            self._slots = self._slot_types()
        if not self._slots:
            return None
        from .scalarize import addr_expr, word0
        d = insn.defines()
        if d is None:
            return None
        base, off, _ = addr_expr(d, self.defs)
        slot_base = off & ~0xF
        byte_off = off & 0xF
        hit = self._slots.get((base, slot_base))
        if hit is None:
            hit = self._slots.get((base, off))
            if hit is None:
                return None
            byte_off = 0
        self.stack_decls[hit[0]] = hit[1]
        eff_hit = self.user_renames.get(hit[0], hit[0])
        if byte_off == 0:
            return "&" + eff_hit
        return "(char *)&%s + %d" % (eff_hit, byte_off)

    def _stack_name(self, insn):
        """
        How this access prints as a named slot: the name, or None.

        A narrower access than the slot's own width takes the address of the
        slot and casts, which keeps one name for one location.
        """
        if self._slots is None:
            self._slots = self._slot_types()
        if not self._slots or len(insn.srcs) < 2:
            return None
        from .scalarize import addr_expr, word0
        base, off, _ = addr_expr(insn.srcs[1], self.defs)
        slot_base = off & ~0xF
        byte_off = off & 0xF
        hit = self._slots.get((base, slot_base))
        if hit is None:
            hit = self._slots.get((base, off))
            if hit is None:
                return None
            byte_off = 0
        name, ct, ew = hit
        self.stack_decls[name] = ct
        eff_name = self.user_renames.get(name, name)
        t = self._access_type(insn)
        if byte_off == 0:
            if insn.ew == ew:
                return eff_name
            return "*(%s *)&%s" % (t, eff_name)
        else:
            return "*((%s *)&%s + %d)" % (t, eff_name, byte_off)

    def _deref(self, insn, depth=0):
        """
        `*p` when the address is a pointer of the right shape, `*(T *)(e)`
        otherwise.  The cast is not decoration -- it is the only thing saying
        how wide the access is when the pointer type is not known.
        """
        addr = insn.srcs[1]
        t = self._access_type(insn)
        # A slot IDA resolved to a frame member prints as that member: it is
        # the name the disassembly shows, including one the user chose.
        slot = self._stack_name(insn)
        if slot is not None:
            return slot
        pt = self.type_of(addr)
        if (pt is not None and pt.kind == types.PTR and addr.is_var
                and addr.key() not in self.inlinable
                and pt.pointee is not None
                and self._pointee_matches(pt.pointee, insn)):
            return "*%s" % self.nm(addr)
        # A store's destination never prints as a string literal: the bytes
        # there may well be text today, but `*(char *)"..." = x` reads as an
        # assignment to a constant rather than as a write into a buffer.
        strings = insn.op not in (Op.STORE, Op.STOREQ)
        addr_text = self.addr(addr, depth, strings=strings)
        if addr_text.startswith("&"):
            if " " not in addr_text:
                return "*(%s *)%s" % (t, addr_text)
            return "*(%s *)(%s)" % (t, addr_text)
        return "*(%s *)(%s)" % (t, addr_text)

    @staticmethod
    def _pointee_matches(pointee, insn):
        if insn.op in (Op.LOADQ, Op.STOREQ, Op.LOADU):
            return pointee.kind == types.VEC
        return pointee.width == int(insn.ew)

    def declarations(self):
        """
        Locals, grouped by type -- the part that makes the output read as C
        rather than as a register listing.

        Must be called *after* the body has been rendered: it declares exactly
        the names the body printed (see :attr:`printed`).  Deciding in advance
        which names the renderer would emit is what previously left deeply
        nested expressions, and muted clobbers read as call arguments,
        mentioning variables nothing declared.
        """
        if self.types is None:
            return []
        # Exclude by *name*, not by register number: a register can be both an
        # argument and a local, because the incoming value is one web and
        # anything the function writes to that register later is another.
        # Only the incoming web is the parameter.
        pnames = set(self.param_name.values()) | {self.user_renames.get(n, n) for n in self.param_name.values()}
        by_name = {}
        for insn in self.func.insns():
            d = insn.defines()
            if d is None or d.reg in regs.PSEUDO:
                continue
            if self._suppressed(insn, d):
                # This definition prints nothing, so it says nothing about
                # the name's type or its group.  A register that is both
                # live in and written by a call's link constant reaches here,
                # and belongs in the live-in group below rather than being
                # claimed as a local that is never assigned.
                continue
            nm = self.namer.name(d)
            if nm not in self.assigned:
                # Folded into its use site, or simply dead -- either way the
                # body never mentions the name.  Note this must NOT test
                # `inlinable` on its own: it is `_inlinable`'s depth pruning
                # that guarantees an inlined definition never falls back to
                # printing its name.
                continue
            eff_nm = self.user_renames.get(nm, nm)
            if eff_nm in by_name or eff_nm in pnames or nm in pnames:
                continue
            ct = self.user_types.get(nm, self.user_types.get(eff_nm, self.ctype(self.type_of(d))))
            by_name[eff_nm] = ct

        for ret_var in getattr(self, "call_return", {}).values():
            nm = self.namer.name(ret_var)
            if nm in self.assigned:
                eff_nm = self.user_renames.get(nm, nm)
                if eff_nm not in by_name and eff_nm not in pnames and nm not in pnames:
                    ct = self.user_types.get(nm, self.user_types.get(eff_nm, self.ctype(self.type_of(ret_var))))
                    by_name[eff_nm] = ct

        # Registers the function reads but never writes, outside the range the
        # ABI calls arguments.  They are genuine inputs -- callee-saved
        # registers a caller left set up, say -- and leaving them out would
        # mean the listing uses names it never declares.
        live_in = {}
        for insn in self.func.insns():
            # An ABI op's operand list is conservative filler, except for the
            # leading operands `ABI_OPS` calls real -- an indirect branch's
            # target among them.  Skipping those too left `goto *r3;` naming a
            # register the listing never declared.
            n_real = ABI_OPS.get(insn.op)
            srcs = insn.srcs if n_real is None else insn.srcs[:n_real]
            for u in [x for x in srcs if x.is_var]:
                if u.ver != 0 or u.reg in regs.PSEUDO:
                    continue
                nm = self.namer.name(u)
                if nm not in self.printed:
                    continue      # the body never mentions it
                eff_nm = self.user_renames.get(nm, nm)
                if eff_nm in pnames or eff_nm in by_name or eff_nm in live_in or nm in pnames:
                    continue
                ct = self.user_types.get(nm, self.user_types.get(eff_nm, self.ctype(self.type_of(u))))
                live_in[eff_nm] = ct

        # Stack slots are storage, not values: a slot this function only
        # reads was filled by something else (a callee handed a pointer to
        # it), so unlike a register name it is declared whether or not
        # anything here assigns it.  The annotation says which kind it is.
        stack = {}
        for nm, ct in self.stack_decls.items():
            eff_nm = self.user_renames.get(nm, nm)
            if eff_nm not in pnames and eff_nm not in by_name and eff_nm not in live_in and nm not in pnames:
                eff_ct = self.user_types.get(nm, self.user_types.get(eff_nm, ct))
                stack[eff_nm] = eff_ct

        for eff_nm, ct in by_name.items():
            self.all_var_types[eff_nm] = ct
        for eff_nm, ct in live_in.items():
            self.all_var_types[eff_nm] = ct
        for eff_nm, ct in stack.items():
            self.all_var_types[eff_nm] = ct

        if not by_name and not live_in and not stack:
            return []
        groups = {}
        for nm, ct in by_name.items():
            groups.setdefault(ct, []).append(nm)

        out = []
        if live_in:
            ins = {}
            for nm, ct in live_in.items():
                ins.setdefault(ct, []).append(nm)
            for ct in sorted(ins):
                out.extend(_decl_lines(ct, sorted(ins[ct]),
                                       "   // live in"))
        for ct in sorted(groups):
            out.extend(_decl_lines(ct, sorted(groups[ct])))
        if stack:
            st = {}
            for nm, ct in stack.items():
                st.setdefault(ct, []).append(nm)
            for ct in sorted(st):
                out.extend(_decl_lines(ct, sorted(st[ct]), "   // stack"))
        return out

    # -- statements --------------------------------------------------------

    def _trivial_phi(self, insn):
        """
        Whether this phi needs no statement of its own.

        :func:`outssa.lower_phis` has already inserted a copy for every
        argument whose name differs from the result's, except on an edge that
        is critical in both directions -- there is nowhere to put one there
        without splitting the edge.  Those are what still print.
        """
        d = insn.defines()
        if d is None:
            return True
        nm = self.namer.name(d)
        return all(not s.is_var or self.namer.name(s) == nm
                   for s in insn.srcs)

    def _phi_arg(self, v, depth, sc):
        """
        One argument of a printed phi.

        An argument whose definition is a muted call clobber has no name in
        the listing -- that is what muting means -- so printing its name left
        the phi mentioning a variable nothing declared.  It is an unknown
        value, and `<clobbered by call>` is how the rest of the listing says
        so.

        The same applies, less obviously, to an argument whose *colour* has no
        visible definition at all: a merge of other phis and muted clobbers
        reaches this point with no assignment anywhere in the function.  Three
        functions in the corpus did that.  An incoming value is different --
        version 0 arrives with a value and is declared as live in.
        """
        if v.is_var and v.key() in self.mute:
            return "<clobbered by call>"
        if v.is_var and v.ver != 0 \
                and self.namer.name(v) not in self.visible:
            return "<unknown>"
        return self.operand(v, depth, sc)


    def _this_reg(self):
        """
        The register holding the implicit ``this``, or None.

        A non-static member takes the object in r3, so a function whose name
        puts it directly inside a class scope (``member_class``, established
        from a vtable/ctor/dtor for that scope -- see data.Members) has ``this``
        there.  The name is trusted rather than confirmed from use, because use
        cannot tell the two things apart: a member may ignore the object
        entirely (its r3 is simply unread), and one that reads a field does so
        through ``this + offset`` address arithmetic that is indistinguishable
        from scalar arithmetic on an integer argument.  The residual risk is a
        *static* member -- a class scope with no object -- printing a spurious
        ``this``; none occur in the corpus this was built against, and a real
        decompiler makes the same name-based call.
        """
        return regs.ARG_FIRST if self.member_class else None

    def _find_sbox_trees(self):
        """
        Identify 7-SELB + 8-SHUFB binary trees implementing SPU AES S-box lookup.
        Returns:
            trees: dict of root_key -> (input_var, is_inv, tree_insns, mask_insn)
            debris_keys: set of SSA keys for all internal tree instructions to suppress
        """
        defs = self.defs
        trees = {}
        debris_keys = set()
        for insn in self.func.insns():
            if insn.op != Op.SELB or len(insn.srcs) < 3:
                continue
            # Level 2 (2 SELBs)
            l2 = []
            ok = True
            for s in insn.srcs[:2]:
                if not s.is_var or s.key() not in defs or defs[s.key()].op != Op.SELB:
                    ok = False
                    break
                l2.append(defs[s.key()])
            if not ok or len(l2) != 2:
                continue
            # Level 1 (4 SELBs)
            l1 = []
            for n in l2:
                for s in n.srcs[:2]:
                    if not s.is_var or s.key() not in defs or defs[s.key()].op != Op.SELB:
                        ok = False
                        break
                    l1.append(defs[s.key()])
                if not ok:
                    break
            if not ok or len(l1) != 4:
                continue
            # Level 0 (8 SHUFBs)
            shufs = []
            for n in l1:
                for s in n.srcs[:2]:
                    if not s.is_var or s.key() not in defs or defs[s.key()].op != Op.SHUFB:
                        ok = False
                        break
                    shufs.append(defs[s.key()])
                if not ok:
                    break
            if not ok or len(shufs) != 8:
                continue

            masks = [s.srcs[2] for s in shufs]
            mask_keys = set(m.key() for m in masks if m.is_var)
            if len(mask_keys) != 1:
                continue
            mask_insn = defs.get(list(mask_keys)[0])
            if mask_insn is None or mask_insn.op != Op.AND or len(mask_insn.srcs) < 2:
                continue
            inp_var = mask_insn.srcs[0]

            # Detect inverse S-box: check if sources trace to 0xCB30..0xCC20
            is_inv = False
            for s in shufs:
                for src in s.srcs[:2]:
                    if src.is_var and src.key() in defs:
                        sd = defs[src.key()]
                        if sd.op in (Op.LOAD, Op.LOADQ) and len(sd.srcs) >= 2:
                            ca = _get_const_addr(sd.srcs[1], defs)
                            if ca is not None and 0xCB30 <= ca <= 0xCC20:
                                is_inv = True

            tree_insns = [insn] + l2 + l1 + shufs
            trees[insn.dst.key()] = (inp_var, is_inv, tree_insns, mask_insn)
            for node in tree_insns[1:]:
                d = node.defines()
                if d is not None:
                    debris_keys.add(d.key())
            d_mask = mask_insn.defines()
            if d_mask is not None:
                debris_keys.add(d_mask.key())

        return trees, debris_keys

    def _compute_debris(self):
        """
        Dead remnants of a store idiom that a call's operand list pinned.

        The SPU lifter names r3..r10 on every call because a callee *might*
        read them, which is what stops DCE from deleting genuine argument setup
        before any arity is known.  By the time this runs the arity *is* known,
        so an operand a known-arity call does not actually take is not an
        argument.  ``recover_stores`` relies on that: folding a
        ``cwd``/``lqd``/``shufb``/``stqd`` group into one scalar store orphans
        the old-quadword reload and the insert mask, and when a call follows --
        a constructor invoking a method on the object it has just built is the
        usual case -- their registers sit in that call's operand list and
        survive DCE.  Printing them puts back the very lines scalarisation
        removed, ``r7 = var_20; r6 = genctl.w(sp_2);`` around a scalar store
        that already says everything.

        Scope is deliberately narrow.  The candidates are only the values
        ``recover_stores`` recorded as feeding a dismantled shufb (and their
        own address arithmetic, reached backwards), never dead values in
        general -- a genuinely dead load the reader put there stays visible.
        And a candidate is dropped only when a full, arity-aware liveness finds
        nothing real reads it: an unknown-arity call keeps *all* of its
        operands, so a value some indeterminate callee might take is never
        removed even if it did feed the idiom.
        """
        self.sbox_trees, sbox_debris = self._find_sbox_trees()
        roots = (getattr(self.func, "store_idiom_roots", None) or set()) | sbox_debris
        defs = self.defs

        # 1. Arity-aware liveness: what a real reader (or a call that really
        #    passes it) keeps alive.  A call/return keeps only its genuine
        #    operands; every other instruction keeps its operands once its own
        #    result is kept.
        kept = set()

        def keeping(insn):
            op = insn.op
            d = insn.defines()
            if d is not None and hasattr(self, 'sbox_trees') and d.key() in self.sbox_trees:
                inp_var = self.sbox_trees[d.key()][0]
                return [inp_var] if inp_var.is_var else []
            if op in ABI_OPS:
                keep = list(insn.srcs[:ABI_OPS[op]])      # indirect target
                if op in (Op.CALL, Op.TCALL):
                    n = (self.arity_of(insn.aux)
                         if self.arity_of is not None and insn.aux is not None
                         else None)
                    keep += (call_arguments(insn, n) or []) if n is not None \
                        else [s for s in insn.srcs if s.is_var and s.key() not in roots]
                elif op in (Op.ICALL, Op.IJMP):
                    keep += [s for s in insn.srcs if s.is_var and s.key() not in roots]
                elif op == Op.RET:
                    keep += [s for s in insn.srcs if s.is_var
                             and s.reg == regs.ARG_FIRST and s.ver != 0
                             and s.key() not in roots]
                return keep
            if op.has_side_effects or op.is_terminator:
                return insn.srcs
            d = insn.defines()
            return insn.srcs if d is not None and d.key() in kept else []

        changed = True
        while changed:
            changed = False
            for insn in self.func.insns():
                for u in keeping(insn):
                    if u.is_var and u.key() in defs and u.key() not in kept:
                        kept.add(u.key())
                        changed = True

        dead = {k for k, insn in defs.items()
                if k not in kept and not insn.op.has_side_effects
                and not insn.op.is_terminator
                and insn.op not in (Op.PHI, Op.UNDEF)}

        # 2. Keep only the idiom's own cone: the recorded roots, plus the
        #    address arithmetic that fed them and nothing live.
        cone = set(roots) & dead
        frontier = list(cone)
        while frontier:
            insn = defs.get(frontier.pop())
            if insn is None:
                continue
            for u in insn.uses():
                if u.key() in dead and u.key() not in cone:
                    cone.add(u.key())
                    frontier.append(u.key())
        # The inserted value reaches the shufb by way of an `ila`, whose
        # constant folds into the scalar store before this idiom is even
        # matched -- so the store spells it inline and the `ila` is left as a
        # dead `r5 = #0x2e698:w4` the call's operand list still pins.  A dead
        # constant or unused stack address computation read by nothing real
        # carries no information wherever it sits, so drop those too.
        cone |= {k for k in dead if defs[k].op in (Op.CONST, Op.ADD, Op.MOV) or
                 (defs[k].op in (Op.LOAD, Op.LOADQ) and len(defs[k].srcs) >= 2
                  and _get_const_addr(defs[k].srcs[1], defs) is not None)}
        cone |= sbox_debris

        # 3. Arithmetic carry/borrow idioms feeding ADDX/SUBX.
        uses_of = {}
        for insn in self.func.insns():
            for u in insn.uses():
                uses_of.setdefault(u.key(), []).append(insn)

        all_carry_ops = set()
        cand_by_insn = {}
        for insn in self.func.insns():
            if insn.op in (Op.ADDX, Op.SUBX) and len(insn.srcs) >= 3:
                c_var = insn.srcs[2]
                if not c_var.is_var or c_var.key() not in defs:
                    continue
                c_def = defs[c_var.key()]
                cands = []
                if c_def.op == Op.SHUFB:
                    cands.append(c_def)
                    for s in c_def.srcs[:2]:
                        if s.is_var and s.key() in defs:
                            s_def = defs[s.key()]
                            if s_def.op in (Op.CG, Op.BG, Op.CGX, Op.BGX):
                                cands.append(s_def)
                    if len(c_def.srcs) > 2 and c_def.srcs[2].is_var and c_def.srcs[2].key() in defs:
                        m_def = defs[c_def.srcs[2].key()]
                        if m_def.op in (Op.LOAD, Op.LOADQ, Op.CONST):
                            cands.append(m_def)
                elif c_def.op in (Op.CG, Op.BG, Op.CGX, Op.BGX):
                    cands.append(c_def)
                cand_by_insn[id(insn)] = cands
                for c in cands:
                    all_carry_ops.add(id(c))
                all_carry_ops.add(id(insn))

        for cands in cand_by_insn.values():
            for cand in cands:
                d = cand.defines()
                if d is None:
                    continue
                users = uses_of.get(d.key(), [])
                real_users = [u for u in users if u.op not in ABI_OPS and u.op != Op.PHI]
                if all(id(u) in all_carry_ops for u in real_users):
                    cone.add(d.key())

        # 4. Condition chains feeding branches (Op.CJMP, Op.CIJMP)
        for insn in self.func.insns():
            if insn.op in (Op.CJMP, Op.CIJMP):
                cur = insn.srcs[0]
                chain = []
                while cur.is_var and cur.key() in defs:
                    d = defs[cur.key()]
                    if d.op in (Op.MOV, Op.EXTS):
                        chain.append(d)
                        cur = d.srcs[0]
                    elif d.op in (Op.CMPGT, Op.CMPGTU) and len(d.srcs) == 2 and d.srcs[1].is_const and (d.srcs[1].val >> 96) in (0xB, 0xF):
                        chain.append(d)
                        cur = d.srcs[0]
                    elif d.op == Op.GB and len(d.srcs) == 1:
                        chain.append(d)
                        cur = d.srcs[0]
                    elif d.op in (Op.CMPEQ, Op.FCMPEQ, Op.CMPGT, Op.CMPGTU):
                        chain.append(d)
                        for x in chain:
                            if x.defines() and x.defines().key() in dead:
                                cone.add(x.defines().key())
                        break
                    else:
                        break

        return cone

    def _suppressed(self, insn, d):
        if id(insn) in getattr(self, "_suppressed_call_undefs", ()):
            return True
        """
        Whether this definition never reaches the output at all.

        Both :meth:`insn_stmt` and :meth:`declarations` need to know: a
        variable that is never assigned anywhere in the listing must not be
        declared either, or the reader is left hunting for a value that is not
        there.  A whole prologue's worth of saved registers used to be
        declared and never mentioned again for exactly that reason.
        """
        if insn.op == Op.PHI:
            return self._trivial_phi(insn)
        if id(insn) in self.frame.hidden:
            return True
        # Dead idiom debris a known-arity call's operand list pinned past DCE.
        if d is not None and d.key() in self.debris:
            return True
        if insn.op == Op.UNDEF:
            return True
        # The link value a `brsl` writes is the return address; the call is
        # printed on the next line and says the same thing more clearly.
        if (insn.op == Op.CONST and d.reg == regs.LR
                and insn.ea in self.call_at):
            return True
        return False

    def insn_stmt(self, insn):
        """One IR instruction as a statement, or None if it is absorbed."""
        if id(insn) in self.frame.hidden:
            return None
        op = insn.op
        if op == Op.PHI and self._trivial_phi(insn):
            return None                       # control flow expresses it now
        d = insn.defines()
        if d is not None and d.key() in self.inlinable:
            return None                       # printed at its use site
        if op == Op.UNDEF:
            return None
        if d is not None and self._suppressed(insn, d):
            return None
        if op == Op.CIJMP:
            kind = insn.aux[1] if (isinstance(insn.aux, tuple) and len(insn.aux) > 1) else (insn.aux or "z")
            from .structure import Cond
            test = self._cond(Cond(insn.srcs[0], kind))
            tgt = insn.aux[0] if (isinstance(insn.aux, tuple) and len(insn.aux) > 0) else None
            if tgt is not None and tgt != 0:
                return "if ( %s ) goto loc_%X;" % (test, tgt)
            target_str = self.operand(insn.srcs[1], 0, True) if len(insn.srcs) > 1 else "?"
            return "if ( %s ) goto *%s;" % (test, target_str)

        if op in (Op.STORE, Op.STOREQ):
            if id(insn) in self.frame.hidden:
                return None                   # stated in the header instead
            return "%s = %s;" % (
                self._deref(insn),
                self.operand(insn.srcs[2], 0, op == Op.STORE, top=True))
        if op == Op.WRCH:
            val = self.operand(insn.srcs[1], 0, True, top=True)
            text = "wrch(%s, %s);" % (channels.label(insn.aux), val)
            # A literal written to MFC_Cmd is a DMA opcode; naming it turns
            # `wrch(21, 0x40)` into something a reader can act on.
            if insn.aux == 21 and insn.srcs[1].is_const:
                cmd = channels.mfc_cmd(insn.srcs[1].val >> 96)
                if cmd:
                    text += "   // %s" % cmd
            return text
        if op == Op.MTSPR:
            return "mtspr(%s, %s);" % (insn.aux,
                                       self.operand(insn.srcs[1], 0, False))
        if op == Op.SYNC:
            return None if insn.aux and str(insn.aux).startswith("after") \
                else "sync();"
        if op == Op.STOP:
            return "stop(%s);" % (insn.aux,)
        if op == Op.HALT:
            cmp_sym = {
                "eq": "==", "gt": ">", "ge": ">=",
                "lt": "<", "le": "<=", "ne": "!=",
            }.get(str(insn.aux), str(insn.aux))
            lhs = self.operand(insn.srcs[0], 0, True, want_scalar=True)
            rhs = self.operand(insn.srcs[1], 0, True, want_scalar=True)
            return "if ( %s %s %s ) __builtin_trap();" % (lhs, cmp_sym, rhs)
        if op in (Op.CALL, Op.ICALL):
            text = self.rhs(insn)
            ret_var = self.call_return.get(id(insn))
            if ret_var is not None:
                return "%s = %s;" % (self.nma(ret_var), text)
            if d is not None and d.reg not in regs.PSEUDO:
                return "%s = %s;" % (self.nma(d), text)
            return text + ";"
        if op == Op.NOP:
            return None

        if d is None:
            return self.rhs(insn) + ";"
        if d.reg in regs.PSEUDO:
            return None                       # pure bookkeeping, not code
        return "%s = %s;" % (self.nma(d), self.rhs(insn))

    # -- the AST -----------------------------------------------------------

    def _add_line(self, out, text, primary_ea=None, inlined_eas=None):
        lineno = len(out)
        out.append(text)
        self.body_eas.append(primary_ea)
        all_eas = ([primary_ea] if primary_ea is not None else []) + list(inlined_eas or [])
        unique_eas = list(dict.fromkeys(e for e in all_eas if e is not None))
        self.body_line_to_eas[lineno] = unique_eas

    def emit(self, stmts, out, indent, info):
        pad = "    " * indent
        for st in stmts:
            if isinstance(st, Basic):
                for insn in st.insns:
                    self._current_inlined_eas = []
                    text = self.insn_stmt(insn)
                    if text:
                        self._add_line(out, pad + text, primary_ea=insn.ea,
                                       inlined_eas=self._current_inlined_eas)
            elif isinstance(st, If):
                self._current_inlined_eas = []
                cond_str = self._cond(st.cond)
                ea = getattr(st, "ea", None)
                self._add_line(out, "%sif ( %s )" % (pad, cond_str),
                               primary_ea=ea, inlined_eas=self._current_inlined_eas)
                self._add_line(out, pad + "{", primary_ea=None)
                self.emit(st.then, out, indent + 1, info)
                self._add_line(out, pad + "}", primary_ea=None)
                if st.els:
                    self._add_line(out, pad + "else", primary_ea=None)
                    self._add_line(out, pad + "{", primary_ea=None)
                    self.emit(st.els, out, indent + 1, info)
                    self._add_line(out, pad + "}", primary_ea=None)
            elif isinstance(st, Loop):
                hdr_ea = getattr(st.header, "start_ea", None) if getattr(st, "header", None) else None
                if st.kind == "while":
                    self._current_inlined_eas = []
                    cond_str = self._cond(st.cond)
                    self._add_line(out, "%swhile ( %s )" % (pad, cond_str),
                                   primary_ea=hdr_ea, inlined_eas=self._current_inlined_eas)
                    self._add_line(out, pad + "{", primary_ea=None)
                    self.emit(st.body, out, indent + 1, info)
                    self._add_line(out, pad + "}", primary_ea=None)
                elif st.kind == "dowhile":
                    self._add_line(out, pad + "do", primary_ea=hdr_ea)
                    self._add_line(out, pad + "{", primary_ea=None)
                    self.emit(st.body, out, indent + 1, info)
                    self._current_inlined_eas = []
                    cond_str = self._cond(st.cond)
                    self._add_line(out, "%s} while ( %s );" % (pad, cond_str),
                                   primary_ea=hdr_ea, inlined_eas=self._current_inlined_eas)
                else:
                    self._add_line(out, "%sfor ( ;; )" % pad, primary_ea=hdr_ea)
                    self._add_line(out, pad + "{", primary_ea=None)
                    self.emit(st.body, out, indent + 1, info)
                    self._add_line(out, pad + "}", primary_ea=None)
            elif isinstance(st, Break):
                self._add_line(out, pad + "break;", primary_ea=None)
            elif isinstance(st, Continue):
                self._add_line(out, pad + "continue;", primary_ea=None)
            elif isinstance(st, Goto):
                tgt_ea = getattr(st.target, "start_ea", None)
                self._add_line(out, "%sgoto %s;" % (pad, _lbl(st.target)), primary_ea=tgt_ea)
            elif isinstance(st, Label):
                # Structuring marks every block; only the ones something
                # actually jumps to become labels in the output.
                if st.block.id in info["labels"]:
                    self._add_line(out, "%s%s:" % ("    " * max(indent - 1, 0),
                                                   _lbl(st.block)), primary_ea=st.block.start_ea)
            elif isinstance(st, Return):
                self._current_inlined_eas = []
                ret_str = self._return(st.insn)
                self._add_line(out, pad + ret_str, primary_ea=getattr(st.insn, "ea", None),
                               inlined_eas=self._current_inlined_eas)
            elif isinstance(st, Tail):
                self._current_inlined_eas = []
                ea = getattr(st.insn, "ea", None)
                if st.insn.op == Op.IJMP:
                    blk = st.insn.block
                    kind = ("jump table" if blk is not None
                            and len(blk.succs) > 1 else "indirect / tail call")
                    txt = "%sgoto *%s;   // %s" % (pad, self.operand(st.insn.srcs[0], 0, True), kind)
                    self._add_line(out, txt, primary_ea=ea, inlined_eas=self._current_inlined_eas)
                elif st.insn.op == Op.TCALL:
                    call_str = self.rhs(st.insn)
                    if self.ret_type != "void":
                        txt = "%sreturn %s;" % (pad, call_str)
                    else:
                        txt = "%s%s;" % (pad, call_str)
                    self._add_line(out, txt, primary_ea=ea, inlined_eas=self._current_inlined_eas)
                elif st.insn.op == Op.JMP:
                    target_name = self.name_of(st.insn.aux) if self.name_of else None
                    if not target_name:
                        target_name = "loc_%X" % st.insn.aux
                    else:
                        target_name = self.user_renames.get(target_name, target_name)
                    self._add_line(out, "%sgoto %s;   // outside this function" % (pad, target_name), primary_ea=ea)
                else:
                    txt = pad + self.rhs(st.insn) + ";"
                    self._add_line(out, txt, primary_ea=ea, inlined_eas=self._current_inlined_eas)

    def _simplify_cond(self, val, k):
        cur = val
        chain_eas = []
        chain_insns = []
        seen = set()
        while cur.is_var and cur.key() in self.defs and cur.key() not in seen:
            seen.add(cur.key())
            d = self.defs[cur.key()]
            if d.op == Op.MOV and len(d.srcs) == 1:
                if getattr(d, "ea", None) is not None:
                    chain_eas.append(d.ea)
                chain_insns.append(d)
                cur = d.srcs[0]
                continue
            if d.op == Op.EXTS and d.ew in (EW.H, EW.W) and len(d.srcs) == 1:
                if getattr(d, "ea", None) is not None:
                    chain_eas.append(d.ea)
                chain_insns.append(d)
                cur = d.srcs[0]
                continue
            break

        if cur.is_var and cur.key() in self.defs:
            d = self.defs[cur.key()]
            if d.op in (Op.CMPGT, Op.CMPGTU) and len(d.srcs) == 2 and d.srcs[1].is_const and (d.srcs[1].val >> 96) in (0xB, 0xF):
                lhs_op = d.srcs[0]
                if lhs_op.is_var and lhs_op.key() in self.defs:
                    gb_insn = self.defs[lhs_op.key()]
                    if gb_insn.op == Op.GB and len(gb_insn.srcs) == 1:
                        cmp_op = gb_insn.srcs[0]
                        if cmp_op.is_var and cmp_op.key() in self.defs:
                            cmp_insn = self.defs[cmp_op.key()]
                            if cmp_insn.op in (Op.CMPEQ, Op.FCMPEQ) and len(cmp_insn.srcs) == 2:
                                for x in (d, gb_insn, cmp_insn):
                                    if getattr(x, "ea", None) is not None:
                                        chain_eas.append(x.ea)
                                    self._suppressed_call_undefs.add(id(x))
                                for ch_insn in chain_insns:
                                    self._suppressed_call_undefs.add(id(ch_insn))
                                self._current_inlined_eas.extend(chain_eas)
                                is_zero = (k in ("z", "hz"))
                                lhs = self.operand(cmp_insn.srcs[0], 0, True, want_scalar=True)
                                rhs = self.operand(cmp_insn.srcs[1], 0, True, want_scalar=True)
                                op_sym = "!=" if is_zero else "=="
                                return "%s %s %s" % (lhs, op_sym, rhs)

            if d.op == Op.GB and len(d.srcs) == 1:
                cmp_op = d.srcs[0]
                if cmp_op.is_var and cmp_op.key() in self.defs:
                    cmp_insn = self.defs[cmp_op.key()]
                    if cmp_insn.op in (Op.CMPEQ, Op.FCMPEQ) and len(cmp_insn.srcs) == 2:
                        for x in (d, cmp_insn):
                            if getattr(x, "ea", None) is not None:
                                chain_eas.append(x.ea)
                            self._suppressed_call_undefs.add(id(x))
                        for ch_insn in chain_insns:
                            self._suppressed_call_undefs.add(id(ch_insn))
                        self._current_inlined_eas.extend(chain_eas)
                        is_zero = (k in ("z", "hz"))
                        lhs = self.operand(cmp_insn.srcs[0], 0, True, want_scalar=True)
                        rhs = self.operand(cmp_insn.srcs[1], 0, True, want_scalar=True)
                        op_sym = "!=" if is_zero else "=="
                        return "%s %s %s" % (lhs, op_sym, rhs)

            if d.op in (Op.CMPEQ, Op.CMPGT, Op.CMPGTU, Op.FCMPEQ, Op.FCMPGT) and len(d.srcs) == 2:
                if getattr(d, "ea", None) is not None:
                    chain_eas.append(d.ea)
                self._current_inlined_eas.extend(chain_eas)
                self._suppressed_call_undefs.add(id(d))
                for ch_insn in chain_insns:
                    self._suppressed_call_undefs.add(id(ch_insn))
                is_zero = (k in ("z", "hz"))
                lhs = self.operand(d.srcs[0], 0, True, want_scalar=True)
                rhs = self.operand(d.srcs[1], 0, True, want_scalar=True)

                if d.srcs[1].is_const and (d.srcs[1].val >> 96) == 0:
                    rhs = "0"
                if d.srcs[0].is_const and (d.srcs[0].val >> 96) == 0:
                    lhs = "0"

                if d.op in (Op.CMPEQ, Op.FCMPEQ):
                    op_sym = "!=" if is_zero else "=="
                elif d.op in (Op.CMPGT, Op.FCMPGT):
                    op_sym = "<=" if is_zero else ">"
                    if op_sym == "<=" and rhs in ("-1", "-0x1", "-0x1:w4"):
                        op_sym = "<"
                        rhs = "0"
                    elif op_sym == ">" and rhs in ("-1", "-0x1", "-0x1:w4"):
                        op_sym = ">="
                        rhs = "0"
                elif d.op == Op.CMPGTU:
                    op_sym = "<=" if is_zero else ">"

                return "%s %s %s" % (lhs, op_sym, rhs)

        return None

    def _invert_or_simplify_relational(self, v, inverted):
        s = v.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()

        ops = [(" == ", "!="), (" != ", "=="), (" <= ", ">"), (" >= ", "<"),
               (" >s ", "<=s"), (" <=s ", ">s"), (" >u ", "<=u"), (" <=u ", ">u"),
               (" < ", ">="), (" > ", "<=")]
        for sym, inv_sym in ops:
            if sym in s:
                lhs, rhs = s.split(sym, 1)
                final_sym = inv_sym.strip() if inverted else sym.strip()
                return "%s %s %s" % (lhs.strip(), final_sym, rhs.strip())
        return None

    def _cond(self, cond):
        if cond is None:
            return "1"
        k = cond.effective
        val = cond.value

        simplified = self._simplify_cond(val, k)
        if simplified is not None:
            return simplified

        v = self.operand(val, 0, True)
        inverted = (k in ("z", "hz"))
        if "gb.w(" in v and ("> 0xB" in v or "> 11" in v or "> 0xb" in v):
            idx0 = v.find("gb.w(") + 5
            idx1 = v.find(")", idx0)
            if idx1 > idx0:
                inner = v[idx0:idx1].strip()
                if inner.startswith("(") and inner.endswith(")"):
                    inner = inner[1:-1].strip()
                rel = self._invert_or_simplify_relational(inner, inverted)
                if rel is not None:
                    return rel
                return "!(%s)" % inner if inverted else inner
        rel = self._invert_or_simplify_relational(v, inverted)
        if rel is not None:
            return rel

        if k == "z":
            return "%s == 0" % v
        if k == "nz":
            return "%s != 0" % v
        if k == "hz":
            return "(%s & 0xFFFF) == 0" % v
        return "(%s & 0xFFFF) != 0" % v

    def _return(self, insn):
        if insn.op == Op.STOP:
            return "stop(%s);" % (insn.aux,)
        roots = getattr(self.func, "store_idiom_roots", ())
        for s in insn.srcs:
            if s.is_var and s.reg == regs.ARG_FIRST and s.ver != 0:
                if s.key() in self.debris or s.key() in roots:
                    continue
                return "return %s;" % self.nm(s)
        return "return;"


# Operations whose second operand is a shift or rotate *count*, which the
# hardware reads from the preferred slot: it is a number, not a quadword.
_COUNT_OPS = frozenset((
    Op.QROTBY, Op.QROTBI, Op.QROTMBY, Op.QROTMBI, Op.QSHLBY, Op.QSHLBI,
))

_EW_CALLS = frozenset((
    Op.FADD, Op.FSUB, Op.FMUL, Op.FMA, Op.FMS, Op.FNMS, Op.FNMA,
    Op.FCMPEQ, Op.FCMPGT, Op.FCMPMEQ, Op.FCMPMGT,
))

def _decl_lines(ctype, names, suffix=""):
    """
    One or more declaration lines for ``names`` of type ``ctype``.

    In C the `*` binds to the declarator, not the type: `T *a, b;` makes only
    `a` a pointer.  So every name in a pointer group carries its own star.
    """
    if ctype.endswith("*"):
        base = ctype[:-1].rstrip()
        names = ["*" + n for n in names]
    else:
        base = ctype
    out = []
    names = list(names)
    while names:
        chunk, names = names[:8], names[8:]
        out.append("    %s %s;%s" % (base, ", ".join(chunk), suffix))
    return out


def _muted_clobbers(func, keep=()):
    """
    Clobbers whose only readers are a later call's ABI operand list.

    Those survive DCE legitimately -- a callee might read that register -- but
    printing `r5 = <clobbered by call>;` when the sole "reader" is the
    conservative argument list of the next call is noise, not information.  A
    clobber that a real instruction reads is kept: that one says the code is
    using a register the callee destroyed, which is worth seeing.

    "Read by a phi" is not by itself evidence either, and this used to count
    it as such.  A clobber that feeds a phi whose own result only ever reaches
    another ABI operand list is just as dead as one read directly from an ABI
    list -- the value never reaches code -- and on a call-heavy function that
    mistake accounted for most of the `<result in rN>` lines in the listing.
    :func:`_real_uses` already draws that distinction transitively, so use it
    rather than a second, weaker rule.
    """
    real_use = _real_uses(func, set(keep or ()) | _returned_keys(func))
    return {i.dst.key() for i in func.insns()
            if i.op == Op.UNDEF and i.dst is not None
            and i.dst.key() not in real_use}


def _returned_keys(func):
    """
    The values the return statement will name.

    A function whose last act is to forward a callee's result -- a wrapper
    ending in `return memset(...)` -- returns the clobber the call left
    behind, and nothing else in the function reads it.  Muting that one
    would print `return r3_2;` with neither an assignment nor a declaration
    for `r3_2` anywhere, which is worse than the noise it removes.  See
    :meth:`CGen._return`, which picks exactly these operands.
    """
    out = set()
    roots = getattr(func, "store_idiom_roots", set())
    for insn in func.insns():
        if insn.op != Op.RET:
            continue
        for s in insn.srcs:
            if s.is_var and s.reg == regs.ARG_FIRST and s.ver != 0 and s.key() not in roots:
                out.add(s.key())
                break
    return out


def _lbl(block):
    return "loc_%X" % block.start_ea


def _real_uses(func, seed=()):
    """
    SSA keys whose value reaches a use that is actually code.

    A use inside a call or return's ABI operand list is not evidence of
    anything: those lists name every argument register because a callee
    *might* read them.  Neither is a phi argument on its own -- it is only a
    real use if the phi's own result is.  Propagating that backwards is what
    separates "this function reads r9" from "r9 was in scope".

    ``seed`` adds keys that count as real for reasons the IR cannot show --
    a value the renderer will print as a call argument or a return operand.
    It has to go in here rather than be unioned onto the result, so that the
    phi propagation carries it backwards too: the `this` pointer a C++ call
    takes is usually a phi of one clobber per arm, and only the phi's own
    result appears in the argument list.
    """
    real = set(seed)
    consumers = {}          # key -> phis that read it
    for insn in func.insns():
        if insn.op in ABI_OPS:
            continue
        if insn.op == Op.MOV and getattr(insn, "comment", None) == "merge":
            continue
        for u in insn.uses():
            if insn.op == Op.PHI:
                consumers.setdefault(u.key(), []).append(insn)
            else:
                real.add(u.key())

    # A phi or merge copy result being real makes its arguments real, transitively.
    changed = True
    while changed:
        changed = False
        for insn in func.insns():
            if insn.op == Op.PHI:
                d = insn.defines()
                if d is None or d.key() not in real:
                    continue
                for u in insn.uses():
                    if u.key() not in real:
                        real.add(u.key())
                        changed = True
            elif insn.op == Op.MOV and getattr(insn, "comment", None) == "merge":
                d = insn.defines()
                if d is None or d.key() not in real:
                    continue
                for u in insn.uses():
                    if u.key() not in real:
                        real.add(u.key())
                        changed = True
    return real


def call_arguments(insn, n):
    """
    The first ``n`` argument operands of a call, in ABI order.

    A call carries the whole argument register set in its operand list, so the
    arguments are picked out by register number rather than by position --
    that stays correct if the ABI operand set is ever widened or narrowed.
    """
    by_reg = {}
    for s in insn.srcs:
        if s.is_var and regs.ARG_FIRST <= s.reg <= regs.ARG_LAST:
            by_reg.setdefault(s.reg, s)
    out = []
    for i in range(n):
        v = by_reg.get(regs.ARG_FIRST + i)
        if v is None:
            return None            # operand set does not reach that far
        out.append(v)
    return out


def _params(func, arity_of=None):
    """
    The function's parameters, as a contiguous list of argument registers.

    The SPU ABI is positional: r3 is the first argument, r4 the second, and so
    on.  So a function that reads r6 has at least four parameters whether or
    not it ever looks at the first three -- an unused parameter is perfectly
    ordinary.  Taking only the registers that happen to be read produced
    signatures like `sub_0(qword r6)`, which cannot be what the function looks
    like from the caller's side.

    Arity therefore comes from the *highest* argument register with a real
    use, and every register below it is a parameter too.
    """
    # Check if this function itself is variadic (>= 16 arg register spills)
    var_spills = sum(1 for i in func.insns()
                     if i.op == Op.STOREQ and len(i.srcs) >= 3
                     and i.srcs[2].is_var and i.srcs[2].ver == 0 and 4 <= i.srcs[2].reg <= 79)
    if var_spills >= 16:
        # Variadic function: only r3 (format string) plus registers with non-STOREQ uses
        hi = regs.ARG_FIRST
        for insn in func.insns():
            if insn.op in (Op.PHI, Op.STOREQ) or insn.op in ABI_OPS:
                continue
            for u in insn.uses():
                if u.ver == 0 and regs.ARG_FIRST <= u.reg <= regs.ARG_LAST:
                    hi = max(hi, u.reg)
        return list(range(regs.ARG_FIRST, hi + 1))

    hi = None
    real = _real_uses(func)
    for insn in func.insns():
        if insn.op == Op.PHI:
            continue
        if insn.op in ABI_OPS:
            if arity_of is None or insn.op not in (Op.CALL, Op.ICALL, Op.TCALL):
                continue
            if insn.op in (Op.CALL, Op.TCALL) and insn.aux is not None:
                try:
                    from . import is_variadic
                    if is_variadic(insn.aux):
                        continue
                except Exception:
                    pass
            n = arity_of(insn.aux) if insn.op in (Op.CALL, Op.TCALL) else None
            if not n:
                continue
            args = call_arguments(insn, n) or []
            for u in args:
                if u.ver == 0 and regs.ARG_FIRST <= u.reg <= regs.ARG_LAST:
                    hi = u.reg if hi is None else max(hi, u.reg)
            continue
        for u in insn.uses():
            if u.ver != 0 or u.key() not in real:
                continue
            if regs.ARG_FIRST <= u.reg <= regs.ARG_LAST:
                hi = u.reg if hi is None else max(hi, u.reg)
    if hi is None:
        return []
    if hi > regs.ARG_FIRST + 7:
        hi = regs.ARG_FIRST + 7
    return list(range(regs.ARG_FIRST, hi + 1))


def _signature(g, func):
    """Return type and parameter list, from user override or recovered types."""
    g.sig_params = []
    if g.user_proto:
        from . import abi
        parsed = abi._parse_proto(g.user_proto, func.name or ("sub_%X" % func.start_ea))
        if parsed is not None:
            g.ret_type = parsed.ret_type
            if parsed.params:
                plist = []
                for p in parsed.params:
                    eff_name = g.user_renames.get(p.name, p.name)
                    eff_type = g.user_types.get(p.name, g.user_types.get(eff_name, p.type_str))
                    sep = "" if eff_type.endswith("*") else " "
                    plist.append(f"{eff_type}{sep}{eff_name}")
                    g.sig_params.append((eff_name, eff_type))
                    g.all_var_types[eff_name] = eff_type
                    g.all_var_types[p.name] = eff_type
                return parsed.ret_type, ", ".join(plist)
            else:
                return parsed.ret_type, "void"

    plist = []
    for r in g.params:
        raw_nm = getattr(g, "raw_param_name", {}).get(r, g.param_name[r])
        nm = g.param_name[r]
        # The implicit `this` is a pointer to the class, whatever demand
        # analysis made of r3 (it is usually only forwarded, so it types as a
        # bare word); the name says what it points at.
        if r == g.this_reg:
            p_type = "%s *" % g.member_class
            plist.append("%s%s" % (p_type, nm))
            g.sig_params.append((nm, p_type))
            g.all_var_types[nm] = p_type
            g.all_var_types[raw_nm] = p_type
            continue
        ty = g.types.of((r, 0)) if g.types is not None else None
        ct = g.ctype(ty)
        ct = g.user_types.get(raw_nm, g.user_types.get(nm, ct))
        plist.append("%s%s%s" % (ct, "" if ct.endswith("*") else " ", nm))
        g.sig_params.append((nm, ct))
        g.all_var_types[nm] = ct
        g.all_var_types[raw_nm] = ct

    # The return type is whatever r3 holds where the function returns, or user override
    if "__return__" in g.user_types:
        ret = g.user_types["__return__"]
    else:
        ret = "void"
        for insn in func.insns():
            if insn.op != Op.RET:
                continue
            for s in insn.srcs:
                if s.is_var and s.reg == regs.ARG_FIRST and s.ver != 0:
                    ret = g.ctype(g.type_of(s), "qword")
                    break
    g.ret_type = ret
    g.all_var_types["__return__"] = ret
    return ret, (", ".join(plist) or "void")


def generate(func, stmts, info, name_of=None, arity_of=None, str_of=None,
             stk_of=None, this_of=None, sym_of=None,
             user_renames=None, user_types=None, user_proto=None):
    """Render the structured AST as pseudocode lines."""
    g = CGen(func, name_of, arity_of, str_of, stk_of, this_of, sym_of,
             user_renames=user_renames, user_types=user_types, user_proto=user_proto)
    raw_name = func.name or "sub_%X" % func.start_ea
    if g.user_proto:
        from . import abi
        parsed = abi._parse_proto(g.user_proto, raw_name)
        if parsed and parsed.name and parsed.name != "sub_unknown":
            raw_name = parsed.name
    name = g.user_renames.get(raw_name, raw_name)
    ret, params = _signature(g, func)

    out = CodeLines()
    out.append("// %s  @ 0x%X" % (name, func.start_ea))
    sc = getattr(func, "scalar_stats", {}) or {}
    if sc:
        out.append("// scalarised: %d stores, %d loads, %d scalar ops"
                   % (sc.get("stores", 0),
                      sc.get("loads", 0) + sc.get("aligned_loads", 0),
                      sc.get("scalars", 0)))
    out.append("// %d loops, %d gotos%s"
               % (info.get("loops", 0), info.get("gotos", 0),
                  "" if not info.get("unreached")
                  else ", %d block(s) not reached" % len(info["unreached"])))
    ts = getattr(func, "type_stats", None)
    if ts and ts.get("kinds"):
        bits = ["%s %d" % kv for kv in sorted(ts["kinds"].items())]
        line = "// types: %s" % ", ".join(bits)
        if ts.get("ambiguous"):
            line += "  (%d read both signed and unsigned)" % ts["ambiguous"]
        if ts.get("contradictions"):
            line += "  (%d contradictory -- suspect)" % ts["contradictions"]
        out.append(line)
    if ts and ts.get("strings"):
        out.append("// %d constant(s) resolved to string literals"
                   % ts["strings"])

    fr = g.frame.describe()
    if fr:
        out.append("// " + fr)
    # Which slots got named is only known once the body has been rendered, so
    # the line is inserted here afterwards rather than appended out of order.
    stack_note_at = len(out)
    if g.params:
        out.append("// parameters: %s"
                   % "  ".join("%s = %s" % (g.param_name[r], regs.reg_name(r))
                               for r in g.params))
    proto_lineno = len(out)
    sep = "" if ret.endswith("*") else " "
    out.append("%s%s%s(%s)" % (ret, sep, name, params))
    out.append("{")
    # The body first: `declarations` declares the names it printed, so it has
    # to run second.
    info = dict(info, labels=set(info.get("labels", ())))
    body = []
    g.emit(stmts, body, 1, info)
    if g.stack_decls or g.stack_unresolved:
        note = "// stack: %d slot(s) named from the frame" % len(
            g.stack_decls)
        if g.stack_unresolved:
            note += ("; %d stack address(es) did not resolve to a slot and "
                     "are assumed not to alias one" % g.stack_unresolved)
        out.insert(stack_note_at, note)
        proto_lineno += 1
    decls = g.declarations()
    if decls:
        out.extend(decls)
        out.append("")
    header_count = len(out)
    out.header_count = header_count
    out.extend(body)
    out.append("}")

    out.line_eas = [None] * header_count + g.body_eas + [None]
    out.line_to_eas = {header_count + k: v for k, v in g.body_line_to_eas.items()}
    out.ea_to_line = {}
    for lineno, eas in out.line_to_eas.items():
        for ea in eas:
            if ea not in out.ea_to_line:
                out.ea_to_line[ea] = lineno

    # Map function prototype and header lines to func.start_ea
    if func.start_ea not in out.ea_to_line:
        out.ea_to_line[func.start_ea] = proto_lineno
    for h_line in range(proto_lineno, header_count):
        if func.start_ea not in out.line_to_eas.setdefault(h_line, []):
            out.line_to_eas[h_line].append(func.start_ea)
        if h_line < len(out.line_eas) and out.line_eas[h_line] is None:
            out.line_eas[h_line] = func.start_ea

    # Dense instruction completion across all basic blocks in func
    if getattr(func, "blocks", None):
        sorted_blocks = sorted(func.blocks, key=lambda b: b.start_ea)
        for b in sorted_blocks:
            b_lines = []
            for ins in b.insns:
                if ins.ea in out.ea_to_line:
                    b_lines.append((ins.ea, out.ea_to_line[ins.ea]))
            for ins in b.insns:
                if ins.ea in out.ea_to_line:
                    continue
                chosen_line = None
                if b_lines:
                    _, chosen_line = min(b_lines, key=lambda pair: abs(pair[0] - ins.ea))
                elif b.start_ea in out.ea_to_line:
                    chosen_line = out.ea_to_line[b.start_ea]
                else:
                    chosen_line = proto_lineno if proto_lineno >= 0 else 0
                if chosen_line is not None:
                    out.ea_to_line[ins.ea] = chosen_line
                    if ins.ea not in out.line_to_eas.setdefault(chosen_line, []):
                        out.line_to_eas[chosen_line].append(ins.ea)

    try:
        import ida_funcs, idautils
        pfn = ida_funcs.get_func(func.start_ea)
        if pfn:
            for item_ea in idautils.FuncItems(pfn.start_ea):
                if item_ea not in out.ea_to_line:
                    if out.ea_to_line:
                        closest_ea = min(out.ea_to_line.keys(), key=lambda e: abs(e - item_ea))
                        chosen = out.ea_to_line[closest_ea]
                        out.ea_to_line[item_ea] = chosen
                        out.line_to_eas.setdefault(chosen, []).append(item_ea)
    except Exception:
        pass

    out.var_types = dict(g.all_var_types)
    out.params = [p[0] for p in g.sig_params]
    out.param_types = dict(g.sig_params)
    out.sig_line = proto_lineno
    out.func_name = name
    out.ret_type = ret

    return out
