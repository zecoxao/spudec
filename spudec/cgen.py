"""
Pseudocode generation from the structured AST.

Three things happen here, none of which touch the SSA IR -- this is a
*rendering*, so `spudec.dump()` still shows the verifiable three-address form
underneath and the SSA verifier still applies to it.

**Phi-web coalescing.** Structured code has no place for phi nodes: the
control flow that selects between their arguments is now explicit. Each phi
web (the transitive closure of a phi's result and arguments) is given one
name, so `r33#2 = phi(r33#1, r33#7)` disappears and every member prints as
`r33`. Distinct webs of the same register get `_2`, `_3` suffixes rather than
being conflated, so two unrelated uses of r33 never silently become one
variable.

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
                 EW_SUFFIX)
from . import regs, channels, types
from .structure import (Basic, If, Loop, Break, Continue, Goto, Label,
                        Return, Tail)

CTYPE = {EW.B: "u8", EW.H: "u16", EW.W: "u32", EW.D: "u64", EW.Q: "qword"}
MAX_INLINE_DEPTH = 4

# Bitwise operations and their exact C spelling, with a flag saying whether
# the form already brackets itself.  `~` binds tighter than `&` and `|`, so
# `a & ~b` needs no inner parentheses.
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
# naming (phi-web coalescing)
# ---------------------------------------------------------------------------


class Namer(object):

    def __init__(self, func, param_names=None):
        self._parent = {}
        self._name = {}
        self._fixed = dict(param_names or {})
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
        """The groups of SSA keys that print as one variable."""
        out = {}
        for k in self._parent:
            out.setdefault(self._find(k), set()).add(k)
        return list(out.values())

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

        seen = {}
        for root in sorted(webs, key=lambda r: (r[0], r[1])):
            members = webs[root]
            # A web containing the function's incoming value for an argument
            # register is that parameter, and takes its positional name.
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


# ---------------------------------------------------------------------------
# inlining
# ---------------------------------------------------------------------------


def _inlinable(func, call_arg_keys=()):
    """
    Keys of definitions that should print at their use site instead.

    ``call_arg_keys`` are values that will be rendered as a call's arguments;
    those may fold into the call even though its operand list is an ABI op,
    which is what turns three `rN = const` lines plus `memcpy()` into
    `memcpy(0, 0xC720, 0x20)`.
    """
    call_arg_keys = set(call_arg_keys)
    defs, use_count, use_site = {}, {}, {}
    for insn in func.insns():
        d = insn.defines()
        if d is not None:
            defs[d.key()] = insn
        for u in insn.uses():
            k = u.key()
            use_count[k] = use_count.get(k, 0) + 1
            use_site[k] = insn

    ok = set()
    for key, insn in defs.items():
        if use_count.get(key, 0) != 1:
            continue
        # UNDEF stays a statement of its own: folding "<clobbered by call>"
        # into an expression hides the very thing the reader needs to see.
        is_arg = key in call_arg_keys
        # A constant normally stays its own statement; as a call argument it
        # reads far better folded in.
        if insn.op in (Op.PHI, Op.UNDEF) or insn.op.has_side_effects:
            continue
        if insn.op == Op.CONST and not is_arg:
            continue
        site = use_site[key]
        if site.op == Op.PHI:
            continue
        if site.op in ABI_OPS and not (is_arg and
                                       site.op in (Op.CALL, Op.ICALL)):
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
    return ok, defs


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


class CGen(object):

    def __init__(self, func, name_of=None, arity_of=None, str_of=None):
        self.func = func
        self.arity_of = arity_of
        # Resolves a constant address to a string literal; see data.py.  None
        # disables it, which is what a database-free caller wants.
        self.str_of = str_of or (lambda ea: None)
        # Which operands of each call are really its arguments.  Needs the
        # callee's arity: the operand list names every argument register
        # because a callee *might* read them, so without knowing how many it
        # actually takes there is no way to tell arguments from noise.
        self.call_args = {}
        if arity_of is not None:
            for insn in func.insns():
                if insn.op != Op.CALL or insn.aux is None:
                    continue
                n = arity_of(insn.aux)
                if n:
                    args = call_arguments(insn, n)
                    if args:
                        self.call_args[id(insn)] = args

        # Parameters get positional names (a1, a2, ...) the way a real
        # decompiler shows them; the register each one came from goes in the
        # header comment so the mapping back to the disassembly is not lost.
        self.params = _params(func, arity_of)
        self.param_name = {}
        pnames = {}
        for i, r in enumerate(self.params):
            nm = "a%d" % (i + 1)
            self.param_name[r] = nm
            pnames[(r, 0)] = nm
        self.namer = Namer(func, pnames)
        # Argument setup may fold into the call now that the call shows its
        # arguments -- but only for operands that are actually rendered.  A
        # register set before a call the callee never reads must stay a
        # visible statement, or the value would vanish from the listing.
        rendered = set()
        for args in self.call_args.values():
            for a in args:
                rendered.add(a.key())
        self.inlinable, self.defs = _inlinable(func, rendered)
        self.name_of = name_of or (lambda ea: "sub_%X" % ea)
        self.mute = _muted_clobbers(func)
        self.call_at = {i.ea: i for i in func.insns()
                        if i.op in (Op.CALL, Op.ICALL)}
        # Recover types now rather than earlier in the pipeline: inference has
        # to unify across the namer's webs, since everything printed under one
        # variable name must end up with one type.
        self.types = getattr(func, "types", None)
        if self.types is None:
            from . import lanes
            demand = getattr(func, "demand", None)
            if demand is None:
                demand = lanes.compute_demand(func)
            self.types, stats = types.infer(
                func, demand, self.namer.webs(),
                is_string=lambda val: self.str_of(val >> 96) is not None)
            stats["kinds"] = self.types.counts()
            func.types = self.types
            func.type_stats = stats

    # -- operands ----------------------------------------------------------

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
        return c.scalar_str() if scalar else str(c)

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
            return self.rhs(self.defs[v.key()], depth + 1, top=top,
                            want_scalar=want_scalar, strings=strings)
        return self.namer.name(v)

    def addr(self, v, depth=0, strings=True):
        if v.is_const:
            if strings:
                lit = self.str_of(v.val >> 96)
                if lit is not None:
                    return lit
            return "0x%X" % (v.val >> 96)
        return self.operand(v, depth, top=True, want_scalar=True,
                            strings=strings)

    # -- right-hand sides --------------------------------------------------

    def rhs(self, insn, depth=0, top=True, want_scalar=False, strings=True):
        op = insn.op
        sc = insn.scalar

        if op == Op.CONST:
            # A constant assigned to something the types say is a scalar
            # prints as a scalar, even when the lifter could not prove it was
            # one: `0x3e000` rather than `#0x3e000:w4`.
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
            return "phi(%s)" % ", ".join(self.operand(s, depth, sc)
                                         for s in insn.srcs)
        if op in (Op.LOAD, Op.LOADQ, Op.LOADU):
            return self._deref(insn, depth)
        if op in (Op.CALL, Op.ICALL):
            tgt = (self.name_of(insn.aux) if op == Op.CALL
                   else "(*%s)" % self.operand(insn.srcs[0], depth))
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
        # doubt: either the value is a scalar, or its type is a vector whose
        # element width matches the operation, so `a + b` on a `vec_uint4`
        # really is the 32-bit lane-wise add the opcode performs.
        if not sc and op in INFIX and len(insn.srcs) == 2 and insn.ew != EW.Q:
            d = insn.defines()
            t = self.type_of(d) if d is not None else None
            if t is not None and t.kind == types.VEC and t.width == int(insn.ew):
                sc = True

        if sc and op in INFIX and len(insn.srcs) == 2:
            lhs = self.operand(insn.srcs[0], depth, True)
            rhs = self.operand(insn.srcs[1], depth, True)
            sym = INFIX[op]
            # `ai rX, rX, -1` is everywhere; print it as subtraction rather
            # than as an addition of a negative literal.
            if op == Op.ADD and insn.srcs[1].is_const and rhs.startswith("-"):
                sym, rhs = "-", rhs[1:]
            text = "%s %s %s" % (lhs, sym, rhs)
            return text if top else "(%s)" % text

        name = OP_NAME.get(op, "op%d" % int(op))
        if op in _EW_CALLS:
            name += "." + EW_SUFFIX[insn.ew]
        args = ", ".join(self.operand(s, depth, sc) for s in insn.srcs)
        if insn.aux is not None and op not in (Op.CALL,):
            args = args + (", " if args else "") + "/*%s*/" % (insn.aux,)
        return "%s(%s)" % (name, args)

    # -- typed rendering ---------------------------------------------------

    def type_of(self, v):
        return self.types.of_var(v) if self.types is not None else None

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

    def _deref(self, insn, depth=0):
        """
        `*p` when the address is a pointer of the right shape, `*(T *)(e)`
        otherwise.  The cast is not decoration -- it is the only thing saying
        how wide the access is when the pointer type is not known.
        """
        addr = insn.srcs[1]
        t = self._access_type(insn)
        pt = self.type_of(addr)
        if (pt is not None and pt.kind == types.PTR and addr.is_var
                and addr.key() not in self.inlinable
                and pt.pointee is not None
                and self._pointee_matches(pt.pointee, insn)):
            return "*%s" % self.namer.name(addr)
        # A store's destination never prints as a string literal: the bytes
        # there may well be text today, but `*(char *)"..." = x` reads as an
        # assignment to a constant rather than as a write into a buffer.
        strings = insn.op not in (Op.STORE, Op.STOREQ)
        return "*(%s *)(%s)" % (t, self.addr(addr, depth, strings=strings))

    @staticmethod
    def _pointee_matches(pointee, insn):
        if insn.op in (Op.LOADQ, Op.STOREQ, Op.LOADU):
            return pointee.kind == types.VEC
        return pointee.width == int(insn.ew)

    def declarations(self):
        """
        Locals, grouped by type -- the part that makes the output read as C
        rather than as a register listing.
        """
        if self.types is None:
            return []
        # Exclude by *name*, not by register number: a register can be both an
        # argument and a local, because the incoming value is one web and
        # anything the function writes to that register later is another.
        # Only the incoming web is the parameter.
        pnames = set(self.param_name.values())
        by_name = {}
        for insn in self.func.insns():
            d = insn.defines()
            if d is None or d.reg in regs.PSEUDO:
                continue
            if d.key() in self.inlinable:
                continue          # printed at its use site, never as a name
            nm = self.namer.name(d)
            if nm in by_name or nm in pnames:
                continue
            by_name[nm] = self.ctype(self.type_of(d))

        # Registers the function reads but never writes, outside the range the
        # ABI calls arguments.  They are genuine inputs -- callee-saved
        # registers a caller left set up, say -- and leaving them out would
        # mean the listing uses names it never declares.
        live_in = {}
        for insn in self.func.insns():
            if insn.op in ABI_OPS:
                continue
            for u in insn.uses():
                if u.ver != 0 or u.reg in regs.PSEUDO:
                    continue
                nm = self.namer.name(u)
                if nm in pnames or nm in by_name or nm in live_in:
                    continue
                live_in[nm] = self.ctype(self.type_of(u))

        if not by_name and not live_in:
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
        return out

    # -- statements --------------------------------------------------------

    def insn_stmt(self, insn):
        """One IR instruction as a statement, or None if it is absorbed."""
        op = insn.op
        if op == Op.PHI:
            return None                       # control flow expresses it now
        d = insn.defines()
        if d is not None and d.key() in self.inlinable:
            return None                       # printed at its use site
        if op == Op.UNDEF and d is not None:
            if d.key() in self.mute:
                return None                   # nothing real reads it
            # A clobber sitting on a call, in a register the ABI uses for
            # return values (r3..r74), is the callee's result.  The call
            # statement is printed immediately above, so naming the callee
            # again on each line would read as several separate calls.
            if (insn.ea in self.call_at
                    and regs.ARG_FIRST <= d.reg <= regs.ARG_LAST):
                return "%s = <result in %s>;" % (self.namer.name(d),
                                                 regs.reg_name(d.reg))
        # The link value a `brsl` writes is the return address; the call is
        # printed on the next line and says the same thing more clearly.
        if (op == Op.CONST and d is not None and d.reg == regs.LR
                and insn.ea in self.call_at):
            return None
        if op == Op.CIJMP:
            kind = insn.aux or "z"
            v = self.operand(insn.srcs[0], 0, True)
            test = {"z": "%s == 0", "nz": "%s != 0",
                    "hz": "(%s & 0xFFFF) == 0",
                    "hnz": "(%s & 0xFFFF) != 0"}[kind] % v
            return "if ( %s ) goto *%s;" % (
                test, self.operand(insn.srcs[1], 0, True))

        if op in (Op.STORE, Op.STOREQ):
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
            return "halt_if(%s %s %s);" % (
                self.operand(insn.srcs[0], 0, True), insn.aux,
                self.operand(insn.srcs[1], 0, True))
        if op in (Op.CALL, Op.ICALL):
            text = self.rhs(insn)
            if d is not None and d.reg not in regs.PSEUDO:
                return "%s = %s;" % (self.namer.name(d), text)
            return text + ";"
        if op == Op.NOP:
            return None

        if d is None:
            return self.rhs(insn) + ";"
        if d.reg in regs.PSEUDO:
            return None                       # pure bookkeeping, not code
        return "%s = %s;" % (self.namer.name(d), self.rhs(insn))

    # -- the AST -----------------------------------------------------------

    def emit(self, stmts, out, indent, info):
        pad = "    " * indent
        for st in stmts:
            if isinstance(st, Basic):
                for insn in st.insns:
                    text = self.insn_stmt(insn)
                    if text:
                        out.append(pad + text)
            elif isinstance(st, If):
                out.append("%sif ( %s )" % (pad, self._cond(st.cond)))
                out.append(pad + "{")
                self.emit(st.then, out, indent + 1, info)
                out.append(pad + "}")
                if st.els:
                    out.append(pad + "else")
                    out.append(pad + "{")
                    self.emit(st.els, out, indent + 1, info)
                    out.append(pad + "}")
            elif isinstance(st, Loop):
                if st.kind == "while":
                    out.append("%swhile ( %s )" % (pad, self._cond(st.cond)))
                    out.append(pad + "{")
                    self.emit(st.body, out, indent + 1, info)
                    out.append(pad + "}")
                elif st.kind == "dowhile":
                    out.append(pad + "do")
                    out.append(pad + "{")
                    self.emit(st.body, out, indent + 1, info)
                    out.append("%s} while ( %s );" % (pad,
                                                      self._cond(st.cond)))
                else:
                    out.append("%sfor ( ;; )" % pad)
                    out.append(pad + "{")
                    self.emit(st.body, out, indent + 1, info)
                    out.append(pad + "}")
            elif isinstance(st, Break):
                out.append(pad + "break;")
            elif isinstance(st, Continue):
                out.append(pad + "continue;")
            elif isinstance(st, Goto):
                out.append("%sgoto %s;" % (pad, _lbl(st.target)))
            elif isinstance(st, Label):
                # Structuring marks every block; only the ones something
                # actually jumps to become labels in the output.
                if st.block.id in info["labels"]:
                    out.append("%s%s:" % ("    " * max(indent - 1, 0),
                                          _lbl(st.block)))
            elif isinstance(st, Return):
                out.append(pad + self._return(st.insn))
            elif isinstance(st, Tail):
                if st.insn.op == Op.IJMP:
                    out.append("%sgoto *%s;   // indirect / tail call"
                               % (pad, self.operand(st.insn.srcs[0], 0, True)))
                elif st.insn.op == Op.JMP:
                    out.append("%sgoto loc_%X;   // outside this function"
                               % (pad, st.insn.aux))
                else:
                    out.append(pad + self.rhs(st.insn) + ";")

    def _cond(self, cond):
        if cond is None:
            return "1"
        v = self.operand(cond.value, 0, True)
        k = cond.effective
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
        for s in insn.srcs:
            if s.is_var and s.reg == regs.ARG_FIRST and s.ver != 0:
                return "return %s;" % self.namer.name(s)
        return "return;"


_EW_CALLS = frozenset((
    Op.ADD, Op.SUB, Op.MUL, Op.CG, Op.BG, Op.ADDX, Op.SUBX, Op.CGX, Op.BGX,
    Op.SHL, Op.SHR, Op.SAR, Op.ROL, Op.ROTM, Op.ROTMA,
    Op.CMPEQ, Op.CMPGT, Op.CMPGTU, Op.GB, Op.FSM, Op.GENCTL, Op.EXTS,
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


def _muted_clobbers(func):
    """
    Clobbers whose only readers are a later call's ABI operand list.

    Those survive DCE legitimately -- a callee might read that register -- but
    printing `r5 = <clobbered by call>;` when the sole "reader" is the
    conservative argument list of the next call is noise, not information.  A
    clobber that a real instruction reads is kept: that one says the code is
    using a register the callee destroyed, which is worth seeing.
    """
    real_use = set()
    for insn in func.insns():
        if insn.op in ABI_OPS:
            continue
        for u in insn.uses():
            real_use.add(u.key())
    return {i.dst.key() for i in func.insns()
            if i.op == Op.UNDEF and i.dst is not None
            and i.dst.key() not in real_use}


def _lbl(block):
    return "loc_%X" % block.start_ea


def _real_uses(func):
    """
    SSA keys whose value reaches a use that is actually code.

    A use inside a call or return's ABI operand list is not evidence of
    anything: those lists name every argument register because a callee
    *might* read them.  Neither is a phi argument on its own -- it is only a
    real use if the phi's own result is.  Propagating that backwards is what
    separates "this function reads r9" from "r9 was in scope".
    """
    real = set()
    consumers = {}          # key -> phis that read it
    for insn in func.insns():
        if insn.op in ABI_OPS:
            continue
        for u in insn.uses():
            if insn.op == Op.PHI:
                consumers.setdefault(u.key(), []).append(insn)
            else:
                real.add(u.key())

    # A phi result being real makes its arguments real, transitively.
    changed = True
    while changed:
        changed = False
        for insn in func.insns():
            if insn.op != Op.PHI:
                continue
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
    hi = None
    real = _real_uses(func)
    for insn in func.insns():
        if insn.op == Op.PHI:
            continue
        if insn.op in ABI_OPS:
            # A call's operand list names every argument register, so it says
            # nothing on its own -- *unless* the callee's arity is known, in
            # which case the operands within that arity really are arguments.
            # That is what catches a parameter this function only forwards.
            if arity_of is None or insn.op not in (Op.CALL, Op.ICALL):
                continue
            n = arity_of(insn.aux) if insn.op == Op.CALL else None
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
    return list(range(regs.ARG_FIRST, hi + 1))


def _signature(g, func):
    """Return type and parameter list, from the recovered types."""
    plist = []
    for r in g.params:
        ty = g.types.of((r, 0)) if g.types is not None else None
        nm = g.param_name[r]
        ct = g.ctype(ty)
        plist.append("%s%s%s" % (ct, "" if ct.endswith("*") else " ", nm))

    # The return type is whatever r3 holds where the function returns.
    ret = "void"
    for insn in func.insns():
        if insn.op != Op.RET:
            continue
        for s in insn.srcs:
            if s.is_var and s.reg == regs.ARG_FIRST and s.ver != 0:
                ret = g.ctype(g.type_of(s), "qword")
                break
    return ret, (", ".join(plist) or "void")


def generate(func, stmts, info, name_of=None, arity_of=None, str_of=None):
    """Render the structured AST as pseudocode lines."""
    g = CGen(func, name_of, arity_of, str_of)
    name = func.name or "sub_%X" % func.start_ea
    ret, params = _signature(g, func)

    out = ["// %s  @ 0x%X" % (name, func.start_ea)]
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

    if g.params:
        out.append("// parameters: %s"
                   % "  ".join("%s = %s" % (g.param_name[r], regs.reg_name(r))
                               for r in g.params))
    sep = "" if ret.endswith("*") else " "
    out.append("%s%s%s(%s)" % (ret, sep, name, params))
    out.append("{")
    decls = g.declarations()
    if decls:
        out.extend(decls)
        out.append("")
    info = dict(info, labels=set(info.get("labels", ())))
    g.emit(stmts, out, 1, info)
    out.append("}")
    return out
