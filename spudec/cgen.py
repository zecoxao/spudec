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
from . import regs, channels
from .structure import (Basic, If, Loop, Break, Continue, Goto, Label,
                        Return, Tail)

CTYPE = {EW.B: "u8", EW.H: "u16", EW.W: "u32", EW.D: "u64", EW.Q: "qword"}
MAX_INLINE_DEPTH = 4


# ---------------------------------------------------------------------------
# naming (phi-web coalescing)
# ---------------------------------------------------------------------------


class Namer(object):

    def __init__(self, func):
        self._parent = {}
        self._name = {}
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
            base = regs.reg_name(root[0])
            n = seen.get(base, 0) + 1
            seen[base] = n
            nm = base if n == 1 else "%s_%d" % (base, n)
            for k in webs[root]:
                self._name[k] = nm

    def name(self, var):
        return self._name.get(var.key(), str(var))


# ---------------------------------------------------------------------------
# inlining
# ---------------------------------------------------------------------------


def _inlinable(func):
    """Keys of definitions that should print at their use site instead."""
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
        if insn.op in (Op.PHI, Op.CONST, Op.UNDEF) or insn.op.has_side_effects:
            continue
        site = use_site[key]
        if site.op == Op.PHI or site.op in ABI_OPS:
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

    def __init__(self, func, name_of=None):
        self.func = func
        self.namer = Namer(func)
        self.inlinable, self.defs = _inlinable(func)
        self.name_of = name_of or (lambda ea: "sub_%X" % ea)
        self.mute = _muted_clobbers(func)
        self.call_at = {i.ea: i for i in func.insns()
                        if i.op in (Op.CALL, Op.ICALL)}

    # -- operands ----------------------------------------------------------

    def const(self, c, scalar):
        return c.scalar_str() if scalar else str(c)

    def operand(self, v, depth=0, scalar=True, top=False):
        """
        ``top`` means the result already sits in a context that brackets it --
        the inside of a `*(u32 *)(...)`, or the whole right-hand side of an
        assignment -- so an infix expression needs no parentheses of its own.
        """
        if v.is_const:
            return self.const(v, scalar)
        if depth < MAX_INLINE_DEPTH and v.key() in self.inlinable:
            return self.rhs(self.defs[v.key()], depth + 1, top=top)
        return self.namer.name(v)

    def addr(self, v, depth=0):
        if v.is_const:
            return "0x%X" % (v.val >> 96)
        return self.operand(v, depth, top=True)

    # -- right-hand sides --------------------------------------------------

    def rhs(self, insn, depth=0, top=True):
        op = insn.op
        sc = insn.scalar

        if op == Op.CONST:
            return self.const(insn.srcs[0], sc)
        if op == Op.MOV:
            return self.operand(insn.srcs[0], depth, sc, top=top)
        if op == Op.UNDEF:
            return "<clobbered by call>"
        if op == Op.PHI:
            return "phi(%s)" % ", ".join(self.operand(s, depth, sc)
                                         for s in insn.srcs)
        if op in (Op.LOAD, Op.LOADQ, Op.LOADU):
            t = CTYPE[insn.ew] if op == Op.LOAD else "qword"
            if op == Op.LOADU:
                t = "qword_unaligned"
            return "*(%s *)(%s)" % (t, self.addr(insn.srcs[1], depth))
        if op in (Op.CALL, Op.ICALL):
            tgt = (self.name_of(insn.aux) if op == Op.CALL
                   else "(*%s)" % self.operand(insn.srcs[0], depth))
            return "%s()" % tgt
        if op == Op.RDCH:
            return "rdch(%s)" % channels.label(insn.aux)
        if op == Op.RCHCNT:
            return "rchcnt(%s)" % channels.label(insn.aux)
        if op == Op.MFSPR:
            return "mfspr(%s)" % insn.aux
        if op == Op.INTRINSIC:
            args = ", ".join(self.operand(s, depth, False) for s in insn.srcs)
            return "%s(%s)" % (insn.aux, args)

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

        if op == Op.STORE:
            return "*(%s *)(%s) = %s;" % (CTYPE[insn.ew],
                                          self.addr(insn.srcs[1]),
                                          self.operand(insn.srcs[2], 0, True,
                                                       top=True))
        if op == Op.STOREQ:
            return "*(qword *)(%s) = %s;" % (self.addr(insn.srcs[1]),
                                             self.operand(insn.srcs[2], 0,
                                                          False))
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


def _params(func):
    """Argument registers read before being written anywhere in the function."""
    live_in = set()
    for insn in func.insns():
        if insn.op in ABI_OPS:
            continue
        for u in insn.uses():
            if u.ver == 0 and regs.ARG_FIRST <= u.reg <= regs.ARG_FIRST + 7:
                live_in.add(u.reg)
    return sorted(live_in)


def generate(func, stmts, info, name_of=None):
    """Render the structured AST as pseudocode lines."""
    g = CGen(func, name_of)
    name = func.name or "sub_%X" % func.start_ea
    params = ", ".join("qword %s" % regs.reg_name(r) for r in _params(func)) \
        or "void"

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
    out.append("void %s(%s)" % (name, params))
    out.append("{")
    info = dict(info, labels=set(info.get("labels", ())))
    g.emit(stmts, out, 1, info)
    out.append("}")
    return out
