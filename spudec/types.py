"""
Type recovery.

The SPU gives an unusual amount of type evidence, because the ISA distinguishes
things most architectures leave implicit:

* **Width** comes from demand analysis (:mod:`lanes`).  A value nothing reads
  outside byte 3 is a byte; outside bytes 0..3, a word; if the whole quadword
  is read, it is a vector.  No other architecture hands you that for free.
* **Signedness** is in the opcode.  `cgt` is signed and `clgt` is not; `rotma`
  is an arithmetic shift and `rotm` a logical one; `xsbh`/`xshw`/`xswd` are
  sign extensions.  Each one is a fact about its operands, not a guess.
* **Floatness** is in the opcode too -- `fa`/`fm`/`dfa` and the four
  conversion instructions say so outright.
* **Pointers** come from use as an address: the operand of a `load.w` is a
  pointer to a 4-byte object, and that propagates back through the address
  arithmetic.

So this is constraint propagation over the SSA graph rather than guesswork.
Every rule below is anchored to an instruction that can only mean one thing.

Where evidence genuinely conflicts -- the same bits read as signed in one place
and unsigned in another, which real code does constantly -- the result is
recorded as ambiguous rather than silently resolved, and the renderer falls
back to the unsigned spelling.  A *contradiction* (a value used as both float
and pointer, say) is counted and reported; it usually means a lifting bug, and
hiding it would waste the signal.

No IDA imports: unit-testable standalone.
"""

from .ir import Op, EW
from . import lanes
from .lanes import W0

# ---------------------------------------------------------------------------
# the lattice
# ---------------------------------------------------------------------------

UNK = 0          # nothing known yet
INT = 1
FLT = 2
PTR = 3
VEC = 4

KIND_NAME = {UNK: "unk", INT: "int", FLT: "flt", PTR: "ptr", VEC: "vec"}

# When two kinds meet, the more specific one wins.  A pointer is also an
# integer on this machine, and float bits are also integer bits; the specific
# evidence is the one that came from an instruction that could not mean
# anything else.
_RANK = {UNK: 0, INT: 1, VEC: 2, FLT: 3, PTR: 4}


class Ty(object):
    """A recovered type: kind, width in bytes, signedness, pointee."""

    __slots__ = ("kind", "width", "signed", "pointee", "ambiguous")

    def __init__(self, kind=UNK, width=0, signed=None, pointee=None,
                 ambiguous=False):
        self.kind = kind
        self.width = width
        self.signed = signed
        self.pointee = pointee
        self.ambiguous = ambiguous

    def copy(self):
        return Ty(self.kind, self.width, self.signed, self.pointee,
                  self.ambiguous)

    def key(self):
        return (self.kind, self.width, self.signed,
                self.pointee.key() if self.pointee else None)

    def __eq__(self, other):
        return isinstance(other, Ty) and other.key() == self.key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.key())

    def __str__(self):
        if self.kind == PTR:
            return "%s *" % (self.pointee or Ty(INT, 4))
        if self.kind == FLT:
            return {4: "float", 8: "double"}.get(self.width, "float")
        if self.kind == VEC:
            return "vec_%s%d" % (
                {1: "uchar", 2: "ushort", 4: "uint", 8: "ullong"}.get(
                    self.width or 1, "uchar"),
                16 // (self.width or 1))
        # A known width with no kind evidence is still an integer of that
        # width -- "qword" would throw away something we do know.
        if self.kind == INT or (self.kind == UNK and
                                self.width in (1, 2, 4, 8)):
            base = {1: "char", 2: "short", 4: "int", 8: "long long"}.get(
                self.width, "int")
            if self.signed is False or self.signed is None:
                return "unsigned " + base if base != "int" else "unsigned int"
            return base
        return "qword"

    __repr__ = __str__


def scalar(width, signed=None):
    return Ty(INT, width, signed)


def ptr_to(pointee):
    return Ty(PTR, 4, False, pointee)


def join(a, b, stats=None):
    """
    Combine two pieces of evidence about the same value.

    Monotone: the result is at least as specific as either input, so the
    fixpoint below terminates.
    """
    if a is None:
        return b
    if b is None:
        return a
    if a.kind == UNK and b.kind == UNK:
        w = max(a.width, b.width)
        s = a.signed if a.signed is not None else b.signed
        return Ty(UNK, w, s)

    kind = a.kind if _RANK[a.kind] >= _RANK[b.kind] else b.kind
    lo, hi = (b, a) if _RANK[a.kind] >= _RANK[b.kind] else (a, b)

    # A genuine contradiction is float evidence meeting pointer evidence:
    # nothing sensible is both.  VEC is *not* in that category -- every SPU
    # register is 128 bits, so "the whole quadword is used" and "the preferred
    # slot holds an address" are both true of the same register all the time,
    # and counting that as a conflict would bury the real signal.
    if stats is not None and {lo.kind, hi.kind} == {FLT, PTR}:
        stats["contradictions"] = stats.get("contradictions", 0) + 1

    # Width: the wider evidence wins, except that a pointer is always a word.
    width = max(a.width, b.width)
    if kind == PTR:
        width = 4

    signed = a.signed
    ambiguous = a.ambiguous or b.ambiguous
    if a.signed is None:
        signed = b.signed
    elif b.signed is not None and b.signed != a.signed:
        # Read both ways.  Extremely common and not an error.
        signed = None
        ambiguous = True

    pointee = hi.pointee or lo.pointee
    if a.pointee is not None and b.pointee is not None:
        pointee = join(a.pointee, b.pointee, stats)

    return Ty(kind, width, signed, pointee, ambiguous)


# ---------------------------------------------------------------------------
# width from demand
# ---------------------------------------------------------------------------

_W_FROM_MASK = ((0x0008, 1), (0x000C, 2), (0x000F, 4), (0x00FF, 8))


def width_from_demand(mask):
    """
    The narrowest scalar that covers everything read from a value.

    Byte 3 alone is a char (the preferred slot is right-aligned for sub-word
    data), bytes 2..3 a short, 0..3 an int, 0..7 a long long.  Anything wider
    or scattered is a vector.
    """
    if mask == lanes.NONE:
        return 0
    for m, w in _W_FROM_MASK:
        if not (mask & ~m):
            return w
    return 16


# ---------------------------------------------------------------------------
# constraint generation
# ---------------------------------------------------------------------------

_FLOAT_OPS = frozenset((Op.FADD, Op.FSUB, Op.FMUL, Op.FMA, Op.FMS, Op.FNMS,
                        Op.FNMA, Op.FI, Op.FREST, Op.FRSQEST))
_FCMP_OPS = frozenset((Op.FCMPEQ, Op.FCMPGT, Op.FCMPMEQ, Op.FCMPMGT))
# Element-wise integer ops: the element width is the type width when the value
# really is a vector.
_LANEWISE = frozenset((Op.ADD, Op.SUB, Op.MUL, Op.CMPEQ, Op.CMPGT,
                       Op.CMPGTU, Op.SHL, Op.SHR, Op.SAR, Op.ROL,
                       Op.AVGB, Op.ABSDB, Op.CNTB, Op.CLZ))


class _Solver(object):

    def __init__(self, func, demand, groups=None):
        self.func = func
        self.demand = demand
        self.ty = {}
        self.stats = {"contradictions": 0, "ambiguous": 0}
        # Values that share a printed variable must share a type.
        self.rep = {}
        if groups:
            for g in groups:
                g = list(g)
                for k in g:
                    self.rep[k] = g[0]

    def _r(self, key):
        return self.rep.get(key, key)

    def get(self, key):
        return self.ty.get(self._r(key))

    def add(self, key, ty):
        if ty is None:
            return False
        k = self._r(key)
        old = self.ty.get(k)
        new = join(old, ty, self.stats)
        if old is None or new.key() != old.key():
            self.ty[k] = new
            return True
        return False

    def add_val(self, v, ty):
        return self.add(v.key(), ty) if v.is_var else False

    # -- seeding -----------------------------------------------------------

    def seed(self):
        """
        Width from demand -- but only where demand says something.

        A mask of ALL is the *default*, not evidence: the ABI operand list on
        every call and return demands all sixteen bytes because a callee might
        read them.  Treating that as "this is a vector" would make every value
        that reaches a return look like one.
        """
        for insn in self.func.insns():
            d = insn.defines()
            if d is None:
                continue
            mask = self.demand.get(d.key(), lanes.ALL)
            if mask == lanes.ALL:
                continue
            w = width_from_demand(mask)
            if w and w != 16:
                self.add(d.key(), Ty(UNK, w))

    # -- one pass over the instructions ------------------------------------

    def step(self):
        changed = False
        for insn in self.func.insns():
            changed |= self._constrain(insn)
        return changed

    def _constrain(self, insn):
        op = insn.op
        ch = False
        d = insn.defines()
        srcs = insn.srcs

        # -- memory: the strongest evidence there is ------------------------
        if op == Op.LOAD:
            w = int(insn.ew)
            ch |= self.add(d.key(), Ty(UNK, w)) if d else False
            ch |= self.add_val(srcs[1], ptr_to(Ty(UNK, w)))
            return ch
        if op == Op.STORE:
            w = int(insn.ew)
            ch |= self.add_val(srcs[1], ptr_to(Ty(UNK, w)))
            ch |= self.add_val(srcs[2], Ty(UNK, w))
            return ch
        if op in (Op.LOADQ, Op.LOADU):
            ch |= self.add(d.key(), Ty(VEC, 1)) if d else False
            ch |= self.add_val(srcs[1], ptr_to(Ty(VEC, 1)))
            return ch
        if op == Op.STOREQ:
            ch |= self.add_val(srcs[1], ptr_to(Ty(VEC, 1)))
            ch |= self.add_val(srcs[2], Ty(VEC, 1))
            return ch

        # -- floating point -------------------------------------------------
        if op in _FLOAT_OPS or op in _FCMP_OPS:
            w = 8 if insn.ew == EW.D else 4
            ft = Ty(FLT, w, True)
            for s in srcs:
                ch |= self.add_val(s, ft)
            if d is not None and op in _FLOAT_OPS:
                ch |= self.add(d.key(), ft)
            return ch
        if op in (Op.CSFLT, Op.CUFLT):
            ch |= self.add_val(srcs[0], scalar(4, op == Op.CSFLT))
            if d:
                ch |= self.add(d.key(), Ty(FLT, 4, True))
            return ch
        if op in (Op.CFLTS, Op.CFLTU):
            ch |= self.add_val(srcs[0], Ty(FLT, 4, True))
            if d:
                ch |= self.add(d.key(), scalar(4, op == Op.CFLTS))
            return ch

        # -- signedness, straight from the opcode ---------------------------
        if op in (Op.CMPGT, Op.CMPGTU):
            t = scalar(int(insn.ew), op == Op.CMPGT)
            for s in srcs:
                ch |= self.add_val(s, t)
            return ch
        if op in (Op.SAR, Op.SHR):
            ch |= self.add_val(srcs[0], scalar(int(insn.ew), op == Op.SAR))
            if d:
                ch |= self.add(d.key(), scalar(int(insn.ew), op == Op.SAR))
            return ch
        if op == Op.EXTS:
            w = int(insn.ew)
            ch |= self.add_val(srcs[0], scalar(max(w // 2, 1), True))
            if d:
                ch |= self.add(d.key(), scalar(w, True))
            return ch

        # -- pointer arithmetic ---------------------------------------------
        if op in (Op.ADD, Op.SUB) and insn.ew == EW.W and len(srcs) == 2:
            for i in (0, 1):
                t = self.get(srcs[i].key()) if srcs[i].is_var else None
                if t is not None and t.kind == PTR and d is not None:
                    ch |= self.add(d.key(), t)
            td = self.get(d.key()) if d is not None else None
            if td is not None and td.kind == PTR:
                # base + constant offset keeps the pointer type
                for i in (0, 1):
                    if srcs[i].is_const:
                        ch |= self.add_val(srcs[1 - i], td)
            # Deliberately no early return: address arithmetic is still
            # arithmetic, so the lane-wise rule below applies too.  Pointer
            # evidence outranks integer evidence in `join`, so adding it
            # cannot demote a pointer back to an int.

        # -- control flow ----------------------------------------------------
        # A branch condition and an indirect branch target are both read out
        # of the preferred slot, so both are word-sized whatever else is true
        # of the register holding them.
        if op == Op.CJMP:
            ch |= self.add_val(srcs[0], Ty(UNK, 4))
            return ch
        if op == Op.CIJMP:
            ch |= self.add_val(srcs[0], Ty(UNK, 4))
            ch |= self.add_val(srcs[1], Ty(UNK, 4))
            return ch
        if op in (Op.IJMP, Op.ICALL) and srcs:
            ch |= self.add_val(srcs[0], Ty(UNK, 4))
            return ch

        # -- copies and joins -----------------------------------------------
        if op in (Op.MOV, Op.PHI):
            if d is None:
                return ch
            td = self.get(d.key())
            for s in srcs:
                if s.is_var:
                    ch |= self.add(d.key(), self.get(s.key()))
                    ch |= self.add_val(s, td)
            return ch

        # -- lane-wise integer work ------------------------------------------
        #
        # Scalar or vector?  The scalarisation pass already answered that:
        # `insn.scalar` means demand analysis proved nothing reads outside the
        # preferred slot.  Guessing again here would be strictly worse.
        if op in _LANEWISE and insn.ew != EW.Q:
            w = int(insn.ew)
            t = scalar(w) if insn.scalar else Ty(VEC, w)
            if d is not None:
                ch |= self.add(d.key(), t)
            for s in srcs:
                ch |= self.add_val(s, t)
            return ch

        # Anything still touching whole quadwords is vector work.
        if op in (Op.SHUFB, Op.SELB, Op.GB, Op.FSM, Op.GENCTL, Op.QROTBY,
                  Op.QROTBI, Op.QROTMBY, Op.QROTMBI, Op.QSHLBY, Op.QSHLBI):
            if d is not None:
                ch |= self.add(d.key(), Ty(VEC, 1))
            return ch
        return ch

    # -- driver -------------------------------------------------------------

    def solve(self, rounds=24):
        self.seed()
        for _ in range(rounds):
            if not self.step():
                break
        for t in self.ty.values():
            if t.ambiguous:
                self.stats["ambiguous"] += 1
        return self.ty, self.stats


def infer(func, demand, groups=None):
    """
    Recover types for every SSA value.

    ``groups`` are sets of SSA keys that must share a type -- the phi webs the
    renderer will print as one variable.  Returns ``(types, stats)`` where
    ``types`` maps an SSA key (or its group representative) to a :class:`Ty`.
    """
    s = _Solver(func, demand, groups)
    tys, stats = s.solve()
    return _Types(tys, s.rep), stats


class _Types(object):
    """Lookup that follows the group representative."""

    def __init__(self, tys, rep):
        self._t = tys
        self._rep = rep

    def of(self, key, default=None):
        return self._t.get(self._rep.get(key, key), default)

    def of_var(self, v, default=None):
        return self.of(v.key(), default) if v.is_var else default

    def __len__(self):
        return len(self._t)

    def counts(self):
        """Count by what each type actually renders as, not by raw kind."""
        out = {}
        for t in self._t.values():
            k = KIND_NAME[t.kind]
            if t.kind == UNK:
                k = "int" if t.width in (1, 2, 4, 8) else "unk"
            out[k] = out.get(k, 0) + 1
        return out
