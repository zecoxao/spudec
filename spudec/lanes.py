"""
Byte-level demand analysis over the SSA graph.

For every SSA value this computes a 16-bit mask saying which *bytes* of the
128-bit register anything actually reads.  Bit ``i`` of the mask is byte ``i``,
and byte 0 is the most significant (SPU numbering), so the preferred slot --
bytes 0..3, the lane the hardware uses for addresses, branch conditions and
scalar operands -- is mask ``0x000F``.

This is what makes scalarisation possible.  The SPU has no scalar registers, so
"this quadword is really an int" is not something the instruction encoding
tells you; it is a property of how the value is *consumed*.  A value whose
demand is contained in 0x000F is a scalar in the preferred slot, and the
operations producing it can be printed as ordinary arithmetic.

The analysis is backward and monotone (masks only grow), iterated to a
fixpoint.  It is conservative everywhere it is unsure: an unknown shuffle
control or an intrinsic demands all sixteen bytes.

Precision notes that matter in practice:

* Bitwise ops are byte-exact -- byte ``i`` of ``and``/``or``/``xor`` depends
  only on byte ``i`` of the inputs.  Arithmetic is not: a carry crosses bytes,
  so demand expands to the whole element.
* ``shufb`` and the quadword rotates are exact byte permutations *when their
  control or count is a known constant*, which after constant folding is the
  usual case.  That exactness is what lets a scalar load or store be
  recognised at all.

No IDA imports: unit-testable standalone.
"""

from .ir import Op, EW, ABI_OPS
from .sem import bytes_of, word0

ALL = 0xFFFF          # every byte
NONE = 0x0000
W0 = 0x000F           # preferred slot: bytes 0..3
H0 = 0x000C           # bytes 2..3 -- the halfword a `brhz` tests
B0 = 0x0008           # byte 3     -- where a scalar char sits in slot 0

# Masks for "the low n bytes of the preferred slot", i.e. where STORE takes
# its value from.
SLOT_LOW = {EW.B: 0x0008, EW.H: 0x000C, EW.W: 0x000F, EW.D: 0x00FF}
# Masks for "the first n bytes", i.e. where LOAD leaves its result.
LEAD = {EW.B: 0x0001, EW.H: 0x0003, EW.W: 0x000F, EW.D: 0x00FF}


def expand(mask, ew):
    """Grow ``mask`` so that any touched element is demanded whole."""
    n = int(ew)
    if n == 1:
        return mask
    out = 0
    full = (1 << n) - 1
    for e in range(16 // n):
        sel = full << (e * n)
        if mask & sel:
            out |= sel
    return out


def rotl_mask(mask, n):
    """Demand on the source of a left-rotate-by-``n``-bytes."""
    n &= 0xF
    if n == 0:
        return mask
    return ((mask << n) | (mask >> (16 - n))) & ALL


# Ops whose result is a pure byte-wise function of the inputs.
_BYTEWISE = frozenset((Op.AND, Op.OR, Op.XOR, Op.NAND, Op.NOR, Op.ANDC,
                       Op.ORC, Op.EQV, Op.NOT, Op.SELB))

# Ops that mix bits within an element only.
_ELEMENTWISE = frozenset((
    Op.ADD, Op.SUB, Op.MUL, Op.CG, Op.BG, Op.ADDX, Op.SUBX, Op.CGX, Op.BGX,
    Op.SHL, Op.SHR, Op.SAR, Op.ROL, Op.ROTM, Op.ROTMA, Op.EXTS,
    Op.CMPEQ, Op.CMPGT, Op.CMPGTU, Op.CLZ, Op.CNTB, Op.AVGB, Op.ABSDB,
    Op.MPY, Op.MPYU, Op.MPYH, Op.MPYS, Op.MPYHH, Op.MPYHHU, Op.MPYHHA,
    Op.MPYHHAU, Op.MPYA,
    Op.FADD, Op.FSUB, Op.FMUL, Op.FMA, Op.FMS, Op.FNMS, Op.FNMA,
    Op.FCMPEQ, Op.FCMPGT, Op.FCMPMEQ, Op.FCMPMGT, Op.FI, Op.FREST,
    Op.FRSQEST, Op.CSFLT, Op.CUFLT, Op.CFLTS, Op.CFLTU,
))

_QUAD_SHIFT = {
    Op.QROTBY: "rot", Op.QROTBI: "bits", Op.QROTMBY: "shr",
    Op.QROTMBI: "bits", Op.QSHLBY: "shl", Op.QSHLBI: "bits",
}


def _const_of(v, defs):
    """The constant value behind ``v``, or None."""
    if v.is_const:
        return v.val
    d = defs.get(v.key())
    while d is not None and d.op == Op.MOV:
        s = d.srcs[0]
        if s.is_const:
            return s.val
        d = defs.get(s.key())
    if d is not None and d.op == Op.CONST:
        return d.srcs[0].val
    return None


def _contrib(insn, out_mask, defs):
    """
    Demand contributed to each source of ``insn``, given ``out_mask`` on its
    result.  Returns a list parallel to ``insn.srcs``.
    """
    op = insn.op
    n = len(insn.srcs)

    # Terminators and side-effecting ops read their operands regardless of
    # whether anything reads a result.
    if op in ABI_OPS:
        return [ALL] * n
    if op == Op.CJMP:
        kind = insn.aux[1] if insn.aux else "z"
        return [H0 if kind in ("hz", "hnz") else W0]
    if op == Op.CIJMP:
        kind = insn.aux or "z"
        return [H0 if kind in ("hz", "hnz") else W0, W0]
    if op in (Op.STOREQ,):
        return [ALL, W0, ALL]                    # mem, addr, full quadword
    if op == Op.STORE:
        return [ALL, W0, SLOT_LOW[insn.ew]]      # only the scalar bytes
    if op in (Op.LOADQ, Op.LOAD, Op.LOADU):
        return [ALL, W0]
    if op in (Op.RDCH, Op.RCHCNT, Op.MFSPR, Op.SYNC):
        return [ALL] * n
    if op in (Op.WRCH, Op.MTSPR):
        return [ALL] * n
    if op in (Op.HALT, Op.INTRINSIC):
        return [ALL] * n

    if out_mask == NONE:
        return [NONE] * n

    if op in (Op.MOV, Op.PHI):
        return [out_mask] * n
    if op == Op.CONST:
        return [NONE] * n
    if op in _BYTEWISE:
        return [out_mask] * n
    if op in _ELEMENTWISE:
        m = expand(out_mask, insn.ew)
        return [m] * n
    if op == Op.GENCTL:
        return [W0]
    if op == Op.FSM:
        return [W0]                              # reads only the preferred slot
    if op == Op.GB:
        return [ALL]
    if op == Op.ORX:
        return [ALL]
    if op == Op.SHUFB:
        ctl = _const_of(insn.srcs[2], defs)
        if ctl is None:
            return [ALL, ALL, ALL]
        da = db = 0
        for i, cb in enumerate(bytes_of(ctl)):
            if not (out_mask >> i) & 1:
                continue
            if (cb & 0xC0) == 0x80 or (cb & 0xE0) == 0xC0 or (cb & 0xE0) == 0xE0:
                continue                         # constant byte, reads nothing
            idx = cb & 0x1F
            if idx < 16:
                da |= 1 << idx
            else:
                db |= 1 << (idx - 16)
        return [da, db, W0]
    if op in _QUAD_SHIFT:
        cnt = _const_of(insn.srcs[1], defs)
        if cnt is None:
            return [ALL, W0]
        c = word0(cnt)
        kind = _QUAD_SHIFT[op]
        if kind == "bits":
            # A bit-granular shift can pull from the neighbouring byte.
            return [ALL, W0]
        if kind == "rot":
            return [rotl_mask(out_mask, c & 0xF), W0]
        if kind == "shl":
            k = c & 0x1F
            return [NONE if k >= 16 else (out_mask << k) & ALL, W0]
        if kind == "shr":
            k = (-c) & 0x1F
            return [NONE if k >= 16 else (out_mask >> k) & ALL, W0]
    return [ALL] * n


def compute_demand(func):
    """
    Returns ``{(reg, ver): mask}`` -- for each SSA value, which of its bytes
    are read.  Values absent from the map are read by nothing.
    """
    defs = {}
    for i in func.insns():
        d = i.defines()
        if d is not None:
            defs[d.key()] = i

    demand = {}
    # Reverse order converges fast for straight-line code; loops need the
    # outer fixpoint anyway.
    order = [i for b in reversed(func.blocks) for i in reversed(b.insns)]

    changed = True
    while changed:
        changed = False
        for insn in order:
            d = insn.defines()
            out = demand.get(d.key(), NONE) if d is not None else NONE
            for src, add in zip(insn.srcs, _contrib(insn, out, defs)):
                if not src.is_var or add == NONE:
                    continue
                k = src.key()
                cur = demand.get(k, NONE)
                new = cur | add
                if new != cur:
                    demand[k] = new
                    changed = True
    return demand


def is_scalar(mask):
    """True if nothing outside the preferred slot is read."""
    return mask != NONE and (mask & ~W0) == 0


def fits(mask, ew, table=LEAD):
    """True if ``mask`` is contained in the given placement for width ``ew``."""
    return mask != NONE and (mask & ~table[ew]) == 0


def narrowest(mask, table=LEAD):
    """The narrowest element width whose placement contains ``mask``."""
    for ew in (EW.B, EW.H, EW.W, EW.D):
        if fits(mask, ew, table):
            return ew
    return None
