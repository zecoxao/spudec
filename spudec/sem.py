"""
Concrete semantics for constant folding.

``evaluate()`` computes the exact 128-bit result of an operation when all of
its inputs are known.  This is not decoration: the SPU builds scalar memory
accesses out of ``cwd``/``shufb`` control masks, and folding those masks is
precisely what lets a later pass recognise

    cwd   ctl, 0(p)          ctl = 0x00010203 10111213 ...
    lqd   old, 0(p)
    shufb new, val, old, ctl
    stqd  new, 0(p)

as ``*(int *)p = val``.  Without an exact shufb/genctl evaluator that idiom
stays an opaque permutation forever.

Lane order is big-endian throughout: element 0 is the most significant.
Returns ``None`` for anything not modelled (floating point in particular --
folding IEEE semantics in Python ints is not worth the fidelity risk).

No IDA imports: unit-testable standalone.
"""

from .ir import Op, EW, MASK128

_F32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# lane access
# ---------------------------------------------------------------------------


def get_lane(v, ew, i):
    bits = int(ew) * 8
    return (v >> (128 - bits * (i + 1))) & ((1 << bits) - 1)


def lanes(v, ew):
    return [get_lane(v, ew, i) for i in range(16 // int(ew))]


def pack(vals, ew):
    bits = int(ew) * 8
    m = (1 << bits) - 1
    out = 0
    for x in vals:
        out = (out << bits) | (x & m)
    return out & MASK128


def bytes_of(v):
    return [(v >> (120 - 8 * i)) & 0xFF for i in range(16)]


def from_bytes(bs):
    out = 0
    for b in bs:
        out = (out << 8) | (b & 0xFF)
    return out & MASK128


def sext(v, bits):
    m = 1 << (bits - 1)
    v &= (1 << bits) - 1
    return (v ^ m) - m


def word0(v):
    return (v >> 96) & _F32


# ---------------------------------------------------------------------------
# permute / control-mask generation
# ---------------------------------------------------------------------------


def shufb(a, b, c):
    """
    shufb rt, ra, rb, rc.

    Control byte semantics, checked in this order (the patterns overlap):
        0b10xxxxxx -> 0x00
        0b110xxxxx -> 0xFF
        0b111xxxxx -> 0x80
        otherwise  -> byte (c & 0x1F) of the concatenation ra:rb
    """
    src = bytes_of(a) + bytes_of(b)
    out = []
    for cb in bytes_of(c):
        if (cb & 0xC0) == 0x80:
            out.append(0x00)
        elif (cb & 0xE0) == 0xC0:
            out.append(0xFF)
        elif (cb & 0xE0) == 0xE0:
            out.append(0x80)
        else:
            out.append(src[cb & 0x1F])
    return from_bytes(out)


# The identity control mask selects every byte from RB (the second shufb
# operand), which by convention is the original quadword being modified.
_IDENTITY_CTL = list(range(0x10, 0x20))


def genctl(addr, ew):
    """
    cbd/cbx/chd/chx/cwd/cwx/cdd/cdx.

    Produces the shufb control mask that inserts the scalar sitting in RA's
    preferred slot into the quadword in RB, at the position implied by ``addr``.
    The inserted control values are the byte offsets of that scalar *within the
    preferred slot*: a byte lives at offset 3, a halfword at 2..3, a word at
    0..3, a doubleword at 0..7.
    """
    n = int(ew)
    t = addr & 0xF
    j = t & ~(n - 1)                       # align down to the element size
    out = list(_IDENTITY_CTL)
    first = 4 - n if n <= 4 else 0         # where the scalar sits in slot 0
    for k in range(n):
        out[j + k] = first + k
    return from_bytes(out)


def fsm(v, ew):
    """Form select mask: element j is all-ones iff bit (n-1-j) of RA.word0."""
    n = 16 // int(ew)
    w = word0(v)
    bits = int(ew) * 8
    full = (1 << bits) - 1
    return pack([full if (w >> (n - 1 - j)) & 1 else 0 for j in range(n)], ew)


def gb(v, ew):
    """Gather the LSB of each element into the low bits of the preferred slot."""
    n = 16 // int(ew)
    out = 0
    for j, e in enumerate(lanes(v, ew)):
        if e & 1:
            out |= 1 << (n - 1 - j)
    return (out & _F32) << 96


# ---------------------------------------------------------------------------
# quadword shifts
# ---------------------------------------------------------------------------


def qrotby(v, n):
    n = (n & 0xF) * 8
    return ((v << n) | (v >> (128 - n))) & MASK128 if n else v


def qrotbi(v, n):
    n &= 0x7
    return ((v << n) | (v >> (128 - n))) & MASK128 if n else v


def qrotmby(v, n):
    n = (-n) & 0x1F
    return 0 if n >= 16 else (v >> (n * 8))


def qrotmbi(v, n):
    n = (-n) & 0x7
    return v >> n


def qshlby(v, n):
    n &= 0x1F
    return 0 if n >= 16 else (v << (n * 8)) & MASK128


def qshlbi(v, n):
    return (v << (n & 0x7)) & MASK128


# ---------------------------------------------------------------------------
# element-wise helpers
# ---------------------------------------------------------------------------


def _binlane(a, b, ew, fn):
    return pack([fn(x, y) for x, y in zip(lanes(a, ew), lanes(b, ew))], ew)


def _clz32(x):
    n = 0
    for i in range(31, -1, -1):
        if (x >> i) & 1:
            break
        n += 1
    return n


def _popcount(x):
    return bin(x).count("1")


def _shl(x, c, bits):
    c &= 0x3F if bits == 32 else 0x1F
    return 0 if c >= bits else (x << c) & ((1 << bits) - 1)


def _shr(x, c, bits):
    c &= 0x3F if bits == 32 else 0x1F
    return 0 if c >= bits else x >> c

def _sar(x, c, bits):
    c &= 0x3F if bits == 32 else 0x1F
    if c >= bits:
        c = bits - 1
    return (sext(x, bits) >> c) & ((1 << bits) - 1)


def _rol(x, c, bits):
    c &= bits - 1
    return ((x << c) | (x >> (bits - c))) & ((1 << bits) - 1) if c else x


# ---------------------------------------------------------------------------
# the evaluator
# ---------------------------------------------------------------------------


def evaluate(op, ew, args, aux=None):
    """
    Evaluate ``op`` over the fully-known 128-bit integer ``args``.
    Returns the 128-bit result, or None if the operation is not modelled.
    """
    n = int(ew)
    bits = n * 8
    full = (1 << bits) - 1

    try:
        if op in (Op.MOV, Op.CONST):
            return args[0]

        # -- bitwise -------------------------------------------------------
        if op == Op.AND:
            return args[0] & args[1]
        if op == Op.OR:
            return args[0] | args[1]
        if op == Op.XOR:
            return args[0] ^ args[1]
        if op == Op.NAND:
            return ~(args[0] & args[1]) & MASK128
        if op == Op.NOR:
            return ~(args[0] | args[1]) & MASK128
        if op == Op.ANDC:
            return args[0] & ~args[1] & MASK128
        if op == Op.ORC:
            return (args[0] | ~args[1]) & MASK128
        if op == Op.EQV:
            return ~(args[0] ^ args[1]) & MASK128
        if op == Op.NOT:
            return ~args[0] & MASK128

        # -- arithmetic ----------------------------------------------------
        if op == Op.ADD:
            return _binlane(args[0], args[1], ew, lambda x, y: (x + y) & full)
        if op == Op.SUB:
            return _binlane(args[0], args[1], ew, lambda x, y: (x - y) & full)
        if op == Op.CG:
            return _binlane(args[0], args[1], ew,
                            lambda x, y: 1 if (x + y) > full else 0)
        if op == Op.BG:
            return _binlane(args[0], args[1], ew,
                            lambda x, y: 1 if y >= x else 0)
        if op == Op.ADDX:
            a, b, t = args
            return pack([(x + y + (c & 1)) & full for x, y, c in
                         zip(lanes(a, ew), lanes(b, ew), lanes(t, ew))], ew)
        if op == Op.SUBX:
            a, b, t = args
            return pack([(y + (~x & full) + (c & 1)) & full for x, y, c in
                         zip(lanes(a, ew), lanes(b, ew), lanes(t, ew))], ew)
        if op == Op.CGX:
            a, b, t = args
            return pack([1 if (x + y + (c & 1)) > full else 0 for x, y, c in
                         zip(lanes(a, ew), lanes(b, ew), lanes(t, ew))], ew)
        if op == Op.BGX:
            a, b, t = args
            return pack([1 if (y + (~x & full) + (c & 1)) > full else 0
                         for x, y, c in
                         zip(lanes(a, ew), lanes(b, ew), lanes(t, ew))], ew)

        # -- multiply (16x16 -> 32, per word) -------------------------------
        if op == Op.MPY:
            return _binlane(args[0], args[1], EW.W,
                            lambda x, y: (sext(x & 0xFFFF, 16) *
                                          sext(y & 0xFFFF, 16)) & _F32)
        if op == Op.MPYU:
            return _binlane(args[0], args[1], EW.W,
                            lambda x, y: ((x & 0xFFFF) * (y & 0xFFFF)) & _F32)
        if op == Op.MPYH:
            return _binlane(args[0], args[1], EW.W,
                            lambda x, y: ((sext(x >> 16, 16) *
                                           (y & 0xFFFF)) << 16) & _F32)
        if op == Op.MPYHH:
            return _binlane(args[0], args[1], EW.W,
                            lambda x, y: (sext(x >> 16, 16) *
                                          sext(y >> 16, 16)) & _F32)
        if op == Op.MPYHHU:
            return _binlane(args[0], args[1], EW.W,
                            lambda x, y: ((x >> 16) * (y >> 16)) & _F32)
        if op == Op.MPYS:
            return _binlane(args[0], args[1], EW.W,
                            lambda x, y: ((sext(x & 0xFFFF, 16) *
                                           sext(y & 0xFFFF, 16)) >> 16) & _F32)

        # -- shifts --------------------------------------------------------
        if op in (Op.SHL, Op.SHR, Op.SAR, Op.ROL):
            fn = {Op.SHL: _shl, Op.SHR: _shr,
                  Op.SAR: _sar, Op.ROL: _rol}[op]
            return _binlane(args[0], args[1], ew,
                            lambda x, c: fn(x, c, bits))
        if op == Op.ROTM:
            return _binlane(args[0], args[1], ew,
                            lambda x, c: _shr(x, -c, bits))
        if op == Op.ROTMA:
            return _binlane(args[0], args[1], ew,
                            lambda x, c: _sar(x, -c, bits))

        # -- quadword shifts (count from the preferred slot) ---------------
        if op in (Op.QROTBY, Op.QROTBI, Op.QROTMBY, Op.QROTMBI,
                  Op.QSHLBY, Op.QSHLBI):
            fn = {Op.QROTBY: qrotby, Op.QROTBI: qrotbi,
                  Op.QROTMBY: qrotmby, Op.QROTMBI: qrotmbi,
                  Op.QSHLBY: qshlby, Op.QSHLBI: qshlbi}[op]
            return fn(args[0], word0(args[1]))

        # -- compares ------------------------------------------------------
        if op == Op.CMPEQ:
            return _binlane(args[0], args[1], ew,
                            lambda x, y: full if x == y else 0)
        if op == Op.CMPGT:
            return _binlane(args[0], args[1], ew,
                            lambda x, y: full if sext(x, bits) > sext(y, bits)
                            else 0)
        if op == Op.CMPGTU:
            return _binlane(args[0], args[1], ew,
                            lambda x, y: full if x > y else 0)

        # -- permute / select ----------------------------------------------
        if op == Op.SELB:
            a, b, c = args
            return (a & ~c | b & c) & MASK128
        if op == Op.SHUFB:
            return shufb(args[0], args[1], args[2])
        if op == Op.GENCTL:
            return genctl(word0(args[0]), ew)
        if op == Op.FSM:
            return fsm(args[0], ew)
        if op == Op.GB:
            return gb(args[0], ew)

        # -- misc unary ----------------------------------------------------
        if op == Op.CLZ:
            return pack([_clz32(x) for x in lanes(args[0], EW.W)], EW.W)
        if op == Op.CNTB:
            return pack([_popcount(x) for x in lanes(args[0], EW.B)], EW.B)
        if op == Op.ORX:
            w = 0
            for x in lanes(args[0], EW.W):
                w |= x
            return (w & _F32) << 96
        if op == Op.EXTS:
            # Sign-extend the low half of each ``ew``-wide element.
            half = bits // 2
            return pack([sext(x & ((1 << half) - 1), half) & full
                         for x in lanes(args[0], ew)], ew)

    except (IndexError, ValueError, ZeroDivisionError):
        return None

    return None
