"""
SPU decompiler: IR core.

Design notes
------------
The SPU is a 128-bit SIMD machine with 128 architectural registers and *no*
scalar register file.  Modelling registers as anything narrower than 128 bits
loses information, so every IR value is a 128-bit bitvector and every operation
carries an element-width tag (``ew``).  Readability (recovering "this quadword
is really an int") is a later pass's job, not the lifter's.

Bit numbering follows the SPU ISA: the register is big-endian, byte 0 is the
most significant byte.  In a Python int holding a 128-bit register value, byte
``j`` occupies bits ``[127-8j : 120-8j]``.  Word 0 (bytes 0..3, bits 127..96) is
the *preferred slot*: the lane the hardware uses for addresses, branch
conditions and scalar operands.

This module deliberately imports nothing from IDA so it can be unit-tested
standalone.
"""

from enum import IntEnum

MASK128 = (1 << 128) - 1

# ---------------------------------------------------------------------------
# element widths
# ---------------------------------------------------------------------------


class EW(IntEnum):
    """Element width in bytes."""
    B = 1
    H = 2
    W = 4
    D = 8
    Q = 16

    @property
    def bits(self):
        return int(self) * 8

    @property
    def count(self):
        """Number of elements of this width in a quadword."""
        return 16 // int(self)


EW_SUFFIX = {EW.B: "b", EW.H: "h", EW.W: "w", EW.D: "d", EW.Q: "q"}


# ---------------------------------------------------------------------------
# opcodes
# ---------------------------------------------------------------------------


class Op(IntEnum):
    # -- pseudo ------------------------------------------------------------
    PHI = 0
    MOV = 1          # dst = src0
    CONST = 2        # dst = Const
    INTRINSIC = 3    # opaque; aux holds the mnemonic
    UNDEF = 4        # value is unknown here (a register clobbered by a call)

    # -- integer arithmetic (element-wise, honours .ew) ---------------------
    ADD = 10
    SUB = 11         # dst = src0 - src1  (note: SPU `sf` is rb - ra, the
                     #                     lifter swaps operands for us)
    MUL = 12
    CG = 13          # carry generate
    BG = 14          # borrow generate
    ADDX = 15
    SUBX = 16
    CGX = 17
    BGX = 18
    ABSDB = 19
    AVGB = 20
    SUMB = 21

    # -- multiply family (16x16 -> 32 etc; see lifter for exact semantics) --
    MPY = 30         # signed low halves
    MPYU = 31
    MPYH = 32
    MPYS = 33
    MPYHH = 34
    MPYHHU = 35
    MPYHHA = 36
    MPYHHAU = 37
    MPYA = 38

    # -- bitwise -----------------------------------------------------------
    AND = 40
    OR = 41
    XOR = 42
    NAND = 43
    NOR = 44
    ANDC = 45        # src0 & ~src1
    ORC = 46         # src0 | ~src1
    EQV = 47         # ~(src0 ^ src1)
    NOT = 48

    # -- shifts / rotates --------------------------------------------------
    # SHL/SHR/SAR take a *positive* count and are what the simplifier emits
    # once it has proved the SPU's negated-count encoding away.
    SHL = 50
    SHR = 51
    SAR = 52
    ROL = 53
    ROTM = 54        # raw SPU semantics: logical right by (-count) mod 2*bits
    ROTMA = 55       # raw SPU semantics: arithmetic right by (-count)

    # -- quadword shifts / rotates (whole 128-bit register) ----------------
    QROTBY = 60      # rotate left by bytes
    QROTBI = 61      # rotate left by bits
    QROTMBY = 62     # shift right by bytes, SPU negated count
    QROTMBI = 63
    QSHLBY = 64
    QSHLBI = 65

    # -- permute / select --------------------------------------------------
    SHUFB = 70       # shufb rt, ra, rb, rc
    SELB = 71        # (ra & ~rc) | (rb & rc)
    GENCTL = 72      # cbd/cbx/chd/chx/cwd/cwx/cdd/cdx; aux = EW
    GB = 73          # gather bits; aux = EW
    FSM = 74         # form select mask; aux = EW

    # -- misc unary --------------------------------------------------------
    CLZ = 80
    CNTB = 81
    ORX = 82
    EXTS = 83        # xsbh/xshw/xswd; ew = *destination* element width

    # -- compares (produce all-ones / all-zeros masks) ---------------------
    CMPEQ = 90
    CMPGT = 91       # signed
    CMPGTU = 92      # logical/unsigned

    # -- floating point ----------------------------------------------------
    FADD = 100
    FSUB = 101
    FMUL = 102
    FMA = 103        # a*b + c
    FMS = 104        # a*b - c
    FNMS = 105       # c - a*b
    FNMA = 106       # -(a*b + c)
    FCMPEQ = 107
    FCMPGT = 108
    FCMPMEQ = 109    # magnitude compare
    FCMPMGT = 110
    FI = 111         # floating interpolate
    FREST = 112
    FRSQEST = 113
    FESD = 114       # extend single to double
    FRDS = 115       # round double to single
    CSFLT = 116      # signed int -> float, aux = scale
    CUFLT = 117
    CFLTS = 118      # float -> signed int, aux = scale
    CFLTU = 119
    DFTSV = 120

    # -- memory (threaded through the MEM pseudo-register) -----------------
    LOADQ = 130      # dst = loadq(mem, addr)          -- addr forced 16-aligned
    STOREQ = 131     # mem' = storeq(mem, addr, val)   -- addr forced 16-aligned

    # Recovered scalar accesses (see scalarize.py).  Byte placement is stated
    # exactly because the two directions are NOT symmetric below word width:
    #   LOAD.ew   dst bytes 0..ew-1 hold the loaded value (rest undefined),
    #             which is where `lqd`+`rotqby` actually leaves it.
    #   STORE.ew  takes the value from the low ew bytes of the preferred slot
    #             (byte 3 for a byte, bytes 2..3 for a halfword, 0..3 for a
    #             word, 0..7 for a doubleword), which is what `cwd`+`shufb`
    #             actually inserts.
    # At word width both conventions coincide, which is the common case.
    LOAD = 132       # dst = load.ew(mem, addr)     -- addr is NOT aligned
    STORE = 133      # mem' = store.ew(mem, addr, val)
    LOADU = 134      # dst = loadu(mem, addr) -- unaligned quadword at addr

    # -- control flow ------------------------------------------------------
    JMP = 140        # aux = target ea
    CJMP = 141       # src0 = condition; aux = (target ea, cond kind)
    CALL = 142       # aux = target ea
    ICALL = 143      # src0 = target register
    IJMP = 144       # src0 = target register
    RET = 145
    # Conditional indirect branch (biz/binz/bihz/bihnz): src0 = tested value,
    # src1 = target register, aux = test kind.  NOT a terminator -- control
    # falls through when the test fails, and IDA gives no edge for the taken
    # side because the target is computed.
    CIJMP = 146

    # -- side-effecting / system ------------------------------------------
    RDCH = 150       # dst = rdch(chain, ch)
    WRCH = 151       # chain' = wrch(chain, ch, val)
    RCHCNT = 152
    MFSPR = 153
    MTSPR = 154
    STOP = 155
    SYNC = 156
    NOP = 157
    HALT = 158       # heq/hgt/hlgt & immediate forms

    @property
    def is_terminator(self):
        return self in _TERMINATORS

    @property
    def has_side_effects(self):
        return self in _SIDE_EFFECTS


# A call is NOT a terminator: it returns and falls through.  Marking it one
# made a block-final call get sliced off with the terminator and vanish from
# the rendered statements entirely.
_TERMINATORS = frozenset(
    (Op.JMP, Op.CJMP, Op.IJMP, Op.RET, Op.STOP)
)

# Ops that must never be removed by dead-code elimination even when their
# result is unused.
_SIDE_EFFECTS = frozenset(
    (Op.STOREQ, Op.STORE, Op.WRCH, Op.RDCH, Op.RCHCNT, Op.MTSPR, Op.MFSPR,
     Op.STOP, Op.SYNC, Op.HALT, Op.INTRINSIC, Op.CALL, Op.ICALL, Op.CIJMP)
)

# Memory reads: removable when their result is unused (local store has no
# device semantics), but they must never be reordered past a store.
MEM_READS = frozenset((Op.LOADQ, Op.LOAD, Op.LOADU))
MEM_WRITES = frozenset((Op.STOREQ, Op.STORE))

OP_NAME = {
    Op.PHI: "phi", Op.MOV: "mov", Op.CONST: "const", Op.INTRINSIC: "intr",
    Op.UNDEF: "undef",
    Op.ADD: "add", Op.SUB: "sub", Op.MUL: "mul", Op.CG: "cg", Op.BG: "bg",
    Op.ADDX: "addx", Op.SUBX: "subx", Op.CGX: "cgx", Op.BGX: "bgx",
    Op.ABSDB: "absdb", Op.AVGB: "avgb", Op.SUMB: "sumb",
    Op.MPY: "mpy", Op.MPYU: "mpyu", Op.MPYH: "mpyh", Op.MPYS: "mpys",
    Op.MPYHH: "mpyhh", Op.MPYHHU: "mpyhhu", Op.MPYHHA: "mpyhha",
    Op.MPYHHAU: "mpyhhau", Op.MPYA: "mpya",
    Op.AND: "and", Op.OR: "or", Op.XOR: "xor", Op.NAND: "nand",
    Op.NOR: "nor", Op.ANDC: "andc", Op.ORC: "orc", Op.EQV: "eqv",
    Op.NOT: "not",
    Op.SHL: "shl", Op.SHR: "shr", Op.SAR: "sar", Op.ROL: "rol",
    Op.ROTM: "rotm", Op.ROTMA: "rotma",
    Op.QROTBY: "qrotby", Op.QROTBI: "qrotbi", Op.QROTMBY: "qrotmby",
    Op.QROTMBI: "qrotmbi", Op.QSHLBY: "qshlby", Op.QSHLBI: "qshlbi",
    Op.SHUFB: "shufb", Op.SELB: "selb", Op.GENCTL: "genctl",
    Op.GB: "gb", Op.FSM: "fsm",
    Op.CLZ: "clz", Op.CNTB: "cntb", Op.ORX: "orx", Op.EXTS: "exts",
    Op.CMPEQ: "cmpeq", Op.CMPGT: "cmpgt", Op.CMPGTU: "cmpgtu",
    Op.FADD: "fadd", Op.FSUB: "fsub", Op.FMUL: "fmul", Op.FMA: "fma",
    Op.FMS: "fms", Op.FNMS: "fnms", Op.FNMA: "fnma",
    Op.FCMPEQ: "fcmpeq", Op.FCMPGT: "fcmpgt", Op.FCMPMEQ: "fcmpmeq",
    Op.FCMPMGT: "fcmpmgt", Op.FI: "fi", Op.FREST: "frest",
    Op.FRSQEST: "frsqest", Op.FESD: "fesd", Op.FRDS: "frds",
    Op.CSFLT: "csflt", Op.CUFLT: "cuflt", Op.CFLTS: "cflts",
    Op.CFLTU: "cfltu", Op.DFTSV: "dftsv",
    Op.LOADQ: "loadq", Op.STOREQ: "storeq",
    Op.LOAD: "load", Op.STORE: "store", Op.LOADU: "loadu",
    Op.JMP: "jmp", Op.CJMP: "cjmp", Op.CALL: "call", Op.ICALL: "icall",
    Op.IJMP: "ijmp", Op.RET: "ret", Op.CIJMP: "cijmp",
    Op.RDCH: "rdch", Op.WRCH: "wrch", Op.RCHCNT: "rchcnt",
    Op.MFSPR: "mfspr", Op.MTSPR: "mtspr",
    Op.STOP: "stop", Op.SYNC: "sync", Op.NOP: "nop", Op.HALT: "halt",
}


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


class Value(object):
    __slots__ = ()
    is_const = False
    is_var = False


class Const(Value):
    """A 128-bit constant.  ``val`` is always masked to 128 bits."""

    __slots__ = ("val",)
    is_const = True

    def __init__(self, val):
        self.val = val & MASK128

    def __eq__(self, other):
        return isinstance(other, Const) and other.val == self.val

    def __hash__(self):
        return hash(("Const", self.val))

    # -- lane accessors ----------------------------------------------------
    def elem(self, ew, i):
        """Element ``i`` (0 = most significant) at element width ``ew``."""
        bits = int(ew) * 8
        sh = 128 - bits * (i + 1)
        return (self.val >> sh) & ((1 << bits) - 1)

    @property
    def word0(self):
        """Preferred slot (bytes 0..3)."""
        return self.val >> 96

    def fmt(self, ew=None):
        """
        Render the constant, preferring the lane view of the operation that
        consumes it.  ``clgtbi`` builds 0x7F in all sixteen bytes; printing
        that as ``#0x7f:b16`` says what the code meant, where the raw 128-bit
        value (or a word-lane view of it) does not.
        """
        v = self.val
        if v == 0:
            return "#0"
        for cand in ([ew] if ew is not None and ew != EW.Q else []) + \
                    [EW.W, EW.H, EW.B, EW.D]:
            bits = int(cand) * 8
            n = 16 // int(cand)
            lane = v >> (128 - bits)
            if all(((v >> (128 - bits * (i + 1))) & ((1 << bits) - 1)) == lane
                   for i in range(n)):
                return "#0x%x:%s%d" % (lane, EW_SUFFIX[cand], n)
        return "#0x%032x" % v

    def addr_str(self):
        """Address operands only use the preferred slot; show just that."""
        return "@0x%X" % (self.val >> 96)

    def scalar_str(self):
        """Preferred-slot value, printed as a plain scalar (signed if small)."""
        w = self.val >> 96
        if w >= 0x80000000 and (0x100000000 - w) <= 0xFFFF:
            return "-0x%X" % (0x100000000 - w)
        return "0x%X" % w

    def __str__(self):
        return self.fmt()

    __repr__ = __str__


class Var(Value):
    """
    An SSA-versioned reference to an architectural or pseudo register.

    Pre-SSA, ``ver`` is -1 and the object merely names a physical register.
    The renaming pass fills in ``ver`` and ``def_insn`` in place, so a Var must
    never be shared between two use sites.
    """

    __slots__ = ("reg", "ver", "def_insn")
    is_var = True

    def __init__(self, reg, ver=-1):
        self.reg = reg
        self.ver = ver
        self.def_insn = None

    def key(self):
        return (self.reg, self.ver)

    def __str__(self):
        from .regs import reg_name
        if self.ver < 0:
            return reg_name(self.reg)
        return "%s#%d" % (reg_name(self.reg), self.ver)

    __repr__ = __str__


# ---------------------------------------------------------------------------
# instructions
# ---------------------------------------------------------------------------


class Insn(object):
    """
    A single IR operation: ``dst = op.ew(srcs...)``.

    ``aux`` carries op-specific payload (branch target, intrinsic name,
    channel number, scale bias, ...).  ``ea`` is the address of the machine
    instruction this was lifted from, so every IR line stays traceable back to
    the disassembly.
    """

    __slots__ = ("op", "dst", "srcs", "ew", "ea", "aux", "block", "comment",
                 "scalar")

    def __init__(self, op, dst=None, srcs=None, ew=EW.W, ea=None, aux=None,
                 comment=None, scalar=False):
        self.op = op
        self.dst = dst
        self.srcs = list(srcs) if srcs else []
        self.ew = ew
        self.ea = ea
        self.aux = aux
        self.block = None
        self.comment = comment
        # Set by the scalarisation pass when demand analysis proves nothing
        # reads outside the preferred slot, so this can be printed as ordinary
        # scalar arithmetic instead of a vector operation.
        self.scalar = scalar

    # -- def/use helpers ---------------------------------------------------
    def uses(self):
        """Yield the Var operands read by this instruction."""
        for s in self.srcs:
            if s.is_var:
                yield s

    def defines(self):
        return self.dst if isinstance(self.dst, Var) else None

    @property
    def is_terminator(self):
        return self.op.is_terminator

    def __str__(self):
        name = OP_NAME.get(self.op, "op%d" % int(self.op))
        # Element width only carries meaning for lane-wise operations.
        if self.op in _EW_SIGNIFICANT:
            name += "." + EW_SUFFIX[self.ew]
        parts = []
        if self.dst is not None:
            parts.append("%-10s = " % str(self.dst))
        else:
            parts.append(" " * 13)
        if self.op == Op.CONST:
            body = self.srcs[0].scalar_str() if self.scalar else str(self.srcs[0])
        elif self.scalar and self.op in INFIX and len(self.srcs) == 2:
            body = "%s %s %s" % (_s(self.srcs[0]), INFIX[self.op],
                                 _s(self.srcs[1]))
        elif self.scalar and self.op == Op.MOV:
            body = _s(self.srcs[0])
        elif self.op in _ABI_OPS:
            # These carry the whole ABI register set as operands so that DCE
            # cannot delete argument setup or return-value computation.  That
            # is 74 operands; printing them all would bury the instruction.
            explicit = self.srcs[:_ABI_OPS[self.op]]
            body = name
            if self.aux is not None:
                body += " {%s}" % _fmt_aux(self.op, self.aux)
            if explicit:
                body += " " + ", ".join(str(s) for s in explicit)
            body += "  [abi: %d regs]" % (len(self.srcs) - len(explicit))
        elif self.op == Op.PHI:
            args = ", ".join(
                "%s:%s" % (str(s), _blkname(b))
                for s, b in zip(self.srcs, self.aux or [])
            )
            body = "phi(%s)" % args
        else:
            lane = self.ew if self.op in _EW_SIGNIFICANT else None
            # For a load/store the address operand only uses the preferred
            # slot, so print it as an address rather than a 128-bit literal.
            addr_at = 1 if self.op in (Op.LOADQ, Op.STOREQ, Op.LOAD,
                                       Op.STORE, Op.LOADU) else -1
            parts_ = []
            for n, s in enumerate(self.srcs):
                if s.is_const:
                    parts_.append(s.addr_str() if n == addr_at else s.fmt(lane))
                else:
                    parts_.append(str(s))
            args = ", ".join(parts_)
            body = "%s %s" % (name, args) if args else name
            if self.aux is not None and self.op not in (Op.PHI,):
                body += " {%s}" % _fmt_aux(self.op, self.aux)
        line = "".join(parts) + body
        if self.comment:
            line += "    ; " + self.comment
        return line

    __repr__ = __str__


_EW_SIGNIFICANT = frozenset((
    Op.ADD, Op.SUB, Op.MUL, Op.CG, Op.BG, Op.ADDX, Op.SUBX, Op.CGX, Op.BGX,
    Op.SHL, Op.SHR, Op.SAR, Op.ROL, Op.ROTM, Op.ROTMA,
    Op.CMPEQ, Op.CMPGT, Op.CMPGTU, Op.GB, Op.FSM, Op.GENCTL, Op.EXTS,
    Op.FADD, Op.FSUB, Op.FMUL, Op.FMA, Op.FMS, Op.FNMS, Op.FNMA,
    Op.FCMPEQ, Op.FCMPGT, Op.FCMPMEQ, Op.FCMPMGT,
    Op.LOAD, Op.STORE,
))

# Scalar (preferred-slot) rendering: op -> C-like infix operator.
INFIX = {
    Op.ADD: "+", Op.SUB: "-", Op.MUL: "*", Op.MPY: "*", Op.MPYU: "*",
    Op.AND: "&", Op.OR: "|", Op.XOR: "^",
    Op.SHL: "<<", Op.SHR: ">>", Op.SAR: ">>s", Op.ROL: "<<<",
    Op.CMPEQ: "==", Op.CMPGT: ">s", Op.CMPGTU: ">u",
    Op.FADD: "+.", Op.FSUB: "-.", Op.FMUL: "*.",
}


# Ops that carry the ABI register set, mapped to how many leading operands are
# the "real" ones (an indirect target) rather than ABI filler.
ABI_OPS = {Op.CALL: 0, Op.ICALL: 1, Op.IJMP: 1, Op.RET: 0, Op.STOP: 0}
_ABI_OPS = ABI_OPS


def _fmt_aux(op, aux):
    if op in (Op.JMP, Op.CALL):
        return "0x%X" % aux
    if op == Op.CJMP:
        return "0x%X if %s" % (aux[0], aux[1])
    if op in (Op.RDCH, Op.WRCH, Op.RCHCNT):
        from .channels import label
        return label(aux)
    if op == Op.INTRINSIC:
        return str(aux)
    return str(aux)


def _s(v):
    """Render an operand in scalar (preferred-slot) form."""
    return v.scalar_str() if v.is_const else str(v)


def _blkname(b):
    return "B%d" % b.id if b is not None else "?"


# ---------------------------------------------------------------------------
# blocks and functions
# ---------------------------------------------------------------------------


class Block(object):
    __slots__ = ("id", "start_ea", "end_ea", "insns", "preds", "succs",
                 "idom", "domfront", "dom_children")

    def __init__(self, bid, start_ea, end_ea):
        self.id = bid
        self.start_ea = start_ea
        self.end_ea = end_ea
        self.insns = []
        self.preds = []
        self.succs = []
        self.idom = None
        self.domfront = set()
        self.dom_children = []

    def add(self, insn):
        insn.block = self
        self.insns.append(insn)
        return insn

    @property
    def phis(self):
        for i in self.insns:
            if i.op == Op.PHI:
                yield i

    @property
    def terminator(self):
        return self.insns[-1] if self.insns and self.insns[-1].is_terminator \
            else None

    def __str__(self):
        return "B%d" % self.id

    __repr__ = __str__


class Function(object):
    def __init__(self, start_ea, name=""):
        self.start_ea = start_ea
        self.name = name
        self.blocks = []
        self.entry = None
        self.ssa = False

    def new_block(self, start_ea, end_ea):
        b = Block(len(self.blocks), start_ea, end_ea)
        self.blocks.append(b)
        if self.entry is None:
            self.entry = b
        return b

    def insns(self):
        for b in self.blocks:
            for i in b.insns:
                yield i

    def dump(self):
        out = []
        out.append("; ---- %s @ 0x%X %s" %
                   (self.name or "sub_%X" % self.start_ea, self.start_ea,
                    "(SSA)" if self.ssa else "(pre-SSA)"))
        for b in self.blocks:
            preds = ", ".join(str(p) for p in b.preds) or "-"
            out.append("")
            out.append("B%d:  ; 0x%X..0x%X  preds: %s" %
                       (b.id, b.start_ea, b.end_ea, preds))
            for i in b.insns:
                out.append("    " + str(i))
        return "\n".join(out)
