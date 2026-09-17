"""
Lift IDA's SPU decode (procs/spu.py) into the SSA-ready IR.

The processor module is used purely as a decoder front end: it gives us
``insn_t`` with itype + operands, and we supply the semantics from the SPU ISA
reference.

Two things about spu.py drive the design here:

1. ``itype_*`` constants are assigned by iterating dict values at module init,
   so they are *not* stable across IDA builds or even table edits.  We resolve
   mnemonics through ``ida_idp.ph_get_instruc()`` at runtime and dispatch on
   the name.

2. Operand order is form-dependent.  For RR forms spu.py assigns
   ``Op1 = RT, Op2 = RA, Op3 = RB``; for RRR forms ``Op1 = RT, Op2 = RA,
   Op3 = RB, Op4 = RC``.  Load/store d-forms put the displacement in ``Op2``
   as ``o_displ`` with the offset *already* scaled (``<<4`` for lqd/stqd, not
   scaled for cbd/chd/cwd/cdd).

Every instruction gets lifted to something.  Anything without a hand-written
semantic becomes an ``INTRINSIC`` with correct defs and uses, so the IR is
always sound even where it is imprecise -- silently dropping an instruction
would corrupt the SSA.
"""

import ida_idp
import ida_ua
import ida_bytes
from ida_ua import o_void, o_reg, o_imm, o_displ, o_mem, o_near

from .ir import Op, EW, Const, Var, Insn, Function, MASK128
from . import regs

FL_SIGNED = 0x01     # spu_processor_t.FL_SIGNED

U64 = (1 << 64) - 1

# SPU local store is 256 KB and accesses wrap within it (the hardware masks
# with LSLR), then the low four bits are dropped for the quadword access.  So
# the effective-address mask is 0x3FFF0, not just ~0xF.  GhidraSPU applies the
# same 0x3FFF0 for absolute and PC-relative forms.
LS_MASK = 0x3FFF0


# ---------------------------------------------------------------------------
# constant helpers (all values are 128-bit, byte 0 = most significant)
# ---------------------------------------------------------------------------


def repl(val, ew):
    """Replicate ``val`` across every ``ew``-wide lane of a quadword."""
    bits = int(ew) * 8
    val &= (1 << bits) - 1
    out = 0
    for _ in range(16 // int(ew)):
        out = (out << bits) | val
    return out


def word0(val):
    """A quadword whose preferred slot is ``val`` and whose other words are 0."""
    return (val & 0xFFFFFFFF) << 96


def sext(val, bits):
    m = 1 << (bits - 1)
    val &= (1 << bits) - 1
    return (val ^ m) - m


def _signed(raw, specval):
    """
    Recover a signed immediate.  spu.py subtracts from the raw field to make it
    negative, but the value lands in an unsigned 64-bit SWIG field, so we sign
    the 64-bit pattern back when FL_SIGNED is set.
    """
    raw &= U64
    if (specval & FL_SIGNED) and (raw >> 63):
        raw -= 1 << 64
    return raw


def fsmbi_const(imm16):
    """Form-select-mask-for-bytes: bit i (from the MSB) sets byte i to 0xFF."""
    out = 0
    for j in range(16):
        out <<= 8
        if (imm16 >> (15 - j)) & 1:
            out |= 0xFF
    return out


# ---------------------------------------------------------------------------


class LiftError(Exception):
    pass


class Lifter(object):

    def __init__(self, drop_hints=True, n_arg_regs=8, n_ret_regs=8):
        # itype -> canonical mnemonic, built from the live processor module.
        self.itype_name = [t[0] for t in ida_idp.ph_get_instruc()]
        self.drop_hints = drop_hints
        # Registers a call may read and a return may hand back.  Without
        # these, DCE deletes argument setup before a call and return-value
        # computation before a ret, because nothing in *this* function reads
        # them.
        #
        # The ABI range is r3..r74, but assuming a call reads all 72 makes
        # every scratch register in the function immortal -- including the
        # exact intermediates scalarisation has just made redundant.  Eight
        # registers is 128 bytes of arguments, past which the ABI passes by
        # hidden pointer; widen with n_arg_regs if a callee really takes large
        # structs by value.
        self.abi_args = list(range(regs.ARG_FIRST,
                                   min(regs.ARG_FIRST + n_arg_regs,
                                       regs.ARG_LAST + 1)))
        self.ret_regs = list(range(regs.ARG_FIRST,
                                   min(regs.ARG_FIRST + n_ret_regs,
                                       regs.ARG_LAST + 1)))
        self.func = None
        self.blk = None
        self.ea = 0
        self._ntmp = 0
        self.unhandled = {}
        self._build_tables()

    # -- emission helpers --------------------------------------------------

    def tmp(self):
        r = regs.NREG + self._ntmp
        self._ntmp += 1
        return r

    def emit(self, op, dst=None, srcs=(), ew=EW.W, aux=None, comment=None):
        i = Insn(op, Var(dst) if dst is not None else None,
                 list(srcs), ew, self.ea, aux, comment)
        self.blk.add(i)
        return i

    def const(self, val, comment=None):
        """Materialise a constant into a fresh temp and return a Var for it."""
        t = self.tmp()
        self.emit(Op.CONST, t, [Const(val)], comment=comment)
        return Var(t)

    # -- operand helpers ---------------------------------------------------

    @staticmethod
    def r(op):
        """A fresh Var for a register operand (never share Var objects)."""
        return Var(op.reg)

    @staticmethod
    def imm(op):
        return _signed(op.value, op.specval)

    @staticmethod
    def disp(op):
        return _signed(op.addr, op.specval)

    # ======================================================================
    # dispatch table
    # ======================================================================

    def _build_tables(self):
        # -- simple RR: dst = op(ra, rb) -----------------------------------
        self.rr = {
            "a":     (Op.ADD, EW.W),   "ah":    (Op.ADD, EW.H),
            "and":   (Op.AND, EW.W),   "or":    (Op.OR, EW.W),
            "xor":   (Op.XOR, EW.W),   "nand":  (Op.NAND, EW.W),
            "nor":   (Op.NOR, EW.W),   "eqv":   (Op.EQV, EW.W),
            "andc":  (Op.ANDC, EW.W),  "orc":   (Op.ORC, EW.W),
            "cg":    (Op.CG, EW.W),    "bg":    (Op.BG, EW.W),
            "absdb": (Op.ABSDB, EW.B), "avgb":  (Op.AVGB, EW.B),
            "sumb":  (Op.SUMB, EW.B),
            "ceq":   (Op.CMPEQ, EW.W), "ceqh":  (Op.CMPEQ, EW.H),
            "ceqb":  (Op.CMPEQ, EW.B),
            "cgt":   (Op.CMPGT, EW.W), "cgth":  (Op.CMPGT, EW.H),
            "cgtb":  (Op.CMPGT, EW.B),
            "clgt":  (Op.CMPGTU, EW.W), "clgth": (Op.CMPGTU, EW.H),
            "clgtb": (Op.CMPGTU, EW.B),
            "rot":   (Op.ROL, EW.W),   "roth":  (Op.ROL, EW.H),
            "rotm":  (Op.ROTM, EW.W),  "rothm": (Op.ROTM, EW.H),
            "rotma": (Op.ROTMA, EW.W), "rotmah": (Op.ROTMA, EW.H),
            "shl":   (Op.SHL, EW.W),   "shlh":  (Op.SHL, EW.H),
            "mpy":   (Op.MPY, EW.W),   "mpyu":  (Op.MPYU, EW.W),
            "mpyh":  (Op.MPYH, EW.W),  "mpys":  (Op.MPYS, EW.W),
            "mpyhh": (Op.MPYHH, EW.W), "mpyhhu": (Op.MPYHHU, EW.W),
            "fa":    (Op.FADD, EW.W),  "fs":    (Op.FSUB, EW.W),
            "fm":    (Op.FMUL, EW.W),
            "dfa":   (Op.FADD, EW.D),  "dfs":   (Op.FSUB, EW.D),
            "dfm":   (Op.FMUL, EW.D),
            "fceq":  (Op.FCMPEQ, EW.W), "fcgt":  (Op.FCMPGT, EW.W),
            "fcmeq": (Op.FCMPMEQ, EW.W), "fcmgt": (Op.FCMPMGT, EW.W),
            "dfceq": (Op.FCMPEQ, EW.D), "dfcgt": (Op.FCMPGT, EW.D),
            "dfcmeq": (Op.FCMPMEQ, EW.D), "dfcmgt": (Op.FCMPMGT, EW.D),
            "fi":    (Op.FI, EW.W),
            # quadword shifts with a register count
            "rotqbi":  (Op.QROTBI, EW.Q),  "rotqmbi": (Op.QROTMBI, EW.Q),
            "shlqbi":  (Op.QSHLBI, EW.Q),
            "rotqby":  (Op.QROTBY, EW.Q),  "rotqmby": (Op.QROTMBY, EW.Q),
            "shlqby":  (Op.QSHLBY, EW.Q),
        }

        # RR forms where RT is also a source.  spu.py marks *every* RR form as
        # CF_USE1|CF_CHG1 regardless, so the feature bits cannot be used to
        # tell these apart -- the list has to be explicit.
        self.rr_acc = {
            "addx":    (Op.ADDX, EW.W),     # rt = ra + rb + (rt & 1)
            "sfx":     (Op.SUBX, EW.W),     # rt = rb + ~ra + (rt & 1)
            "cgx":     (Op.CGX, EW.W),
            "bgx":     (Op.BGX, EW.W),
            "mpyhha":  (Op.MPYHHA, EW.W),
            "mpyhhau": (Op.MPYHHAU, EW.W),
            "dfma":    (Op.FMA, EW.D),
            "dfms":    (Op.FMS, EW.D),
            "dfnms":   (Op.FNMS, EW.D),
            "dfnma":   (Op.FNMA, EW.D),
        }

        # "...bybi" forms take the byte count from bits 3:6 of RB's word 0.
        self.rr_bybi = {
            "rotqbybi":  Op.QROTBY,
            "rotqmbybi": Op.QROTMBY,
            "shlqbybi":  Op.QSHLBY,
        }

        # -- unary RR: dst = op(ra) ----------------------------------------
        self.unary = {
            "clz":     (Op.CLZ, EW.W),   "cntb":  (Op.CNTB, EW.B),
            "orx":     (Op.ORX, EW.W),
            "xsbh":    (Op.EXTS, EW.H),  "xshw":  (Op.EXTS, EW.W),
            "xswd":    (Op.EXTS, EW.D),
            "gb":      (Op.GB, EW.W),    "gbh":   (Op.GB, EW.H),
            "gbb":     (Op.GB, EW.B),
            "fsm":     (Op.FSM, EW.W),   "fsmh":  (Op.FSM, EW.H),
            "fsmb":    (Op.FSM, EW.B),
            "frest":   (Op.FREST, EW.W), "frsqest": (Op.FRSQEST, EW.W),
            "fesd":    (Op.FESD, EW.D),  "frds":  (Op.FRDS, EW.W),
        }

        # -- RI10: dst = op(ra, repl(imm, ew)) -----------------------------
        self.ri10 = {
            "ori":   (Op.OR, EW.W),   "orhi":  (Op.OR, EW.H),
            "orbi":  (Op.OR, EW.B),
            "andi":  (Op.AND, EW.W),  "andhi": (Op.AND, EW.H),
            "andbi": (Op.AND, EW.B),
            "xori":  (Op.XOR, EW.W),  "xorhi": (Op.XOR, EW.H),
            "xorbi": (Op.XOR, EW.B),
            "ai":    (Op.ADD, EW.W),  "ahi":   (Op.ADD, EW.H),
            "ceqi":  (Op.CMPEQ, EW.W), "ceqhi": (Op.CMPEQ, EW.H),
            "ceqbi": (Op.CMPEQ, EW.B),
            "cgti":  (Op.CMPGT, EW.W), "cgthi": (Op.CMPGT, EW.H),
            "cgtbi": (Op.CMPGT, EW.B),
            "clgti": (Op.CMPGTU, EW.W), "clgthi": (Op.CMPGTU, EW.H),
            "clgtbi": (Op.CMPGTU, EW.B),
            "mpyi":  (Op.MPY, EW.W),  "mpyui": (Op.MPYU, EW.W),
        }

        # -- RI7 with an immediate shift/rotate count ----------------------
        # (name, ew, kind) where kind selects the count normalisation
        self.ri7_shift = {
            "roti":    (EW.W, "rol"),   "rothi":   (EW.H, "rol"),
            "rotmi":   (EW.W, "shr"),   "rothmi":  (EW.H, "shr"),
            "rotmai":  (EW.W, "sar"),   "rotmahi": (EW.H, "sar"),
            "shli":    (EW.W, "shl"),   "shlhi":   (EW.H, "shl"),
        }
        self.ri7_quad = {
            "rotqbyi":  "qrotby", "rotqmbyi": "qrotmby",
            "shlqbyi":  "qshlby",
            "rotqbii":  "qrotbi", "rotqmbii": "qrotmbi",
            "shlqbii":  "qshlbi",
        }

        # -- shuffle-control generators ------------------------------------
        self.genctl = {
            "cbd": EW.B, "cbx": EW.B, "chd": EW.H, "chx": EW.H,
            "cwd": EW.W, "cwx": EW.W, "cdd": EW.D, "cdx": EW.D,
        }

        # -- conditional branches ------------------------------------------
        # mnemonic -> condition kind tested on RT
        self.cond = {
            "brz": "z", "brnz": "nz", "brhz": "hz", "brhnz": "hnz",
            "biz": "z", "binz": "nz", "bihz": "hz", "bihnz": "hnz",
        }

        self.halts = {
            "heq": "eq", "hgt": "gt", "hlgt": "lgt",
            "heqi": "eq", "hgti": "gt", "hlgti": "lgt",
        }

        self.converts = {
            "cflts": Op.CFLTS, "cfltu": Op.CFLTU,
            "csflt": Op.CSFLT, "cuflt": Op.CUFLT,
        }

        self.hints = frozenset(("hbr", "hbra", "hbrr"))
        self.nops = frozenset(("nop", "lnop"))

    # ======================================================================
    # per-instruction lifting
    # ======================================================================

    def lift(self, insn, blk):
        """Lift one decoded ``insn_t`` into ``blk``."""
        self.blk = blk
        self.ea = insn.ea
        name = self.itype_name[insn.itype] \
            if insn.itype < len(self.itype_name) else "?"

        h = getattr(self, "_i_" + name.replace(".", "_"), None)
        if h is not None:
            h(insn, name)
            return

        if name in self.rr:
            op, ew = self.rr[name]
            self.emit(op, insn.Op1.reg,
                      [self.r(insn.Op2), self.r(insn.Op3)], ew)
        elif name in self.rr_acc:
            op, ew = self.rr_acc[name]
            self.emit(op, insn.Op1.reg,
                      [self.r(insn.Op2), self.r(insn.Op3), Var(insn.Op1.reg)],
                      ew)
        elif name in self.rr_bybi:
            self._bybi(insn, self.rr_bybi[name])
        elif name in self.unary:
            op, ew = self.unary[name]
            self.emit(op, insn.Op1.reg, [self.r(insn.Op2)], ew)
        elif name in self.ri10:
            op, ew = self.ri10[name]
            c = self.const(repl(self.imm(insn.Op3), ew))
            self.emit(op, insn.Op1.reg, [self.r(insn.Op2), c], ew)
        elif name in self.ri7_shift:
            self._shift_imm(insn, *self.ri7_shift[name])
        elif name in self.ri7_quad:
            self._quad_imm(insn, self.ri7_quad[name])
        elif name in self.genctl:
            self._genctl(insn, self.genctl[name])
        elif name in self.cond:
            self._branch_cond(insn, name)
        elif name in self.halts:
            self._halt(insn, name)
        elif name in self.converts:
            self.emit(self.converts[name], insn.Op1.reg, [self.r(insn.Op2)],
                      EW.W, aux="scale=%d" % self.imm(insn.Op3))
        elif name in self.hints:
            if not self.drop_hints:
                self.emit(Op.NOP, None, [], aux=name, comment="branch hint")
        elif name in self.nops:
            pass
        else:
            self._intrinsic(insn, name)

    # -- fallback ----------------------------------------------------------

    def _intrinsic(self, insn, name):
        """
        Opaque but sound: read every operand IDA marks as used, write the one
        it marks as changed.  Keeps SSA correct for anything we do not model.
        """
        self.unhandled[name] = self.unhandled.get(name, 0) + 1
        feat = insn.get_canon_feature()
        srcs, dst = [], None
        for n in range(4):
            op = insn[n]
            if op.type == o_void:
                continue
            if op.type == o_reg:
                if feat & (ida_idp.CF_USE1 << n):
                    srcs.append(Var(op.reg))
                if feat & (ida_idp.CF_CHG1 << n) and dst is None:
                    dst = op.reg
            elif op.type == o_imm:
                srcs.append(Const(repl(self.imm(op), EW.W)))
        self.emit(Op.INTRINSIC, dst, srcs, EW.Q, aux=name)

    # -- helpers used by several handlers ----------------------------------

    def _bybi(self, insn, op):
        """rotqbybi & friends: shift count is bits 3:6 of RB's preferred slot."""
        t = self.tmp()
        c = self.const(repl(3, EW.W))
        self.emit(Op.SHR, t, [self.r(insn.Op3), c], EW.W,
                  comment="byte count from bits 3:6")
        self.emit(op, insn.Op1.reg, [self.r(insn.Op2), Var(t)], EW.Q)

    def _shift_imm(self, insn, ew, kind):
        """
        Element shifts with an immediate count.

        SPU encodes right shifts as a *negated* count: `rotmi rt,ra,-3` shifts
        right by 3.  With a literal count we can prove that away and emit a
        plain SHR/SAR, which is both correct and far more readable.  Counts
        that exceed the element width produce 0 (logical) or a sign splat
        (arithmetic), exactly as the hardware does.
        """
        bits = ew.bits
        raw = self.imm(insn.Op3)
        mask = 0x3F if ew == EW.W else 0x1F

        if kind == "shl":
            n = raw & mask
            if n >= bits:
                self.emit(Op.CONST, insn.Op1.reg, [Const(0)],
                          comment="shl count %d >= %d" % (n, bits))
                return
            self.emit(Op.SHL, insn.Op1.reg,
                      [self.r(insn.Op2), self.const(repl(n, ew))], ew)
        elif kind == "rol":
            n = raw & (bits - 1)
            if n == 0:
                self.emit(Op.MOV, insn.Op1.reg, [self.r(insn.Op2)], ew)
                return
            self.emit(Op.ROL, insn.Op1.reg,
                      [self.r(insn.Op2), self.const(repl(n, ew))], ew)
        else:                                   # shr / sar
            n = (-raw) & mask
            if n >= bits:
                if kind == "shr":
                    self.emit(Op.CONST, insn.Op1.reg, [Const(0)],
                              comment="shr count %d >= %d" % (n, bits))
                    return
                n = bits - 1                    # arithmetic: splat the sign
            op = Op.SHR if kind == "shr" else Op.SAR
            self.emit(op, insn.Op1.reg,
                      [self.r(insn.Op2), self.const(repl(n, ew))], ew)

    def _quad_imm(self, insn, kind):
        """Quadword shifts/rotates with an immediate count."""
        raw = self.imm(insn.Op3)
        if kind in ("qrotby", "qrotbi"):
            n = raw & (0xF if kind == "qrotby" else 0x7)
            if n == 0:
                self.emit(Op.MOV, insn.Op1.reg, [self.r(insn.Op2)], EW.Q)
                return
            op = Op.QROTBY if kind == "qrotby" else Op.QROTBI
        elif kind in ("qrotmby", "qrotmbi"):
            n = (-raw) & (0x1F if kind == "qrotmby" else 0x7)
            if kind == "qrotmby" and n >= 16:
                self.emit(Op.CONST, insn.Op1.reg, [Const(0)],
                          comment="shift right %d bytes" % n)
                return
            op = Op.QROTMBY if kind == "qrotmby" else Op.QROTMBI
        else:                                   # qshlby / qshlbi
            n = raw & (0x1F if kind == "qshlby" else 0x7)
            if kind == "qshlby" and n >= 16:
                self.emit(Op.CONST, insn.Op1.reg, [Const(0)],
                          comment="shift left %d bytes" % n)
                return
            if n == 0:
                self.emit(Op.MOV, insn.Op1.reg, [self.r(insn.Op2)], EW.Q)
                return
            op = Op.QSHLBY if kind == "qshlby" else Op.QSHLBI
        self.emit(op, insn.Op1.reg,
                  [self.r(insn.Op2), self.const(word0(n))], EW.Q)

    def _genctl(self, insn, ew):
        """
        cbd/cwd/... generate the shufb control mask used to insert a scalar
        into a quadword.  Folding these together with the shufb is what turns
        the cwd/lqd/shufb/stqd idiom back into `*(int *)p = x`, so the address
        expression is kept explicit rather than hidden in an intrinsic.
        """
        if insn.Op2.type == o_displ:            # d-form: cwd rt, off(ra)
            off = self.disp(insn.Op2)
            base = Var(insn.Op2.reg)
            if off:
                t = self.tmp()
                self.emit(Op.ADD, t, [base, self.const(repl(off, EW.W))], EW.W)
                addr = Var(t)
            else:
                addr = base
        else:                                   # x-form: cwx rt, ra, rb
            t = self.tmp()
            self.emit(Op.ADD, t, [self.r(insn.Op2), self.r(insn.Op3)], EW.W)
            addr = Var(t)
        self.emit(Op.GENCTL, insn.Op1.reg, [addr], ew)

    # -- addressing --------------------------------------------------------

    def _ea_displ(self, op):
        """Effective address for a d-form load/store: (RA + off) & ~0xF."""
        off = self.disp(op)
        base = Var(op.reg)
        if off:
            t = self.tmp()
            self.emit(Op.ADD, t, [base, self.const(repl(off, EW.W))], EW.W)
            base = Var(t)
        t = self.tmp()
        self.emit(Op.AND, t, [base, self.const(repl(LS_MASK, EW.W))], EW.W,
                  comment="local store, quadword aligned")
        return Var(t)

    def _ea_index(self, ra, rb):
        """Effective address for an x-form load/store: (RA + RB) & LS_MASK."""
        t = self.tmp()
        self.emit(Op.ADD, t, [self.r(ra), self.r(rb)], EW.W)
        t2 = self.tmp()
        self.emit(Op.AND, t2, [Var(t), self.const(repl(LS_MASK, EW.W))],
                  EW.W, comment="local store, quadword aligned")
        return Var(t2)

    def _load(self, dst, addr):
        self.emit(Op.LOADQ, dst, [Var(regs.R_MEM), addr], EW.Q)

    def _store(self, val, addr):
        self.emit(Op.STOREQ, regs.R_MEM, [Var(regs.R_MEM), addr, val], EW.Q)

    # ======================================================================
    # named handlers (anything not expressible as a plain table entry)
    # ======================================================================

    # -- moves and immediates ---------------------------------------------

    def _i_lr(self, insn, name):
        self.emit(Op.MOV, insn.Op1.reg, [self.r(insn.Op2)], EW.Q)

    def _i_il(self, insn, name):
        self.emit(Op.CONST, insn.Op1.reg,
                  [Const(repl(sext(self.imm(insn.Op2), 16), EW.W))])

    def _i_ilh(self, insn, name):
        self.emit(Op.CONST, insn.Op1.reg,
                  [Const(repl(self.imm(insn.Op2) & 0xFFFF, EW.H))])

    def _i_ilhu(self, insn, name):
        self.emit(Op.CONST, insn.Op1.reg,
                  [Const(repl((self.imm(insn.Op2) & 0xFFFF) << 16, EW.W))])

    def _i_ila(self, insn, name):
        self.emit(Op.CONST, insn.Op1.reg,
                  [Const(repl(self.imm(insn.Op2) & 0x3FFFF, EW.W))])

    def _i_iohl(self, insn, name):
        # iohl reads RT as well as writing it.
        c = self.const(repl(self.imm(insn.Op2) & 0xFFFF, EW.W))
        self.emit(Op.OR, insn.Op1.reg, [Var(insn.Op1.reg), c], EW.W)

    def _i_fsmbi(self, insn, name):
        self.emit(Op.CONST, insn.Op1.reg,
                  [Const(fsmbi_const(self.imm(insn.Op2) & 0xFFFF))],
                  comment="fsmbi")

    # -- subtract (SPU reverses the operands) ------------------------------

    def _i_sf(self, insn, name):
        # sf rt, ra, rb  =>  rt = rb - ra
        self.emit(Op.SUB, insn.Op1.reg,
                  [self.r(insn.Op3), self.r(insn.Op2)], EW.W)

    def _i_sfh(self, insn, name):
        self.emit(Op.SUB, insn.Op1.reg,
                  [self.r(insn.Op3), self.r(insn.Op2)], EW.H)

    def _i_sfi(self, insn, name):
        c = self.const(repl(self.imm(insn.Op3), EW.W))
        self.emit(Op.SUB, insn.Op1.reg, [c, self.r(insn.Op2)], EW.W)

    def _i_sfhi(self, insn, name):
        c = self.const(repl(self.imm(insn.Op3), EW.H))
        self.emit(Op.SUB, insn.Op1.reg, [c, self.r(insn.Op2)], EW.H)

    # -- ternary RRR -------------------------------------------------------

    def _i_selb(self, insn, name):
        self.emit(Op.SELB, insn.Op1.reg,
                  [self.r(insn.Op2), self.r(insn.Op3), self.r(insn.Op4)],
                  EW.Q)

    def _i_shufb(self, insn, name):
        self.emit(Op.SHUFB, insn.Op1.reg,
                  [self.r(insn.Op2), self.r(insn.Op3), self.r(insn.Op4)],
                  EW.Q)

    def _i_mpya(self, insn, name):
        self.emit(Op.MPYA, insn.Op1.reg,
                  [self.r(insn.Op2), self.r(insn.Op3), self.r(insn.Op4)],
                  EW.W)

    def _i_fma(self, insn, name):
        self.emit(Op.FMA, insn.Op1.reg,
                  [self.r(insn.Op2), self.r(insn.Op3), self.r(insn.Op4)],
                  EW.W)

    def _i_fms(self, insn, name):
        self.emit(Op.FMS, insn.Op1.reg,
                  [self.r(insn.Op2), self.r(insn.Op3), self.r(insn.Op4)],
                  EW.W)

    def _i_fnms(self, insn, name):
        self.emit(Op.FNMS, insn.Op1.reg,
                  [self.r(insn.Op2), self.r(insn.Op3), self.r(insn.Op4)],
                  EW.W)

    # -- loads and stores --------------------------------------------------

    def _i_lqd(self, insn, name):
        self._load(insn.Op1.reg, self._ea_displ(insn.Op2))

    def _i_stqd(self, insn, name):
        self._store(self.r(insn.Op1), self._ea_displ(insn.Op2))

    def _i_lqx(self, insn, name):
        self._load(insn.Op1.reg, self._ea_index(insn.Op2, insn.Op3))

    def _i_stqx(self, insn, name):
        self._store(self.r(insn.Op1), self._ea_index(insn.Op2, insn.Op3))

    def _i_lqa(self, insn, name):
        self._load(insn.Op1.reg,
                   self.const(word0(insn.Op2.addr & LS_MASK)))

    _i_lqr = _i_lqa

    def _i_stqa(self, insn, name):
        self._store(self.r(insn.Op1),
                    self.const(word0(insn.Op2.addr & LS_MASK)))

    _i_stqr = _i_stqa

    # -- branches ----------------------------------------------------------

    def _abi(self):
        """
        What a *call* reads: every argument register, because the callee's
        arity is unknown, plus memory, the channel chain, and LR (the callee
        reads it to return).  Being conservative here is what stops DCE from
        deleting argument setup.
        """
        return ([Var(regs.R_MEM), Var(regs.R_CH), Var(regs.LR)] +
                [Var(r) for r in self.abi_args])

    def _abi_ret(self):
        """
        What a *return* hands back.  The same conservatism would be wrong
        here: claiming all of r3..r74 is live-out keeps every scratch value
        computed anywhere in the function alive to the exit, which buries the
        listing in dead intermediates.  A return value lives in r3 onward --
        eight registers is 128 bytes, past which the ABI returns via a hidden
        pointer in memory.
        """
        return ([Var(regs.R_MEM), Var(regs.R_CH)] +
                [Var(r) for r in self.ret_regs])

    def _link(self, dst):
        """brsl/bisl write the return address into RT's preferred slot."""
        self.emit(Op.CONST, dst, [Const(word0(self.ea + 4))],
                  comment="link register")

    def _i_br(self, insn, name):
        self.emit(Op.JMP, None, [], aux=insn.Op1.addr)

    _i_bra = _i_br

    def _i_brsl(self, insn, name):
        self._link(insn.Op1.reg)
        # A call reads the argument registers and clobbers memory.
        self.emit(Op.CALL, regs.R_MEM, self._abi(), aux=insn.Op2.addr)

    _i_brasl = _i_brsl

    def _i_bi(self, insn, name):
        if insn.Op1.reg == regs.LR:
            self.emit(Op.RET, None, self._abi_ret())
        else:
            # An indirect branch out of the function is a tail call, so it
            # passes arguments -- the full set, not the return set.
            self.emit(Op.IJMP, None, [self.r(insn.Op1)] + self._abi())

    def _i_iret(self, insn, name):
        self.emit(Op.RET, None, self._abi_ret(), comment="interrupt return")

    def _i_bisl(self, insn, name):
        self._link(insn.Op1.reg)
        self.emit(Op.ICALL, regs.R_MEM,
                  [self.r(insn.Op2)] + self._abi())

    def _branch_cond(self, insn, name):
        kind = self.cond[name]
        if name.startswith("br"):               # relative, RT tested
            self.emit(Op.CJMP, None, [self.r(insn.Op1)],
                      aux=(insn.Op2.addr, kind))
        else:
            # biz/binz/bihz/bihnz: branch to a *register* if the test passes.
            # Emitting an unconditional IJMP here would turn a conditional
            # branch into an unconditional one and lose the fall-through path
            # entirely.  CIJMP is not a terminator, so control correctly
            # continues into the next instruction when the test fails.
            self.emit(Op.CIJMP, None,
                      [self.r(insn.Op1), self.r(insn.Op2)], aux=kind)

    def _halt(self, insn, name):
        cond = self.halts[name]
        if name.endswith("i"):
            srcs = [self.r(insn.Op2),
                    self.const(repl(self.imm(insn.Op3), EW.W))]
        else:
            srcs = [self.r(insn.Op2), self.r(insn.Op3)]
        self.emit(Op.HALT, None, srcs, EW.W, aux=cond)

    # -- system ------------------------------------------------------------

    def _i_stop(self, insn, name):
        code = insn.Op1.value if insn.Op1.type == o_imm else 0
        self.emit(Op.STOP, None, self._abi_ret(), aux="0x%X" % code)

    def _i_stopd(self, insn, name):
        self.emit(Op.STOP, None, self._abi_ret(), aux="stopd")

    def _i_sync(self, insn, name):
        self.emit(Op.SYNC, regs.R_MEM, [Var(regs.R_MEM)], aux="sync")

    def _i_dsync(self, insn, name):
        self.emit(Op.SYNC, regs.R_MEM, [Var(regs.R_MEM)], aux="dsync")

    def _i_rdch(self, insn, name):
        ch = insn.Op2.reg - 256
        self.emit(Op.RDCH, insn.Op1.reg, [Var(regs.R_CH)], EW.Q, aux=ch)
        # Bump the channel chain so two reads of the same channel stay ordered.
        self.emit(Op.SYNC, regs.R_CH, [Var(regs.R_CH)], aux="after rdch %d" % ch)

    def _i_rchcnt(self, insn, name):
        ch = insn.Op2.reg - 256
        self.emit(Op.RCHCNT, insn.Op1.reg, [Var(regs.R_CH)], EW.Q, aux=ch)

    def _i_wrch(self, insn, name):
        # spu.py swaps the operands for wrch: Op1 = channel, Op2 = source.
        ch = insn.Op1.reg - 256
        self.emit(Op.WRCH, regs.R_CH,
                  [Var(regs.R_CH), self.r(insn.Op2)], EW.Q, aux=ch)

    def _i_mfspr(self, insn, name):
        self.emit(Op.MFSPR, insn.Op1.reg, [Var(regs.R_SPR)], EW.Q,
                  aux=insn.Op2.reg - 128)

    def _i_mtspr(self, insn, name):
        self.emit(Op.MTSPR, regs.R_SPR,
                  [Var(regs.R_SPR), self.r(insn.Op2)], EW.Q,
                  aux=insn.Op1.reg - 128)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def lift_range(lifter, func, blocks):
    """
    ``blocks`` is a list of (start_ea, end_ea) pairs already split into basic
    blocks.  Returns the list of IR blocks in the same order.
    """
    lifter.func = func
    out = []
    for start, end in blocks:
        b = func.new_block(start, end)
        out.append(b)
        ea = start
        insn = ida_ua.insn_t()
        while ea < end:
            n = ida_ua.decode_insn(insn, ea)
            if n <= 0:
                lifter.blk, lifter.ea = b, ea
                lifter.emit(Op.INTRINSIC, None, [], aux="undecodable")
                ea += 4
                continue
            lifter.lift(insn, b)
            ea += n
    return out








