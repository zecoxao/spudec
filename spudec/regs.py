"""
SPU register-file model.

The IR register space is the 128 architectural GPRs plus a small number of
pseudo-registers used to thread ordering-sensitive state (memory, channel
traffic, SPR traffic) through SSA.  Modelling memory as just another register
means the SSA machinery, copy propagation and DCE all work on it for free, and
a load can name exactly which store it is ordered after.
"""

NGPR = 128

R_MEM = 128      # local-store contents
R_CH = 129       # channel ordering chain (rdch/wrch/rchcnt)
R_SPR = 130      # special-purpose register file
NREG = 131

# Architectural names, matching what spu.py prints so IR lines line up with
# the disassembly.
REG_NAMES = ["lr", "sp"] + ["r%d" % i for i in range(2, NGPR)]
REG_NAMES += ["mem", "ch", "spr"]

LR = 0           # $0  link register
SP = 1           # $1  stack pointer (16-byte aligned, back chain at 0(sp))
ENV = 2          # $2  environment pointer / scratch

# SPU ABI (SPU Application Binary Interface Specification 1.9)
ARG_FIRST, ARG_LAST = 3, 74      # arguments and return values
VOLATILE_LAST = 79               # r75..r79 additionally volatile
CALLEE_SAVED_FIRST = 80          # r80..r127 preserved across calls

PSEUDO = frozenset((R_MEM, R_CH, R_SPR))

# Registers a call destroys.  Per the SPU ABI (and confirmed against the
# GhidraSPU cspec, whose <unaffected> set is exactly r80..r127): r1 is the
# stack pointer and r80..r127 are callee-saved; everything else is volatile.
# r0 is included because `brsl`/`bisl` overwrite the link register.
VOLATILE = frozenset([LR] + list(range(ENV, CALLEE_SAVED_FIRST)))


def reg_name(r):
    if 0 <= r < len(REG_NAMES):
        return REG_NAMES[r]
    if r >= NREG:
        return "t%d" % (r - NREG)      # lifter-allocated temporary
    return "?reg%d" % r


def is_temp(r):
    return r >= NREG


def is_gpr(r):
    return 0 <= r < NGPR


def is_callee_saved(r):
    return CALLEE_SAVED_FIRST <= r < NGPR


def is_volatile(r):
    return r == LR or r == ENV or (ARG_FIRST <= r <= VOLATILE_LAST)


def arg_regs(n):
    """The first ``n`` argument registers, in order."""
    return list(range(ARG_FIRST, min(ARG_FIRST + n, ARG_LAST + 1)))
