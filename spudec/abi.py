"""
Calling-convention modelling.

The register sets here follow the SPU ABI and were cross-checked against the
GhidraSPU processor module's `spu.cspec`, whose `<input>`/`<output>` lists are
r3..r74 and whose `<unaffected>` list is exactly r80..r127.

No IDA imports: unit-testable standalone.
"""

from .ir import Op, Insn, Var
from . import regs


def insert_call_clobbers(func):
    """
    Give every call a definition of each volatile register it destroys.

    Without this, a register written before a call and read after it carries
    the same SSA name across the call, so constant propagation happily folds
    the pre-call value into the post-call use.  That is the worst class of
    decompiler bug: the output is confidently wrong and looks fine.

    Only registers this function actually writes need clobbering.  Everything
    else is already live-in at version 0, so a clobber would add no
    information -- and this keeps the cost to a handful of instructions per
    call instead of one per volatile register.  DCE then removes every clobber
    whose value nothing reads, so the ones that survive are exactly the
    genuine read-after-call sites.
    """
    written = {i.dst.reg for i in func.insns()
               if i.dst is not None and regs.is_gpr(i.dst.reg)}
    vol = sorted(written & regs.VOLATILE)
    if not vol:
        return 0

    n = 0
    for b in func.blocks:
        out = []
        for insn in b.insns:
            out.append(insn)
            if insn.op in (Op.CALL, Op.ICALL):
                for r in vol:
                    c = Insn(Op.UNDEF, Var(r), [], ea=insn.ea,
                             comment="clobbered by call")
                    c.block = b
                    out.append(c)
                    n += 1
        b.insns = out
    return n
