"""
Build the IR control-flow graph for a function.

Basic-block boundaries come from IDA's own flow analysis, which spu.py already
feeds through ``handle_operand``/``add_cref``.  Reusing it means branch-target
discovery, switch tables and noreturn calls stay consistent with what the user
sees in the disassembly, instead of being re-derived (and disagreeing).

The SPU has no delay slots, so a block ends cleanly at its branch.
"""

import ida_funcs
import ida_gdl

from .ir import Op, Insn, Function
from .lifter import Lifter, lift_range
from . import abi


def build(func_ea, drop_hints=True):
    """
    Lift the function containing ``func_ea`` into a pre-SSA IR ``Function``.

    Returns ``(func, lifter)``; the lifter carries the tally of instructions
    that fell through to INTRINSIC, which is the honest measure of how much of
    the function we actually modelled.
    """
    f = ida_funcs.get_func(func_ea)
    if f is None:
        raise ValueError("no function at 0x%X" % func_ea)

    fc = ida_gdl.FlowChart(f, flags=ida_gdl.FC_PREDS)
    ranges = []
    index = {}
    for blk in fc:
        index[blk.id] = len(ranges)
        ranges.append((blk.start_ea, blk.end_ea))

    func = Function(f.start_ea, ida_funcs.get_func_name(f.start_ea) or "")
    lifter = Lifter(drop_hints=drop_hints)
    irblocks = lift_range(lifter, func, ranges)

    # -- wire up the edges -------------------------------------------------
    for blk in fc:
        src = irblocks[index[blk.id]]
        for s in blk.succs():
            dst = irblocks[index[s.id]]
            src.succs.append(dst)
            dst.preds.append(src)

    _add_fallthrough_jumps(func, lifter)
    _prune_unreachable(func)
    func.clobbers = abi.insert_call_clobbers(func)
    return func, lifter


def _add_fallthrough_jumps(func, lifter):
    """
    Give every block an explicit terminator.  A block that just falls into the
    next one gets a JMP, so later passes never have to reason about implicit
    control flow.
    """
    for b in func.blocks:
        if b.insns and b.insns[-1].is_terminator:
            continue
        if len(b.succs) == 1:
            b.add(Insn(Op.JMP, None, [], ea=b.end_ea,
                       aux=b.succs[0].start_ea, comment="fallthrough"))
        elif not b.succs:
            if b.start_ea == b.end_ea:
                # IDA's flow chart emits a zero-length stub for a branch
                # target *outside* the function.  Calling that a return is
                # wrong and actively misleading -- control goes to that
                # address, it does not come back to the caller.
                b.add(Insn(Op.JMP, None, [], ea=b.start_ea,
                           aux=b.start_ea, comment="outside this function"))
            else:
                # A real tail block with no outgoing edge: an unrecognised
                # tail call or the end of a noreturn path.
                b.add(Insn(Op.RET, None, lifter._abi_ret(), ea=b.end_ea,
                           comment="implicit end of block"))


def _prune_unreachable(func):
    """
    Drop blocks unreachable from the entry, recording what was dropped.

    Dominator computation assumes a single reachable entry, so these cannot
    stay.  But they are not noise: in ROM code a helper entered only through
    an indirect branch lands here, and dropping it silently means the listing
    omits real code with no indication.  The ranges go on
    ``func.unreachable`` so callers can say so -- and create functions there.
    """
    seen = set()
    work = [func.entry] if func.entry else []
    while work:
        b = work.pop()
        if b.id in seen:
            continue
        seen.add(b.id)
        work.extend(b.succs)

    # Merge adjacent ranges.  A switch dispatched through a jump table leaves
    # one orphan block per case -- `inflate` in lv2ldr has 339 consecutive
    # four-byte ones -- and reporting those individually turns a single
    # unresolved jump table into a frightening number.
    raw = sorted((b.start_ea, b.end_ea) for b in func.blocks
                 if b.id not in seen and b.end_ea > b.start_ea)
    merged = []
    for lo, hi in raw:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    func.unreachable = merged

    if len(seen) == len(func.blocks):
        return

    keep = [b for b in func.blocks if b.id in seen]
    for b in keep:
        b.preds = [p for p in b.preds if p.id in seen]
        b.succs = [s for s in b.succs if s.id in seen]
    for n, b in enumerate(keep):
        b.id = n
    func.blocks = keep
    func.entry = keep[0] if keep else None
