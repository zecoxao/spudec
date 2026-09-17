"""
SSA-level cleanup: phi simplification, copy/constant propagation, constant
folding and dead-code elimination.

These run to a fixpoint.  On real SPU code the payoff is large because the
lifter is deliberately verbose -- every effective address is spelled out as
add/and against a materialised constant, and every immediate becomes its own
CONST.  Folding collapses all of that back down, and folding the ``cwd``/
``shufb`` control masks (see :mod:`sem`) is what later makes scalar memory
accesses recoverable.

Every pass preserves SSA form: instructions are rewritten in place and no
definition is ever duplicated.

No IDA imports: unit-testable standalone.
"""

from .ir import Op, EW, Const, Var, Insn, ABI_OPS, MASK128
from .sem import evaluate


def _defmap(func):
    m = {}
    for i in func.insns():
        d = i.defines()
        if d is not None:
            m[d.key()] = i
    return m


def _usecount(func):
    n = {}
    for i in func.insns():
        for u in i.uses():
            k = u.key()
            n[k] = n.get(k, 0) + 1
    return n


def _clone(v):
    """A fresh Var with the same binding (Vars must not be shared)."""
    c = Var(v.reg, v.ver)
    c.def_insn = v.def_insn
    return c


# ---------------------------------------------------------------------------


def simplify_phis(func):
    """
    phi(x, x, ...) -> mov x;  phi(c, c, ...) -> const c.

    The rewrite is in place, so a converted phi that was not the last one
    leaves real phis sitting behind a non-phi.  That breaks the invariant that
    phis come first -- which later passes rely on (structuring treats a
    phi-only block as a marker, and would stop recognising top-tested loops) --
    so any touched block is re-canonicalised afterwards.  Moving non-phis after
    phis is safe: a phi's arguments are the values arriving from its
    predecessors, never the result of a sibling instruction in its own block.
    """
    changed = False
    touched = []
    for b in func.blocks:
        for insn in b.insns:
            if insn.op != Op.PHI or not insn.srcs:
                continue
            dst = insn.dst
            # Ignore arguments that are the phi's own result: a loop-carried
            # value that is only ever itself is still just the other argument.
            args = [s for s in insn.srcs
                    if not (s.is_var and s.key() == dst.key())]
            if not args:
                continue
            first = args[0]
            if first.is_const:
                if all(s.is_const and s.val == first.val for s in args):
                    insn.op, insn.srcs, insn.aux = Op.CONST, [Const(first.val)], None
                    changed = True
                    touched.append(b)
            elif all(s.is_var and s.key() == first.key() for s in args):
                insn.op, insn.srcs, insn.aux = Op.MOV, [_clone(first)], None
                changed = True
                touched.append(b)

    for b in touched:
        phis = [i for i in b.insns if i.op == Op.PHI]
        if phis:
            b.insns = phis + [i for i in b.insns if i.op != Op.PHI]
    return changed


def propagate(func):
    """Forward copies and constants into their use sites."""
    defs = _defmap(func)
    changed = False
    for insn in func.insns():
        # Never fold a constant into a call/return's ABI operand list: nothing
        # can be optimised there, and doing so kills the instruction that set
        # the argument up, making it vanish from the listing.
        if insn.op in ABI_OPS:
            continue
        for n, s in enumerate(insn.srcs):
            if not s.is_var:
                continue
            d = defs.get(s.key())
            if d is None or d is insn:
                continue                       # live-in, or self-reference
            if d.op == Op.CONST:
                insn.srcs[n] = Const(d.srcs[0].val)
                changed = True
            elif d.op == Op.MOV:
                src = d.srcs[0]
                if src.is_const:
                    insn.srcs[n] = Const(src.val)
                    changed = True
                elif src.is_var and src.key() != s.key():
                    insn.srcs[n] = _clone(src)
                    changed = True
    return changed


def fold(func):
    """Evaluate operations whose inputs are all known constants."""
    changed = False
    for insn in func.insns():
        if insn.op in (Op.CONST, Op.PHI) or insn.dst is None:
            continue
        if insn.op.has_side_effects or insn.op.is_terminator:
            continue
        if not insn.srcs or not all(s.is_const for s in insn.srcs):
            continue
        res = evaluate(insn.op, insn.ew, [s.val for s in insn.srcs], insn.aux)
        if res is None:
            continue
        if insn.comment is None:
            insn.comment = "folded %s" % insn.op.name.lower()
        insn.op, insn.srcs, insn.aux = Op.CONST, [Const(res)], None
        changed = True
    return changed


def simplify_identities(func):
    """
    Peephole the identities the SPU's own idioms create.

    The ISA has no `not`, so compilers spell it `nor rt,ra,ra` -- which
    otherwise renders as `~(a | a)`.  Same-operand `and`/`or` come out of
    register moves, and `xor rt,ra,ra` is the standard zero.
    """
    changed = False
    for insn in func.insns():
        if len(insn.srcs) != 2:
            continue
        a, b = insn.srcs
        same = ((a.is_var and b.is_var and a.key() == b.key())
                or (a.is_const and b.is_const and a.val == b.val))
        if not same:
            continue
        if insn.op in (Op.AND, Op.OR):
            insn.op, insn.srcs = Op.MOV, [a]
        elif insn.op in (Op.NOR, Op.NAND):
            insn.op, insn.srcs = Op.NOT, [a]
        elif insn.op == Op.XOR:
            insn.op, insn.srcs, insn.ew = Op.CONST, [Const(0)], EW.Q
        elif insn.op == Op.EQV:
            insn.op, insn.srcs, insn.ew = Op.CONST, [Const(MASK128)], EW.Q
        else:
            continue
        changed = True
    return changed


def dce(func):
    """Remove definitions nothing reads, iterating until stable."""
    removed = 0
    while True:
        used = _usecount(func)
        victims = []
        for b in func.blocks:
            for insn in b.insns:
                if insn.op.has_side_effects or insn.op.is_terminator:
                    continue
                d = insn.defines()
                if d is not None and used.get(d.key(), 0) == 0:
                    victims.append((b, insn))
        if not victims:
            break
        for b, insn in victims:
            b.insns.remove(insn)
        removed += len(victims)
    return removed


def optimize(func, rounds=16):
    """Run the cleanup pipeline to a fixpoint (bounded)."""
    stats = {"rounds": 0, "removed": 0}
    for _ in range(rounds):
        changed = False
        changed |= simplify_phis(func)
        changed |= simplify_identities(func)
        changed |= propagate(func)
        changed |= fold(func)
        stats["rounds"] += 1
        if not changed:
            break
    stats["removed"] = dce(func)
    return stats
