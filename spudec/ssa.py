"""
SSA construction: dominators, dominance frontiers, phi insertion, renaming.

Standard Cytron et al. placement, with two practical adjustments:

* **Semi-pruned.**  Only registers with an upward-exposed use somewhere (i.e.
  read in a block before being written there) get phi nodes.  The lifter emits
  a lot of single-block temporaries for address arithmetic and materialised
  constants; without this, every one of them would collect dead phis at every
  join and drown the output.

* **Iterative, not recursive.**  Renaming walks the dominator tree with an
  explicit stack.  Large hand-written SPU routines nest deeply enough to blow
  Python's recursion limit otherwise.

Registers live on entry (arguments, sp, lr, and the memory/channel pseudo
registers) get version 0 with ``def_insn is None``, which reads as "value on
entry to the function".

No IDA imports here: this is unit-testable standalone.
"""

from .ir import Op, Insn, Var
from . import regs


# ---------------------------------------------------------------------------
# dominator tree
# ---------------------------------------------------------------------------


def reverse_postorder(func):
    """Iterative DFS postorder from the entry, reversed."""
    order = []
    seen = set()
    stack = [(func.entry, iter(func.entry.succs))]
    seen.add(func.entry.id)
    while stack:
        blk, it = stack[-1]
        advanced = False
        for s in it:
            if s.id not in seen:
                seen.add(s.id)
                stack.append((s, iter(s.succs)))
                advanced = True
                break
        if not advanced:
            order.append(blk)
            stack.pop()
    order.reverse()
    return order


def compute_dominators(func):
    """
    Cooper/Harvey/Kennedy iterative dominators.  Sets ``idom`` on every block
    and fills ``dom_children``.  The entry block's idom is itself.
    """
    rpo = reverse_postorder(func)
    rpo_num = {b.id: i for i, b in enumerate(rpo)}

    for b in func.blocks:
        b.idom = None
        b.dom_children = []
    entry = func.entry
    entry.idom = entry

    def intersect(a, b):
        while a is not b:
            while rpo_num[a.id] > rpo_num[b.id]:
                a = a.idom
            while rpo_num[b.id] > rpo_num[a.id]:
                b = b.idom
        return a

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b is entry:
                continue
            new_idom = None
            for p in b.preds:
                if p.idom is None or p.id not in rpo_num:
                    continue            # not yet processed / unreachable
                new_idom = p if new_idom is None else intersect(p, new_idom)
            if new_idom is not None and b.idom is not new_idom:
                b.idom = new_idom
                changed = True

    for b in func.blocks:
        if b.idom is not None and b is not entry:
            b.idom.dom_children.append(b)
    return rpo


def compute_dominance_frontiers(func):
    for b in func.blocks:
        b.domfront = set()
    for b in func.blocks:
        if len(b.preds) < 2:
            continue
        for p in b.preds:
            runner = p
            while runner is not None and runner is not b.idom:
                runner.domfront.add(b)
                if runner.idom is runner:       # reached the entry
                    break
                runner = runner.idom


class _VirtualExitType(object):
    __slots__ = ()

    def __repr__(self):
        return "<exit>"


_VirtualExit = _VirtualExitType()


def compute_postdominators(func):
    """
    Immediate post-dominators, as ``{block: block}``.

    These are just dominators of the reversed CFG rooted at a virtual exit
    that every returning block flows to.  The virtual exit is represented by
    ``None``, so an entry mapping a block to ``None`` means "this conditional
    has no join point inside the function" -- one arm returns.

    A block that cannot reach the exit at all (inside an endless loop) is
    absent from the result.
    """
    exits = [b for b in func.blocks
             if not b.succs or (b.insns and
                                b.insns[-1].op in (Op.RET, Op.STOP))]
    if not exits:
        exits = func.blocks[-1:] if func.blocks else []
    if not exits:
        return {}

    # A distinct sentinel, not None: None is what dict.get() returns for a
    # missing key, and conflating the two silently drops the exit's own entry.
    EXIT = _VirtualExit
    rsucc = {EXIT: list(exits)}
    rpred = {EXIT: []}
    for b in func.blocks:
        rsucc[b] = list(b.preds)
        rpred[b] = list(b.succs)
    for b in exits:
        rpred[b] = rpred[b] + [EXIT]

    # Post-order over the reversed graph, then reverse it.
    order = []
    seen = {EXIT}
    stack = [(EXIT, iter(rsucc[EXIT]))]
    while stack:
        nd, it = stack[-1]
        advanced = False
        for s in it:
            if s not in seen:
                seen.add(s)
                stack.append((s, iter(rsucc[s])))
                advanced = True
                break
        if not advanced:
            order.append(nd)
            stack.pop()
    order.reverse()
    num = {nd: i for i, nd in enumerate(order)}

    ipdom = {EXIT: EXIT}

    def intersect(a, b):
        while a is not b:
            while num[a] > num[b]:
                a = ipdom[a]
            while num[b] > num[a]:
                b = ipdom[b]
        return a

    changed = True
    while changed:
        changed = False
        for nd in order:
            if nd is EXIT:
                continue
            new = _MISSING
            for p in rpred[nd]:
                if p not in ipdom or p not in num:
                    continue
                new = p if new is _MISSING else intersect(p, new)
            if new is not _MISSING and ipdom.get(nd, _MISSING) is not new:
                ipdom[nd] = new
                changed = True

    del ipdom[EXIT]
    # Hand back None for "post-dominated only by the function exit", which is
    # what a caller wants to see: this conditional has no join point.
    return {b: (None if pd is EXIT else pd) for b, pd in ipdom.items()}


_MISSING = object()


def dominates(a, b):
    """True if block ``a`` dominates block ``b``."""
    while b is not None:
        if b is a:
            return True
        if b.idom is b:
            return False
        b = b.idom
    return False


# ---------------------------------------------------------------------------
# phi placement
# ---------------------------------------------------------------------------


def _collect(func):
    """
    Returns ``(globals_, defsites)``.

    ``globals_`` is the set of registers with an upward-exposed use in some
    block -- the registers that can possibly need a phi.  ``defsites[r]`` is
    the set of blocks that define ``r``.
    """
    globals_ = set()
    defsites = {}
    for b in func.blocks:
        killed = set()
        for insn in b.insns:
            for u in insn.uses():
                if u.reg not in killed:
                    globals_.add(u.reg)
            d = insn.defines()
            if d is not None:
                killed.add(d.reg)
                defsites.setdefault(d.reg, set()).add(b)
    return globals_, defsites


def insert_phis(func):
    globals_, defsites = _collect(func)
    placed = {}                                  # reg -> set of block ids
    count = 0

    for reg in sorted(globals_):
        sites = defsites.get(reg)
        if not sites:
            continue
        placed[reg] = set()
        work = list(sites)
        while work:
            b = work.pop()
            for df in b.domfront:
                if df.id in placed[reg]:
                    continue
                placed[reg].add(df.id)
                phi = Insn(Op.PHI, Var(reg),
                           [Var(reg) for _ in df.preds],
                           ea=df.start_ea, aux=list(df.preds))
                phi.block = df
                df.insns.insert(0, phi)
                count += 1
                if df not in sites:
                    work.append(df)
    return count


# ---------------------------------------------------------------------------
# renaming
# ---------------------------------------------------------------------------


def rename(func):
    """
    Walk the dominator tree, giving every definition a fresh version and every
    use the version currently in scope.
    """
    stacks = {}
    counter = {}

    def top(reg):
        st = stacks.get(reg)
        if not st:
            # Live on entry: version 0, no defining instruction.
            v = Var(reg, 0)
            stacks[reg] = [v]
            counter.setdefault(reg, 0)
            return v
        return st[-1]

    ENTER, EXIT = 0, 1
    scope = {}                      # block id -> regs pushed in that block
    work = [(func.entry, ENTER)]
    while work:
        blk, phase = work.pop()

        if phase == EXIT:
            for reg in scope.pop(blk.id):
                stacks[reg].pop()
            continue

        pushed = []
        for insn in blk.insns:
            if insn.op != Op.PHI:
                for u in insn.srcs:
                    if u.is_var:
                        cur = top(u.reg)
                        u.ver = cur.ver
                        u.def_insn = cur.def_insn
            d = insn.defines()
            if d is not None:
                n = counter.get(d.reg, 0) + 1
                counter[d.reg] = n
                d.ver = n
                d.def_insn = insn
                stacks.setdefault(d.reg, [Var(d.reg, 0)]).append(d)
                pushed.append(d.reg)

        # Fill in our slot in each successor's phi nodes.  A block can appear
        # more than once in a successor's pred list (both edges of a
        # conditional landing on the same block), so fill every matching slot.
        for s in blk.succs:
            for j, p in enumerate(s.preds):
                if p is not blk:
                    continue
                for phi in s.phis:
                    src = phi.srcs[j]
                    cur = top(src.reg)
                    src.ver = cur.ver
                    src.def_insn = cur.def_insn

        scope[blk.id] = pushed
        work.append((blk, EXIT))
        for c in blk.dom_children:
            work.append((c, ENTER))


def to_ssa(func):
    """Convert ``func`` in place from pre-SSA to SSA form."""
    if func.ssa:
        return func
    compute_dominators(func)
    compute_dominance_frontiers(func)
    insert_phis(func)
    rename(func)
    func.ssa = True
    return func


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify(func):
    """
    Check the SSA invariants and return a list of human-readable problems.

    Worth running after every pass while the lifter is still growing: a
    mis-modelled instruction usually shows up here as a use with no reaching
    definition long before it shows up as wrong output.
    """
    problems = []
    defs = {}

    for b in func.blocks:
        for insn in b.insns:
            d = insn.defines()
            if d is None:
                continue
            if d.key() in defs:
                problems.append(
                    "0x%X: %s defined twice" % (insn.ea or 0, d))
            defs[d.key()] = (insn, b)

    for b in func.blocks:
        seen_nonphi = False
        for insn in b.insns:
            if insn.op == Op.PHI:
                if seen_nonphi:
                    problems.append("B%d: phi after a non-phi" % b.id)
                if len(insn.srcs) != len(b.preds):
                    problems.append(
                        "B%d: phi for %s has %d args but %d preds"
                        % (b.id, insn.dst, len(insn.srcs), len(b.preds)))
            else:
                seen_nonphi = True

            for u in insn.uses():
                if u.ver == 0:
                    continue                     # live-in, fine
                if u.key() not in defs:
                    problems.append(
                        "0x%X: use of %s with no definition" % (insn.ea or 0, u))
                    continue
                dinsn, dblk = defs[u.key()]
                if insn.op == Op.PHI:
                    # A phi argument must be available at the end of the
                    # corresponding predecessor, not at the phi itself.
                    pred = insn.aux[insn.srcs.index(u)]
                    if not dominates(dblk, pred):
                        problems.append(
                            "B%d: phi arg %s not available from B%d"
                            % (b.id, u, pred.id))
                elif not dominates(dblk, b):
                    problems.append(
                        "0x%X: use of %s not dominated by its definition"
                        % (insn.ea or 0, u))

        if b.insns and not b.insns[-1].is_terminator:
            problems.append("B%d: no terminator" % b.id)

    return problems
