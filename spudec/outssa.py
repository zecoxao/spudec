"""
Leaving SSA: deciding which values may share a name, and making the merges
that no longer hold into real copies.

Every member of a phi web used to print as one variable, unconditionally.
That is only sound when no two members are live at the same time, and in
sc_iso's ``ss::sc_proxy_hdr::make_hdr`` they were not::

    r9#19  = selb r3#22, r9#18, r11#16
    t60#1  = r9#1 & 0x3FFF0          <- the incoming pointer, still live
    mem#20 = storeq mem#19, t60#1, r9#19

which printed as ``*(qword *)(r9 & 0x3FFF0) = r9;`` -- a store whose address
is the value assigned on the line above.  Not untidy: wrong, and wrong in the
way that is hardest to catch by reading, because each line looks reasonable.

Copy propagation is what creates these.  The compiler stashed the pointer in
another register across the region (``lr r12, r9``) and propagation folded the
copy away, so the incoming value's live range now reaches past a redefinition
of the same register.

So webs are *coloured* rather than coalesced whole: a member joins a colour
only if it interferes with none of that colour's members.  A phi whose
arguments no longer all share its colour is a real merge of distinct
variables, and the copy that realises it is inserted here, into the IR,
before structuring.

Inserting into the IR rather than at rendering time is not a detail.  The
copy belongs on one edge, and the block on that edge can be one the
structurer elides -- an empty fall-through arm of an `if` is exactly that --
so a copy emitted by the renderer simply vanished, leaving a path on which
the variable was never assigned.  A block holding an instruction cannot be
elided.

An edge that is critical in both directions -- the predecessor branches, and
the join is reached from elsewhere too -- has nowhere to put a copy, so the
edge is split: a block holding just the copy is threaded between the two.
That is the standard remedy, and it is what removes the last `phi(...)` lines
from the output.
"""

from .ir import Op, Insn, Var, ABI_OPS


def _liveness_uses(insn, call_args=None):
    """
    The operands of ``insn`` that really read a value.

    An ABI op's operand list is deliberately conservative -- it exists so that
    dead-code elimination cannot delete argument setup -- and taking it at
    face value here makes every incoming argument look live from entry to
    exit, so it interferes with every later value in the same register and
    every one of those webs splits.

    A return's list is a guess about what the *caller* reads, and ``ABI_OPS``
    says how many of its leading operands are real (none).  A call's list is
    narrowed to the arguments the callee's arity says it really takes, the
    same set the call renders; where the arity is unknown there is no entry
    and the whole list stays live, which is the safe direction.
    """
    op = insn.op
    if op not in ABI_OPS:
        return [x for x in insn.uses()]
    n = ABI_OPS.get(op, 0)
    fixed = [x for x in insn.srcs[:n] if x.is_var]
    if op == Op.RET:
        return fixed
    args = None if call_args is None else call_args.get(id(insn))
    if args is None:
        return [x for x in insn.uses()]
    return fixed + [x for x in args if x.is_var]


def liveness(func, call_args=None):
    """
    Live-out sets per block, over SSA keys.

    A phi argument is *not* live in the block holding the phi; it is live out
    of the predecessor that edge comes from, so each argument has to be placed
    on the right edge.  The correspondence is ``srcs[j]`` to ``block.preds[j]``
    -- the pairing :func:`ssa.rename` fills and :func:`ssa.verify` checks the
    length of.  A phi's ``aux`` looks like the same list but is the snapshot
    taken when the phi was inserted; using it put every argument on the wrong
    edge in a loop, and the incoming value of every argument register came out
    live in every block.
    """
    use_of, def_of, edge_use = {}, {}, {}
    for b in func.blocks:
        u, d = set(), set()
        for ins in b.insns:
            if ins.op == Op.PHI:
                dd = ins.defines()
                if dd is not None:
                    d.add(dd.key())
                for arg, p in zip(ins.srcs, b.preds):
                    if arg.is_var:
                        edge_use.setdefault((p.id, b.id), set()).add(arg.key())
                continue
            for x in _liveness_uses(ins, call_args):
                if x.key() not in d:
                    u.add(x.key())
            dd = ins.defines()
            if dd is not None:
                d.add(dd.key())
        use_of[b.id], def_of[b.id] = u, d

    live_in = {b.id: set() for b in func.blocks}
    live_out = {b.id: set() for b in func.blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(func.blocks):
            out = set()
            for s in b.succs:
                out |= live_in[s.id] | edge_use.get((b.id, s.id), set())
            inn = use_of[b.id] | (out - def_of[b.id])
            if out != live_out[b.id] or inn != live_in[b.id]:
                live_out[b.id], live_in[b.id] = out, inn
                changed = True
    return live_out


def interference(func, web_of, call_args=None):
    """
    Pairs of SSA keys in the same web that are live at the same time.

    Walks each block backwards from its live-out set: at a definition, every
    value still live is one whose range overlaps this one's.  Only pairs
    within a web matter, since only those are candidates for sharing a name.
    """
    live_out = liveness(func, call_args)
    inter = set()
    for b in func.blocks:
        live = set(live_out[b.id])
        for ins in reversed(b.insns):
            d = ins.defines()
            if d is not None:
                k = d.key()
                live.discard(k)
                w = web_of.get(k)
                if w is not None:
                    for o in live:
                        if web_of.get(o) == w:
                            inter.add(frozenset((k, o)))
            if ins.op != Op.PHI:
                for x in _liveness_uses(ins, call_args):
                    live.add(x.key())
    return inter


def _webs(func):
    """Phi webs, as ``{root key: [member keys]}`` plus a key -> root map."""
    parent = {}

    def find(k):
        root = k
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(k, k) != k:
            parent[k], k = root, parent[k]
        return root

    for insn in func.insns():
        d = insn.defines()
        if d is not None:
            parent.setdefault(d.key(), d.key())
        for u in insn.uses():
            parent.setdefault(u.key(), u.key())
    for insn in func.insns():
        if insn.op != Op.PHI:
            continue
        d = insn.defines()
        if d is None:
            continue
        for s in insn.srcs:
            if s.is_var:
                ra, rb = find(d.key()), find(s.key())
                if ra != rb:
                    parent[rb] = ra

    webs = {}
    for k in sorted(parent, key=lambda x: (x[0], x[1])):
        webs.setdefault(find(k), []).append(k)
    return webs, {k: find(k) for k in parent}


def colours(func, call_args=None):
    """
    ``{key: colour id}``: which values may share one name.

    A greedy colouring of each web against the interference graph.  Members
    are taken in SSA order so the result is stable, and the first colour of a
    web keeps its lowest-numbered member, which is what makes the incoming
    value of a register keep the plain name.
    """
    webs, web_of = _webs(func)
    inter = interference(func, web_of, call_args)
    out, groups = {}, []
    for root in sorted(webs, key=lambda r: (r[0], r[1])):
        picked = []
        for k in sorted(webs[root]):
            for c in picked:
                if all(frozenset((k, m)) not in inter for m in c):
                    c.append(k)
                    break
            else:
                picked.append([k])
        for c in picked:
            for k in c:
                out[k] = len(groups)
            groups.append(c)
    return out, groups


def _split_edge(func, pred, succ, j, next_ea):
    """
    Thread an empty block between ``pred`` and ``succ``, and return it.

    ``j`` is the index of ``pred`` in ``succ.preds``, which is also the index
    of the corresponding phi argument.  The new block gets an address of its
    own, past every real one, so nothing confuses it with a block that came
    from the binary -- the structurer identifies a conditional branch's taken
    successor by comparing addresses, so the terminator is repointed too.

    Returns None when the same block appears twice in ``pred.succs`` (both
    arms of a branch landing on one join).  The successor index and the
    predecessor index then no longer identify each other, and guessing which
    edge is which is exactly the kind of assumption that produces a listing
    that is confidently wrong.
    """
    if pred.succs.count(succ) != 1:
        return None

    nb = func.new_block(next_ea, next_ea)
    nb.insns.append(Insn(Op.JMP, None, [], ea=next_ea, aux=succ.start_ea,
                         comment="split edge"))
    nb.insns[-1].block = nb

    pred.succs[pred.succs.index(succ)] = nb
    nb.preds.append(pred)
    nb.succs.append(succ)
    succ.preds[j] = nb

    # The taken edge of a branch is found by address, so it has to point here
    # now; a plain jump likewise.
    term = pred.insns[-1] if pred.insns else None
    if term is not None and term.op == Op.CJMP and term.aux \
            and term.aux[0] == succ.start_ea:
        term.aux = (next_ea,) + tuple(term.aux[1:])
    elif term is not None and term.op == Op.JMP and term.aux == succ.start_ea:
        term.aux = next_ea

    # `ssa.verify` checks a phi argument against the predecessor recorded in
    # the phi's own `aux`, so that list has to follow the rewiring.
    for phi in succ.insns:
        if phi.op != Op.PHI:
            break
        if isinstance(phi.aux, list) and j < len(phi.aux):
            phi.aux[j] = nb
    return nb


def lower_phis(func, call_args=None):
    """
    Give every phi argument the colour of its result, by inserting copies.

    Returns the number of copies inserted.  Afterwards every phi argument
    shares its result's colour, so no phi needs to print -- except across an
    edge :func:`_split_edge` declined to touch.
    """
    if not func.ssa:
        return 0
    colour, _ = colours(func, call_args)
    version = {}
    for insn in func.insns():
        d = insn.defines()
        if d is not None:
            version[d.reg] = max(version.get(d.reg, 0), d.ver)
    # Addresses for split blocks, past every real one.
    next_ea = max([b.end_ea for b in func.blocks] or [func.start_ea]) + 4
    split = 0

    n = 0
    for b in list(func.blocks):
        for insn in b.insns:
            if insn.op != Op.PHI:
                continue
            d = insn.defines()
            if d is None:
                continue
            want = colour.get(d.key())
            for j, (arg, pred) in enumerate(zip(list(insn.srcs),
                                                list(b.preds))):
                if not arg.is_var or colour.get(arg.key()) == want:
                    continue
                if len(pred.succs) == 1:
                    where, at_end = pred, True
                elif len(b.preds) == 1:
                    where, at_end = b, False
                else:
                    where = _split_edge(func, pred, b, j, next_ea + 4 * split)
                    if where is None:
                        continue
                    split += 1
                    at_end = True
                version[d.reg] = version.get(d.reg, 0) + 1
                fresh = Var(d.reg, version[d.reg])
                copy = Insn(Op.MOV, fresh, [Var(arg.reg, arg.ver)],
                            ew=insn.ew, ea=where.start_ea,
                            comment="merge")
                copy.block = where
                if at_end:
                    # before the terminator, so control flow stays last
                    k = len(where.insns)
                    while k and where.insns[k - 1].is_terminator:
                        k -= 1
                    where.insns.insert(k, copy)
                else:
                    k = 0
                    while k < len(where.insns) \
                            and where.insns[k].op == Op.PHI:
                        k += 1
                    where.insns.insert(k, copy)
                insn.srcs[j] = Var(fresh.reg, fresh.ver)
                colour[fresh.key()] = want
                n += 1

    if split:
        # New blocks invalidate the dominator tree, and `ssa.verify` checks
        # phi arguments against it.
        from . import ssa
        ssa.compute_dominators(func)
        ssa.compute_dominance_frontiers(func)
    return n
