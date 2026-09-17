"""
Control-flow structuring: CFG -> nested if / while / do-while / loop.

Interval-free structural analysis in the Cifuentes tradition, driven by the
dominator tree (for loops) and the post-dominator tree (for conditional join
points).  Both are already available from :mod:`ssa`.

Shape of the algorithm:

* **Loops** come from back edges ``latch -> header`` where the header
  dominates the latch.  The body is everything that reaches the latch without
  passing through the header; the follow is where the loop exits to.
* **Conditionals** end at their immediate post-dominator.  If that is the
  virtual exit -- one arm returns -- the conditional simply has no join and
  both arms run to completion.
* **Everything is emitted as an endless loop first**, with `break` and
  `continue` inserted whenever control reaches the follow or the header.  A
  refinement pass then recognises `while` (test at the top) and `do-while`
  (test at the bottom).  Doing it in that order avoids special-casing
  self-loops, headers that carry statements, and multi-latch loops, all of
  which appear in real SPU code.
* Anything that does not fit -- irreducible flow, a cross edge into the
  middle of a region -- becomes an explicit `goto` and a label.  Structuring
  never silently drops an edge.

No IDA imports: unit-testable standalone.
"""

from .ir import Op

MAX_DEPTH = 200

# Tail-duplication limits (see _Ctx.may_duplicate).  Measured over two of the
# larger corpus binaries, against no duplication at all (458 gotos, 32797
# lines of pseudocode):
#
#     8 / 2 /  60   ->  352 gotos (-23%), +1.6% lines
#    16 / 3 / 120   ->  279 gotos (-39%), +5.0% lines     <- chosen
#    24 / 4 / 200   ->  231 gotos (-50%), +6.7% lines
#
# The middle setting buys a removed goto for about six extra lines; the
# loosest one costs twice that per goto.  A goto breaks structured reading
# outright, so trading some duplication for it is worth it -- but not without
# limit.
DUP_MAX_INSNS = 16       # only blocks this small get copied
DUP_PER_BLOCK = 3        # how many copies of one block are allowed
DUP_BUDGET = 120         # total copies per function


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------


class Cond(object):
    """
    A branch condition: an SPU conditional branch tests one register against
    zero, so a condition is a value, a test kind, and a negation flag.
    Negation is exact -- `z` and `nz` are complements, as are `hz`/`hnz` --
    which is what makes arm-swapping safe.
    """

    __slots__ = ("value", "kind", "neg")

    _FLIP = {"z": "nz", "nz": "z", "hz": "hnz", "hnz": "hz"}

    def __init__(self, value, kind, neg=False):
        self.value = value
        self.kind = kind
        self.neg = neg

    def negate(self):
        return Cond(self.value, self.kind, not self.neg)

    @property
    def effective(self):
        return self._FLIP[self.kind] if self.neg else self.kind

    def __str__(self):
        k = self.effective
        v = str(self.value)
        if k == "z":
            return "%s == 0" % v
        if k == "nz":
            return "%s != 0" % v
        if k == "hz":
            return "(%s & 0xFFFF) == 0" % v
        return "(%s & 0xFFFF) != 0" % v

    __repr__ = __str__


def cond_of(insn):
    """The condition a CJMP tests, true when the branch is taken."""
    kind = insn.aux[1] if insn.aux else "z"
    return Cond(insn.srcs[0], kind)


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


class Stmt(object):
    __slots__ = ()


class Basic(Stmt):
    """A run of straight-line IR instructions (the terminator excluded)."""
    __slots__ = ("block", "insns")

    def __init__(self, block, insns):
        self.block = block
        self.insns = insns


class If(Stmt):
    __slots__ = ("cond", "then", "els", "ea")

    def __init__(self, cond, then, els, ea=None):
        self.cond = cond
        self.then = then
        self.els = els
        self.ea = ea


class Loop(Stmt):
    """kind is 'loop' (endless), 'while' (top test) or 'dowhile' (bottom)."""
    __slots__ = ("kind", "cond", "body", "header")

    def __init__(self, body, header, kind="loop", cond=None):
        self.body = body
        self.header = header
        self.kind = kind
        self.cond = cond


class Break(Stmt):
    __slots__ = ()


class Continue(Stmt):
    __slots__ = ()


class Goto(Stmt):
    __slots__ = ("target",)

    def __init__(self, target):
        self.target = target


class Label(Stmt):
    __slots__ = ("block",)

    def __init__(self, block):
        self.block = block


class Return(Stmt):
    __slots__ = ("insn",)

    def __init__(self, insn):
        self.insn = insn


class Tail(Stmt):
    """An indirect jump or other terminator we cannot structure further."""
    __slots__ = ("insn",)

    def __init__(self, insn):
        self.insn = insn


# ---------------------------------------------------------------------------
# loops
# ---------------------------------------------------------------------------


class LoopInfo(object):
    __slots__ = ("header", "latches", "body", "follow", "kind")

    def __init__(self, header):
        self.header = header
        self.latches = []
        self.body = set()
        self.follow = None
        self.kind = "loop"


def _natural_body(header, latch):
    """
    Blocks that reach ``latch`` without passing through ``header``.

    The header seeds the visited set so the backward walk stops there.  A
    self-loop (latch is the header) must therefore not start the walk at all:
    stepping into the header's predecessors would drag the loop's *entry*
    block -- and everything before it -- into the body, which then puts the
    loop's follow in the wrong place and emits the code after the loop inside
    it.
    """
    body = {header.id}
    stack = []
    if latch is not header:
        body.add(latch.id)
        stack.append(latch)
    while stack:
        b = stack.pop()
        for p in b.preds:
            if p.id not in body:
                body.add(p.id)
                stack.append(p)
    return body


def find_loops(func, rpo_index):
    """
    ``{header_id: LoopInfo}``, one entry per loop header (a header with two
    back edges is one loop with two latches, not two loops).
    """
    from . import ssa

    loops = {}
    for u in func.blocks:
        for v in u.succs:
            if not ssa.dominates(v, u):
                continue
            li = loops.get(v.id)
            if li is None:
                li = loops[v.id] = LoopInfo(v)
            li.latches.append(u)
            li.body |= _natural_body(v, u)

    by_id = {b.id: b for b in func.blocks}
    for li in loops.values():
        exits = []
        for bid in li.body:
            for s in by_id[bid].succs:
                if s.id not in li.body and s not in exits:
                    exits.append(s)
        # Several exits are possible (breaks); the follow is the one the code
        # falls out to naturally, which is the earliest in reverse post-order.
        if exits:
            exits.sort(key=lambda b: rpo_index.get(b.id, 1 << 30))
            li.follow = exits[0]

        head_term = li.header.insns[-1] if li.header.insns else None
        head_tests = (head_term is not None and head_term.op == Op.CJMP and
                      any(s.id not in li.body for s in li.header.succs))
        if li.header in li.latches:
            # A self-loop runs its body before testing: bottom-tested.
            li.kind = "dowhile"
        elif head_tests:
            li.kind = "while"
        else:
            li.kind = "dowhile" if any(
                l.insns and l.insns[-1].op == Op.CJMP and
                any(s.id not in li.body for s in l.succs)
                for l in li.latches) else "loop"
    return loops


# ---------------------------------------------------------------------------
# structuring
# ---------------------------------------------------------------------------


class _Ctx(object):
    def __init__(self, func, ipdom, loops):
        self.func = func
        self.ipdom = ipdom
        self.loops = loops
        self.emitted = set()
        self.open_loops = set()
        self.loop_stack = []
        self.labels = set()
        self.gotos = 0
        self.why = {}
        self.dups = 0
        self.dup_per = {}

    def may_duplicate(self, b):
        """
        Whether re-arriving at ``b`` should re-emit it instead of jumping.

        Emitting a block twice is semantically free -- only one path runs it --
        and it removes the commonest goto by far: a shared tail that one arm
        of a conditional emitted inline and the other can only jump to,
        because their join sits further out than the tail does.

        Bounded three ways so it cannot blow up: small blocks only, a cap per
        block, and a budget per function.  A loop header is never duplicated;
        that is what `continue` is for.
        """
        if b.id in self.loops or self.dups >= DUP_BUDGET:
            return False
        if self.dup_per.get(b.id, 0) >= DUP_PER_BLOCK:
            return False
        return sum(1 for i in b.insns if i.op != Op.PHI) <= DUP_MAX_INSNS

    def note_duplicate(self, b):
        self.dups += 1
        self.dup_per[b.id] = self.dup_per.get(b.id, 0) + 1

    def goto(self, b, why):
        self.labels.add(b.id)
        self.gotos += 1
        self.why[why] = self.why.get(why, 0) + 1
        return Goto(b)

    def loop_control(self, b):
        """Break / continue / goto when control reaches a loop boundary."""
        if not self.loop_stack:
            return None
        h, f = self.loop_stack[-1]
        if b is h:
            return Continue()
        if f is not None and b is f:
            return Break()
        for oh, of in reversed(self.loop_stack[:-1]):
            if b is oh or (of is not None and b is of):
                return self.goto(b, 'outer-loop boundary')
        return None

    def cjmp_targets(self, b):
        """(taken, not-taken) successors of a block ending in CJMP."""
        term = b.insns[-1]
        target_ea = term.aux[0] if term.aux else None
        taken = fall = None
        for s in b.succs:
            if s.start_ea == target_ea and taken is None:
                taken = s
            else:
                fall = s
        if taken is None:
            taken = b.succs[0] if b.succs else None
        if fall is None:
            fall = taken
        return taken, fall

    def if_follow(self, b, stop):
        f = self.ipdom.get(b)
        if f is None:
            return stop              # one arm leaves the function
        return f


def _stmts(b, stop, ctx, depth=0, skip_control=None):
    out = []
    cur = b
    first = True
    while cur is not None and cur is not stop:
        if depth > MAX_DEPTH:
            out.append(ctx.goto(cur, 'nesting depth limit'))
            return out

        if not (first and cur is skip_control):
            ctl = ctx.loop_control(cur)
            if ctl is not None:
                out.append(ctl)
                return out

        if cur.id in ctx.emitted:
            # Several guards jumping to one `return` is the most common source
            # of gotos in real code.  Re-emitting a bare return is identical in
            # meaning -- only one of them can ever run -- and far more readable
            # than a label, so duplicate instead of branching.
            if _duplicable(cur):
                last = cur.insns[-1]
                out.append(Tail(last) if last.op == Op.JMP
                           else Return(last))
                return out
            if ctx.may_duplicate(cur):
                ctx.note_duplicate(cur)
                # fall through and emit the block again
            else:
                out.append(ctx.goto(cur, 'block already emitted'))
                return out

        li = ctx.loops.get(cur.id)
        if li is not None and cur.id not in ctx.open_loops:
            out.extend(_loop(li, ctx, depth + 1))
            cur = li.follow
            first = False
            continue

        ctx.emitted.add(cur.id)
        first = False

        # A Label marker for every block, always: a goto target may be a block
        # with no statements of its own (just a terminator), and hanging label
        # emission off the statement list would then produce a goto to a label
        # that is never printed.  The renderer drops the ones nothing jumps to.
        out.append(Label(cur))

        term = cur.insns[-1] if cur.insns else None
        body = cur.insns[:-1] if (term is not None and term.is_terminator) \
            else cur.insns
        if body:
            out.append(Basic(cur, body))

        if term is None:
            break
        if term.op == Op.CJMP:
            f = ctx.if_follow(cur, stop)
            taken, fall = ctx.cjmp_targets(cur)
            then = _stmts(taken, f, ctx, depth + 1)
            els = _stmts(fall, f, ctx, depth + 1)
            # An arm holding nothing but labels is an arm that goes straight
            # to the join.  Emitting `if (c) { loc_X: }` is noise; the label
            # belongs after the if, which is exactly where that arm leads --
            # so a jump to it still lands in the right place.
            trailing = []
            if then and all(_marker(s) for s in then):
                trailing += then
                then = []
            if els and all(_marker(s) for s in els):
                trailing += els
                els = []
            cond = cond_of(term)
            if not then and els:
                cond, then, els = cond.negate(), els, []
            # Always emit the branch, even if both arms came out empty: it is
            # still a real test, and dropping it would let DCE delete the
            # computation behind the condition.
            out.append(If(cond, then, els, term.ea))
            out.extend(trailing)
            cur = f
        elif term.op in (Op.RET, Op.STOP):
            out.append(Return(term))
            return out
        elif term.op == Op.IJMP:
            out.append(Tail(term))
            return out
        elif term.op == Op.JMP and not cur.succs:
            out.append(Tail(term))          # leaves the function
            return out
        elif len(cur.succs) == 1:
            cur = cur.succs[0]
        elif not cur.succs:
            break
        else:
            cur = cur.succs[0]
    return out


def _duplicable(b):
    """
    A block that is nothing but a return, a stop, or a jump out of the
    function: no successors, no statements of its own beyond phis.  Copying
    one is free of side effects by construction, and `goto loc_3A0;` repeated
    reads far better than a label and a chain of gotos into it.
    """
    if b.succs or not b.insns:
        return False
    if b.insns[-1].op not in (Op.RET, Op.STOP, Op.JMP):
        return False
    return all(i.op == Op.PHI for i in b.insns[:-1])


def _loop(li, ctx, depth):
    ctx.open_loops.add(li.header.id)
    ctx.loop_stack.append((li.header, li.follow))
    body = _stmts(li.header, None, ctx, depth, skip_control=li.header)
    ctx.loop_stack.pop()
    ctx.open_loops.discard(li.header.id)

    # A label on the header belongs outside the loop: `loc_X: while (...)`
    # re-enters at the top, which is exactly what a jump to the header means.
    lead = []
    while body and isinstance(body[0], Label):
        lead.append(body.pop(0))
    return lead + [_refine(Loop(body, li.header))]


def _marker(s):
    """
    A statement that carries no executable code of its own.

    Labels are obviously markers.  So is a Basic holding nothing but phi
    nodes: a loop header almost always has them, and the renderer absorbs
    them into variable names, so letting one sit at the top of a body would
    stop every top-tested loop from being recognised as a `while`.
    """
    if isinstance(s, Label):
        return True
    return isinstance(s, Basic) and all(i.op == Op.PHI for i in s.insns)


def _real(body):
    """(index, statement) pairs, skipping markers."""
    return [(i, s) for i, s in enumerate(body) if not _marker(s)]


def _drop(body, *idx):
    skip = set(idx)
    return [s for i, s in enumerate(body) if i not in skip]


def _is(arm, cls):
    return len(arm) == 1 and isinstance(arm[0], cls)


def _refine(loop):
    """
    Turn an endless loop with a boundary test into `while` or `do-while`.

    Working this way round means a header that carries statements, a
    self-loop, or several latches all fall out of the same code: if the test
    is not isolated at a boundary the loop simply stays endless with an
    explicit break, which is still correct -- just less pretty.
    """
    body = loop.body
    real = _real(body)
    if not real:
        return loop

    # -- top test -> while -------------------------------------------------
    # `if (c) break; else { body; continue; }` -- a header that only tests,
    # with the exit and the body as the two arms.
    if len(real) == 1 and isinstance(real[0][1], If):
        i0, s0 = real[0]
        for brk, rest, invert in ((s0.then, s0.els, True),
                                  (s0.els, s0.then, False)):
            if _is(brk, Break) and rest and isinstance(rest[-1], Continue):
                loop.kind = "while"
                loop.cond = s0.cond.negate() if invert else s0.cond
                loop.body = body[:i0] + rest[:-1]
                return loop

    # `if (c) { body; continue; } break;` -- same loop, but the conditional's
    # follow was the loop follow, so the exit landed after the if.
    if len(real) == 2:
        (i0, s0), (i1, s1) = real
        if isinstance(s0, If) and isinstance(s1, Break):
            for arm, other, invert in ((s0.then, s0.els, False),
                                       (s0.els, s0.then, True)):
                if arm and not other and isinstance(arm[-1], Continue):
                    loop.kind = "while"
                    loop.cond = s0.cond.negate() if invert else s0.cond
                    loop.body = body[:i0] + arm[:-1]
                    return loop

    i0, s0 = real[0]
    if isinstance(s0, If) and len(real) > 1:
        if _is(s0.then, Break) and not s0.els:
            loop.kind, loop.cond = "while", s0.cond.negate()
            loop.body = _drop(body, i0)
            return loop
        if _is(s0.els, Break) and not s0.then:
            loop.kind, loop.cond = "while", s0.cond
            loop.body = _drop(body, i0)
            return loop

    # -- bottom test -> do-while ------------------------------------------
    # `if (c) continue; break;` is the shape a self-loop naturally produces:
    # the conditional's follow is the loop follow, so the break lands after
    # the if rather than inside it.
    if len(real) >= 2:
        (i1, s1), (i2, s2) = real[-2], real[-1]
        if isinstance(s1, If) and isinstance(s2, Break):
            if _is(s1.then, Continue) and not s1.els:
                loop.kind, loop.cond = "dowhile", s1.cond
                loop.body = _drop(body, i1, i2)
                return loop
            if _is(s1.els, Continue) and not s1.then:
                loop.kind, loop.cond = "dowhile", s1.cond.negate()
                loop.body = _drop(body, i1, i2)
                return loop

    iL, sL = real[-1]
    if isinstance(sL, If):
        for arm, other, cls, invert in (
                (sL.then, sL.els, Continue, False),
                (sL.els, sL.then, Continue, True),
                (sL.then, sL.els, Break, True),
                (sL.els, sL.then, Break, False)):
            if _is(arm, cls) and (not other or _is(other, _OTHER[cls])):
                loop.kind = "dowhile"
                loop.cond = sL.cond.negate() if invert else sL.cond
                loop.body = _drop(body, iL)
                return loop
    elif isinstance(sL, Continue):
        loop.body = _drop(body, iL)
    return loop


_OTHER = {Continue: Break, Break: Continue}


def structure(func):
    """
    Structure ``func`` and return ``(statements, info)``.

    ``info`` carries the label set and goto count -- a non-zero goto count is
    the honest signal that part of the flow did not fit a nested construct.
    """
    from . import ssa

    rpo = ssa.compute_dominators(func)
    rpo_index = {b.id: i for i, b in enumerate(rpo)}
    ssa.compute_dominance_frontiers(func)
    ipdom = ssa.compute_postdominators(func)
    loops = find_loops(func, rpo_index)

    ctx = _Ctx(func, ipdom, loops)
    body = _stmts(func.entry, None, ctx)

    unreached = [b for b in func.blocks if b.id not in ctx.emitted]
    return body, {
        "labels": ctx.labels,
        "gotos": ctx.gotos,
        "goto_why": ctx.why,
        "duplicated": ctx.dups,
        "loops": len(loops),
        "unreached": unreached,
    }
