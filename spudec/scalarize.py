"""
Scalarisation: recover scalar memory accesses and preferred-slot arithmetic.

The SPU can only load and store aligned quadwords, so a compiler builds scalar
accesses out of idioms.  Left alone these dominate the IR and make it
unreadable; recognised, they collapse to one line each.

**Scalar store** -- ``*(int *)p = v``::

    cwd    ctl, 0(p)          ctl = shuffle control for p's position in its qword
    lqd    old, 0(p)          read the containing quadword
    shufb  new, v, old, ctl   splice v into it
    stqd   new, 0(p)          write it back

  becomes ``store.w mem, p, v``.  Four instructions plus their address
  arithmetic, gone.

**Scalar load** -- ``v = *(int *)p``::

    lqd    q, 0(p)            read the containing quadword
    rotqby v, q, p            rotate the wanted bytes into the preferred slot

  becomes ``load.w mem, p``.  The width comes from demand analysis: the rotate
  leaves the value at byte 0, and how many bytes anything reads from there is
  exactly the access size.

Matching is driven by :mod:`lanes` demand analysis and by exact address
comparison, not by instruction adjacency -- the scheduler interleaves these
idioms heavily on a dual-issue in-order machine, so a peephole over adjacent
instructions would miss most of them.

No IDA imports: unit-testable standalone.
"""

from .ir import Op, EW, Var, Const, Insn, INFIX
from .sem import word0, pack
from . import lanes, regs
from .lanes import ALL, NONE, W0, LEAD, SLOT_LOW


# Masks the lifter uses to form a quadword-aligned local-store address.  The
# LS wrap (0x3FFF0) is the accurate one; plain ~0xF is accepted so the pass
# keeps working if the lifter is configured without the wrap.
ALIGN_MASKS = frozenset((0x0003FFF0, 0xFFFFFFF0))


def _clone(v):
    if v.is_const:
        return Const(v.val)
    c = Var(v.reg, v.ver)
    c.def_insn = v.def_insn
    return c


def _defmap(func):
    m = {}
    for i in func.insns():
        d = i.defines()
        if d is not None:
            m[d.key()] = i
    return m


# ---------------------------------------------------------------------------
# address normalisation
# ---------------------------------------------------------------------------


def addr_expr(v, defs, limit=16):
    """
    Normalise an address value to ``(base, offset, aligned)``.

    ``base`` is the SSA key of the register the address is built from, or None
    for a literal.  ``aligned`` means the value is ``(base + offset) & ~0xF``,
    i.e. it went through the quadword-align mask the lifter emits for lqd/stqd.

    Walking is outside-in, so an ``and`` seen before any ``add`` really is the
    outermost operation.  If an ``add`` is seen first the expression is
    ``(x & ~0xF) + k``, which is *not* ``(x + k) & ~0xF`` -- that case is
    rejected rather than mis-simplified.
    """
    off = 0
    aligned = False
    saw_add = False
    cur = v
    for _ in range(limit):
        if cur.is_const:
            off = (off + word0(cur.val)) & 0xFFFFFFFF
            return (None, off, aligned or (off & 0xF) == 0)

        d = defs.get(cur.key())
        if d is None:
            return (cur.key(), off & 0xFFFFFFFF, aligned)

        if d.op == Op.MOV:
            cur = d.srcs[0]
            continue
        if d.op == Op.CONST:
            off = (off + word0(d.srcs[0].val)) & 0xFFFFFFFF
            return (None, off, aligned or (off & 0xF) == 0)
        if (d.op == Op.AND and len(d.srcs) == 2 and d.srcs[1].is_const
                and word0(d.srcs[1].val) in ALIGN_MASKS):
            if saw_add:
                return (cur.key(), off & 0xFFFFFFFF, False)
            aligned = True
            cur = d.srcs[0]
            continue
        if d.op == Op.ADD and d.ew == EW.W and len(d.srcs) == 2:
            a, b = d.srcs
            if b.is_const:
                off = (off + word0(b.val)) & 0xFFFFFFFF
                saw_add = True
                cur = a
                continue
            if a.is_const:
                off = (off + word0(a.val)) & 0xFFFFFFFF
                saw_add = True
                cur = b
                continue
        return (cur.key(), off & 0xFFFFFFFF, aligned)
    return (cur.key() if cur.is_var else None, off & 0xFFFFFFFF, aligned)


def _same_target(a, b):
    """Two address expressions naming the same byte address."""
    return a[0] == b[0] and a[1] == b[1]


def _same_slot(a, b):
    """
    Same base, and the same byte position within a quadword.

    ``cwd``/``cbd`` only consume ``addr & 0xF``, so a compiler is free to
    generate the control mask from a cheaper but congruent address: real code
    pairs ``lqd r40, 0x400(r33)`` with ``cbd r41, 0(r33)`` because 0x400 is a
    multiple of 16 and the byte position is therefore identical.  Requiring
    the two offsets to be *equal* misses every one of those.
    """
    return a[0] == b[0] and (a[1] & 0xF) == (b[1] & 0xF)


def _slot_of(expr):
    """
    The byte position within a quadword that an address names, when that is
    provable without knowing the base register's value.

    Two cases are provable.  A literal address: the low four bits are right
    there.  And anything built from the stack pointer: the SPU ABI requires
    the stack to stay quadword-aligned, so `sp + k` sits at `k & 0xF`.

    This matters because compilers generate the insertion mask from whatever
    known-aligned register is cheapest rather than from the address being
    written -- real code here pairs `stqd`/`lqd` on `r3` with `cwd $5, 0($sp)`,
    because both are 16-byte aligned so the control mask is the same either
    way.  Requiring a shared base rejects all of those.
    """
    base, off, _ = expr
    if base is None:
        return off & 0xF
    if base[0] == regs.SP:
        return off & 0xF
    return None


def _disjoint(a, b):
    """
    Provably different quadwords.

    Only decidable with a common base: two addresses ``base+i`` and ``base+j``
    with ``|i-j| >= 16`` always fall in different quadwords, whatever ``base``
    is.  Different or unknown bases are not provable and must be assumed to
    alias.
    """
    if a[0] != b[0]:
        return False
    d = (a[1] - b[1]) & 0xFFFFFFFF
    if d >= 0x80000000:
        d = 0x100000000 - d
    return d >= 16


def unaligned_of(v, defs, limit=16):
    """The value feeding the outermost quadword-align mask, if there is one."""
    cur = v
    for _ in range(limit):
        if cur.is_const:
            return cur
        d = defs.get(cur.key())
        if d is None:
            return cur
        if d.op == Op.MOV:
            cur = d.srcs[0]
            continue
        if (d.op == Op.AND and len(d.srcs) == 2 and d.srcs[1].is_const
                and word0(d.srcs[1].val) in ALIGN_MASKS):
            return d.srcs[0]
        return cur
    return cur


def _fresh_reg(func):
    """A register index nothing else in the function uses."""
    n = getattr(func, "_scal_tmp", None)
    if n is None:
        n = 0
        for i in func.insns():
            if i.dst is not None:
                n = max(n, i.dst.reg)
            for u in i.uses():
                n = max(n, u.reg)
        n += 1
    func._scal_tmp = n + 1
    return n


def _offset_addr(func, insn, base_val, slot):
    """
    Materialise ``base + slot`` just before ``insn`` and return a Var for it.

    Used when the scalar sits at a non-zero byte position in its quadword: the
    hardware reads ``align(addr) + slot``, so that is the address to show --
    and it is exact, not an assumption about the base's alignment.
    """
    if slot == 0:
        return _clone(base_val)
    r = _fresh_reg(func)
    add = Insn(Op.ADD, Var(r, 1), [_clone(base_val),
                                   Const(pack([slot] * 4, EW.W))],
               ew=EW.W, ea=insn.ea, scalar=True,
               comment="byte %d within the quadword" % slot)
    blk = insn.block
    add.block = blk
    blk.insns.insert(blk.insns.index(insn), add)
    v = Var(r, 1)
    v.def_insn = add
    return v


def _effective_addr(addr, sa, ca, defs):
    """
    The byte address of the scalar being accessed.

    With a register base this is just the pre-align value (``base + off``),
    because the align mask and the in-quadword position reconstruct it
    exactly.  With a literal the low bits were masked away by the lifter, so
    they come back from the control-mask address.
    """
    if sa[0] is None:
        return Const((((sa[1] & ~0xF) | (ca[1] & 0xF)) & 0xFFFFFFFF) << 96)
    return _clone(unaligned_of(addr, defs))


# Ops that define a new memory state, with the previous state in srcs[0].
_MEM_DEFS = frozenset((Op.STOREQ, Op.STORE, Op.SYNC, Op.CALL, Op.ICALL))


def _chain_hazards(store_mem, load_mem, defs, target, limit=64):
    """
    Walk the memory chain back from the store's input state to the load's.

    Returns a list of intervening writes that could not be proved disjoint
    from ``target`` (empty means the read-modify-write is airtight), or None
    if the chain cannot be followed at all -- a phi, a gap, or a depth blowout.
    """
    hazards = []
    cur = store_mem
    for _ in range(limit):
        if not cur.is_var:
            return None
        if cur.key() == load_mem.key():
            return hazards
        d = defs.get(cur.key())
        if d is None or d.op == Op.PHI or d.op not in _MEM_DEFS:
            return None
        if d.op in (Op.STOREQ, Op.STORE):
            if not _disjoint(addr_expr(d.srcs[1], defs), target):
                hazards.append(d)
        else:
            hazards.append(d)                # a call may write anything
        cur = d.srcs[0]
    return None


# ---------------------------------------------------------------------------
# idiom matching
# ---------------------------------------------------------------------------


def _match_store(insn, defs, strict=False):
    """
    ``storeq(mem, addr, shufb(val, loadq(mem', addr), genctl(addr)))``
    -> ``(genctl insn, value operand, effective address, hazards)``.
    """
    mem, addr, val = insn.srcs
    if not val.is_var:
        return None
    shuf = defs.get(val.key())
    if shuf is None or shuf.op != Op.SHUFB:
        return None
    a, b, c = shuf.srcs
    if not (b.is_var and c.is_var):
        return None
    ctl = defs.get(c.key())
    if ctl is None or ctl.op != Op.GENCTL:
        return None
    ld = defs.get(b.key())
    if ld is None or ld.op != Op.LOADQ:
        return None

    sa = addr_expr(addr, defs)
    la = addr_expr(ld.srcs[1], defs)
    ca = addr_expr(ctl.srcs[0], defs)
    if not (sa[2] and la[2]):
        return None                      # both must be the aligned form
    if not _same_target(sa, la):
        return None

    slot = None
    if _same_slot(sa, ca):
        eff = _effective_addr(addr, sa, ca, defs)
    else:
        # The mask may be generated from a different, known-aligned register.
        # When its slot is provable the scalar lands at that byte of the
        # quadword the store targets, whatever base the mask came from.
        slot = _slot_of(ca)
        if slot is None:
            return None
        eff = None                      # built by the caller, needs the block

    # The read-modify-write reads one memory state and writes back onto
    # another.  If anything wrote the same quadword in between, folding the
    # merge away changes what survives -- the machine code would be discarding
    # that write, which a compiler only emits when it knows they cannot alias.
    hazards = _chain_hazards(mem, ld.srcs[0], defs, sa)
    if hazards is None:
        return None
    if hazards and strict:
        return None
    return ctl, a, eff, hazards, slot


def _match_load(insn, defs):
    """``rotqby(loadq(mem, addr & ~0xF), addr)`` -> the loadq instruction."""
    if insn.op != Op.QROTBY:
        return None
    q, cnt = insn.srcs
    if not q.is_var:
        return None
    ld = defs.get(q.key())
    if ld is None or ld.op != Op.LOADQ:
        return None

    la = addr_expr(ld.srcs[1], defs)
    ca = addr_expr(cnt, defs)
    if not la[2]:
        return None

    # Same relaxation as the store side: the rotate count only contributes
    # its low 4 bits, so it may come from a congruent address.
    if _same_slot(la, ca):
        return ld, _effective_addr(ld.srcs[1], la, ca, defs), None

    # Absolute form: `lqa` at an aligned literal plus `rotqbyi` by the byte
    # offset within that quadword -- two literals naming one address.
    if la[0] is None and ca[0] is None:
        return (ld,
                Const((((la[1] & ~0xF) | (ca[1] & 0xF)) & 0xFFFFFFFF) << 96),
                None)

    # The rotate count comes from somewhere else whose slot is provable (the
    # stack pointer, or a literal).  The rotate brings that byte of the loaded
    # quadword to the front, so the scalar is at align(load addr) + slot.
    slot = _slot_of(ca)
    if slot is not None:
        return ld, None, slot
    return None


# ---------------------------------------------------------------------------
# passes
# ---------------------------------------------------------------------------


def recover_stores(func, strict=False):
    defs = _defmap(func)
    # Match everything first: rewriting can insert an address computation,
    # which would shift a block's instruction list mid-iteration.
    matches = []
    for insn in func.insns():
        if insn.op == Op.STOREQ:
            m = _match_store(insn, defs, strict=strict)
            if m is not None:
                matches.append((insn, m))

    n = assumed = 0
    for insn, m in matches:
        ctl, val, addr, hazards, slot = m
        if addr is None:
            addr = _offset_addr(func, insn, insn.srcs[1], slot)
        insn.op = Op.STORE
        insn.ew = ctl.ew
        insn.srcs = [insn.srcs[0], addr, _clone(val)]
        if hazards:
            assumed += 1
            where = ", ".join("0x%X" % (h.ea or 0) for h in hazards[:3])
            insn.comment = ("was cwd/lqd/shufb/stqd; assumes no alias with "
                            "the write at %s" % where)
        else:
            insn.comment = "was cwd/lqd/shufb/stqd"
        n += 1
    return n, assumed


def recover_loads(func, demand):
    defs = _defmap(func)
    matches = []
    for insn in func.insns():
        m = _match_load(insn, defs)
        if m is not None:
            matches.append((insn, m))

    n = 0
    for insn, (ld, addr, slot) in matches:
        if addr is None:
            addr = _offset_addr(func, insn, ld.srcs[1], slot)
        mask = demand.get(insn.dst.key(), ALL)
        ew = lanes.narrowest(mask, LEAD)
        insn.op = Op.LOAD if ew else Op.LOADU
        insn.ew = ew or EW.Q
        insn.srcs = [_clone(ld.srcs[0]), addr]
        insn.comment = "was lqd/rotqby"
        n += 1
    return n


def recover_aligned_loads(func, demand):
    """
    A `lqd`/`lqa` whose address is provably 16-aligned and whose result is only
    read in the preferred slot is already a scalar word load -- no rotate
    needed, because word 0 of the quadword *is* the value at that address.
    """
    defs = _defmap(func)
    n = 0
    for insn in func.insns():
        if insn.op != Op.LOADQ:
            continue
        mask = demand.get(insn.dst.key(), ALL)
        if mask == NONE or (mask & ~W0):
            continue
        a = addr_expr(insn.srcs[1], defs)
        if not a[2]:
            continue
        insn.op = Op.LOAD
        insn.ew = EW.W
        insn.comment = "aligned word load"
        n += 1
    return n


def mark_scalars(func, demand):
    """
    Flag operations whose result nothing reads outside the preferred slot, so
    they print as ordinary scalar arithmetic.

    Restricted to word-wide operations: that is what "the preferred slot"
    means.  A narrower element width with slot-0-only demand is still several
    packed values and is not a scalar.

    Note the SPU's comparisons produce all-ones/all-zeros *masks*, not 0/1, so
    a scalar-printed `==` yields 0xFFFFFFFF when true.
    """
    n = 0
    for insn in func.insns():
        d = insn.defines()
        if d is None or not lanes.is_scalar(demand.get(d.key(), NONE)):
            continue
        if insn.op == Op.CONST or insn.op == Op.MOV:
            insn.scalar = True
            n += 1
        elif insn.op in INFIX and insn.ew == EW.W and len(insn.srcs) == 2:
            insn.scalar = True
            n += 1
    return n


def scalarize(func, strict=False):
    """
    Run the whole scalarisation pipeline.  Returns a stats dict.

    Stores are recovered first: doing so deletes the `loadq` feeding each
    `shufb`, which sharpens the demand masks that load recovery and scalar
    marking then depend on.

    ``strict=True`` refuses any read-modify-write whose intervening writes
    cannot be proved disjoint.  The default trusts the compiler (such a
    sequence would be discarding the other write, which is a miscompile if
    they really alias) and annotates each assumption in the instruction
    comment rather than making it silently.
    """
    stats = {"stores": 0, "loads": 0, "aligned_loads": 0, "scalars": 0,
             "assumed_noalias": 0}

    stats["stores"], stats["assumed_noalias"] = recover_stores(
        func, strict=strict)

    demand = lanes.compute_demand(func)
    stats["loads"] = recover_loads(func, demand)

    demand = lanes.compute_demand(func)
    stats["aligned_loads"] = recover_aligned_loads(func, demand)

    demand = lanes.compute_demand(func)
    stats["scalars"] = mark_scalars(func, demand)
    func.demand = demand
    return stats

