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


def _same_base(k1, k2, defs=None, depth=0):
    if k1 == k2:
        return True
    if k1 is None or k2 is None or defs is None or depth > 4:
        return False
    while k1 in defs and defs[k1].op == Op.MOV and defs[k1].srcs[0].is_var:
        k1 = defs[k1].srcs[0].key()
    while k2 in defs and defs[k2].op == Op.MOV and defs[k2].srcs[0].is_var:
        k2 = defs[k2].srcs[0].key()
    if k1 == k2:
        return True
    d1 = defs.get(k1)
    d2 = defs.get(k2)
    if d1 is None or d2 is None:
        return False
    if d1.op == d2.op and d1.ew == d2.ew and len(d1.srcs) == len(d2.srcs):
        if d1.op == Op.ADD:
            a1, b1 = d1.srcs
            a2, b2 = d2.srcs
            def match_src(x, y):
                if x.is_var and y.is_var:
                    return _same_base(x.key(), y.key(), defs, depth + 1)
                elif x.is_const and y.is_const:
                    return x.val == y.val
                return False
            if (match_src(a1, a2) and match_src(b1, b2)) or (match_src(a1, b2) and match_src(b1, a2)):
                return True
    return False


def _same_target(a, b, defs=None):
    """Two address expressions naming the same byte address."""
    return _same_base(a[0], b[0], defs) and a[1] == b[1]


def _same_slot(a, b, defs=None):
    """
    Same base, and the same byte position within a quadword.

    ``cwd``/``cbd`` only consume ``addr & 0xF``, so a compiler is free to
    generate the control mask from a cheaper but congruent address: real code
    pairs ``lqd r40, 0x400(r33)`` with ``cbd r41, 0(r33)`` because 0x400 is a
    multiple of 16 and the byte position is therefore identical.  Requiring
    the two offsets to be *equal* misses every one of those.
    """
    if _same_base(a[0], b[0], defs) and (a[1] & 0xF) == (b[1] & 0xF):
        return True
    if defs is not None and a[0] is not None and b[0] is not None:
        d1 = defs.get(a[0])
        d2 = defs.get(b[0])
        if d1 is not None and d2 is not None:
            if d1.op == Op.ADD and d2.op == Op.ADD and d1.ew == EW.W and d2.ew == EW.W:
                s1_0, s1_1 = d1.srcs
                s2_0, s2_1 = d2.srcs
                for idx1, base1 in ((s1_0, s1_1), (s1_1, s1_0)):
                    for idx2, base2 in ((s2_0, s2_1), (s2_1, s2_0)):
                        if idx1.is_var and idx2.is_var and _same_base(idx1.key(), idx2.key(), defs):
                            if (base1.is_var and base1.reg == regs.SP) or (base2.is_var and base2.reg == regs.SP):
                                if (a[1] & 0xF) == (b[1] & 0xF):
                                    return True
            if d1.op == Op.ADD and d1.ew == EW.W:
                for idx, base in ((d1.srcs[0], d1.srcs[1]), (d1.srcs[1], d1.srcs[0])):
                    if idx.is_var and _same_base(idx.key(), b[0], defs):
                        if base.is_var and base.reg == regs.SP:
                            if (a[1] & 0xF) == (b[1] & 0xF):
                                return True
            if d2.op == Op.ADD and d2.ew == EW.W:
                for idx, base in ((d2.srcs[0], d2.srcs[1]), (d2.srcs[1], d2.srcs[0])):
                    if idx.is_var and _same_base(idx.key(), a[0], defs):
                        if base.is_var and base.reg == regs.SP:
                            if (a[1] & 0xF) == (b[1] & 0xF):
                                return True
    return False


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


def _disjoint(a, b, defs=None):
    """
    Provably different quadwords.

    Only decidable with a common base: two addresses ``base+i`` and ``base+j``
    with ``|i-j| >= 16`` always fall in different quadwords, whatever ``base``
    is.  Different or unknown bases are not provable and must be assumed to
    alias.
    """
    if not _same_base(a[0], b[0], defs):
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


def _offset_addr(func, insn, base_val, slot, defs=None):
    """
    Materialise ``base + slot`` just before ``insn`` and return a Var for it.

    Used when the scalar sits at a non-zero byte position in its quadword: the
    hardware reads ``align(addr) + slot``, so that is the address to show --
    and it is exact, not an assumption about the base's alignment.
    """
    base = unaligned_of(base_val, defs) if defs else base_val
    if slot == 0:
        return _clone(base)
    r = _fresh_reg(func)
    add = Insn(Op.ADD, Var(r, 1), [_clone(base),
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
_MEM_DEFS = frozenset((Op.STOREQ, Op.STORE, Op.SYNC, Op.CALL, Op.ICALL, Op.TCALL))


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
        if d is None:
            return None
        if d.op == Op.PHI:
            if cur.key() == store_mem.key():
                return hazards
            return None
        if d.op not in _MEM_DEFS:
            return None
        if d.op in (Op.STOREQ, Op.STORE):
            if not _disjoint(addr_expr(d.srcs[1], defs), target, defs):
                hazards.append(d)
        else:
            hazards.append(d)                # a call may write anything
        cur = d.srcs[0]
    return None


# ---------------------------------------------------------------------------
# idiom matching
# ---------------------------------------------------------------------------


def _unwrap_shufb_chain(val, defs, limit=32):
    """
    Unwrap a chain of shufb operations:
    shufb(val_N, ... shufb(val_1, base, ctl_1) ..., ctl_N)
    Returns (chain, base) where chain is [(shuf_insn, val_i, ctl_insn), ...]
    ordered from first applied to last applied (shuf_1 to shuf_N).
    """
    chain = []
    cur = val
    for _ in range(limit):
        if not cur.is_var:
            break
        shuf = defs.get(cur.key())
        if shuf is None or shuf.op != Op.SHUFB or len(shuf.srcs) != 3:
            break
        a, b, c = shuf.srcs
        if not c.is_var:
            break
        ctl = defs.get(c.key())
        if ctl is None or ctl.op != Op.GENCTL:
            break
        chain.append((shuf, a, ctl))
        cur = b
    if not chain:
        return None, None
    chain.reverse()
    return chain, cur


def _match_store(insn, defs, strict=False):
    """
    ``storeq(mem, addr, shufb(val, loadq(mem', addr), genctl(addr)))``
    -> ``(items, ld, hazards, shufs)``
    where items is ``[(ctl, val_operand, effective_addr, slot), ...]``.
    """
    mem, addr, val = insn.srcs
    if not val.is_var:
        return None
    chain, base = _unwrap_shufb_chain(val, defs)
    if not chain or base is None:
        return None
    ld = defs.get(base.key()) if base.is_var else None
    is_loadq = (ld is not None and ld.op == Op.LOADQ)
    is_const0 = (base.is_const and base.val == 0) or (ld is not None and ld.op == Op.CONST and ld.srcs[0].val == 0)
    if not (is_loadq or is_const0):
        return None

    sa = addr_expr(addr, defs)
    if is_loadq:
        la = addr_expr(ld.srcs[1], defs)
        if not (sa[2] and la[2]):
            return None                      # both must be the aligned form
        if not _same_target(sa, la, defs):
            return None
    else:
        if not sa[2]:
            return None

    items = []
    shufs = []
    for shuf, a, ctl in chain:
        shufs.append(shuf)
        ca = addr_expr(ctl.srcs[0], defs)
        slot = None
        if _same_slot(sa, ca, defs):
            eff = _effective_addr(addr, sa, ca, defs)
        else:
            slot = _slot_of(ca)
            if slot is None:
                return None
            eff = None
        items.append((ctl, a, eff, slot))

    # The read-modify-write reads one memory state and writes back onto
    # another.  If anything wrote the same quadword in between, folding the
    # merge away changes what survives -- the machine code would be discarding
    # that write, which a compiler only emits when it knows they cannot alias.
    hazards = _chain_hazards(mem, ld.srcs[0], defs, sa) if is_loadq else []
    if hazards is None:
        return None
    if hazards and strict:
        return None
    return items, (ld if is_loadq else None), hazards, shufs


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
    if _same_slot(la, ca, defs):
        return ld, _effective_addr(ld.srcs[1], la, ca, defs), None

    # Absolute form: `lqa` at an aligned literal plus `rotqbyi` by the byte
    # offset within that quadword -- two literals naming one address.
    if la[0] is None and ca[0] is None:
        return (ld,
                Const((((la[1] & ~0xF) | (ca[1] & 0xF)) & 0xFFFFFFFF) << 96),
                None)

    # If both expressions share a base register and the offset difference falls
    # within the loaded quadword, cnt is the exact byte address of the scalar.
    if _same_base(la[0], ca[0], defs):
        d = (ca[1] - la[1]) & 0xFFFFFFFF
        if d < 16 or d >= 0xFFFFFFF0:
            if ca[0] is None:
                return ld, Const((ca[1] & 0xFFFFFFFF) << 96), None
            return ld, _clone(unaligned_of(cnt, defs)), None

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


def hoist_phi_shufb(func):
    """
    When multiple branches each compute a value to insert into the same quadword
    using the same element width and control mask (e.g. word store in if/else),
    the values merge in a PHI that feeds a single store:
        r2#phi = phi(shufb(v1, base, ctl), shufb(v2, base, ctl), ...)
        storeq mem, addr, r2#phi
    Hoist the PHI to the scalar values so the store can be scalarized:
        v#phi = phi(v1, v2, ...)
        r2#phi = shufb(v#phi, base, ctl)
    """
    defs = _defmap(func)
    max_ver = {}
    for i in func.insns():
        for v in [i.dst] + i.srcs:
            if v and v.is_var:
                max_ver[v.reg] = max(max_ver.get(v.reg, 0), v.ver)

    changed = False
    for b in func.blocks:
        for insn in b.insns:
            if insn.op == Op.PHI and insn.dst and insn.dst.reg != 128:
                shufs = []
                for s in insn.srcs:
                    if s.is_var and s.key() in defs:
                        d = defs[s.key()]
                        if d.op == Op.SHUFB and len(d.srcs) == 3:
                            shufs.append(d)
                if len(shufs) == len(insn.srcs) and len(shufs) >= 2:
                    ctl0 = defs.get(shufs[0].srcs[2].key()) if shufs[0].srcs[2].is_var else None
                    if not ctl0 or ctl0.op != Op.GENCTL:
                        continue
                    ca0 = addr_expr(ctl0.srcs[0], defs)
                    slot0 = _slot_of(ca0)
                    compatible = True
                    for sh in shufs[1:]:
                        ctli = defs.get(sh.srcs[2].key()) if sh.srcs[2].is_var else None
                        if not ctli or ctli.op != Op.GENCTL or ctli.ew != ctl0.ew:
                            compatible = False
                            break
                        cai = addr_expr(ctli.srcs[0], defs)
                        sloti = _slot_of(cai)
                        if slot0 is not None and sloti is not None:
                            if slot0 != sloti:
                                compatible = False
                                break
                        elif not _same_slot(ca0, cai, defs):
                            compatible = False
                            break
                    if not compatible:
                        continue

                    base0 = shufs[0].srcs[1]
                    ld0 = defs.get(base0.key()) if base0.is_var else None
                    if not ld0 or ld0.op != Op.LOADQ:
                        continue
                    la0 = addr_expr(ld0.srcs[1], defs)
                    for sh in shufs[1:]:
                        basei = sh.srcs[1]
                        ldi = defs.get(basei.key()) if basei.is_var else None
                        if not ldi or ldi.op != Op.LOADQ:
                            compatible = False
                            break
                        lai = addr_expr(ldi.srcs[1], defs)
                        if not _same_base(la0[0], lai[0], defs) or la0[1] != lai[1]:
                            compatible = False
                            break
                    if not compatible:
                        continue

                    target_reg = shufs[0].srcs[0].reg if shufs[0].srcs[0].is_var else (insn.dst.reg if insn.dst.reg > 2 else 100)
                    nv = max_ver.get(target_reg, 0) + 1
                    max_ver[target_reg] = nv
                    new_phi_dst = Var(target_reg, nv)
                    scalar_phi = Insn(Op.PHI, new_phi_dst, [sh.srcs[0] for sh in shufs], ea=insn.ea)
                    scalar_phi.block = b
                    b.insns.insert(b.insns.index(insn), scalar_phi)
                    insn.op = Op.SHUFB
                    insn.srcs = [new_phi_dst, shufs[0].srcs[1], shufs[0].srcs[2]]
                    changed = True
    return changed


def recover_stores(func, strict=False):
    hoist_phi_shufb(func)
    defs = _defmap(func)
    # Match everything first: rewriting can insert an address computation,
    # which would shift a block's instruction list mid-iteration.
    matches = []
    for insn in func.insns():
        if insn.op == Op.STOREQ:
            m = _match_store(insn, defs, strict=strict)
            if m is not None:
                matches.append((insn, m))

    # The values that fed the dismantled cwd/lqd/shufb: the inserted scalar,
    # the old-quadword reload and the insert mask.  Once the merge is a scalar
    # store the shufb is dead and DCE takes it, but the reload and the mask can
    # be pinned past DCE by a following call's conservative operand list -- a
    # constructor that immediately calls a method on the object it just built
    # is the usual case.  The renderer suppresses whatever of these is dead
    # (see cgen._compute_debris); recording the roots keeps that strictly
    # scoped to this idiom rather than to dead values in general.
    roots = getattr(func, "store_idiom_roots", None)
    if roots is None:
        roots = func.store_idiom_roots = set()

    max_mem = 0
    max_ver = {}
    for i in func.insns():
        for v in [i.dst] + i.srcs:
            if v is not None and v.is_var:
                if v.reg == 128:
                    max_mem = max(max_mem, v.ver)
                max_ver[v.reg] = max(max_ver.get(v.reg, 0), v.ver)

    n = assumed = 0
    for insn, (items, ld, hazards, shufs) in matches:
        orig_val = insn.srcs[2]
        base_addr = _clone(insn.srcs[1])
        other_users = [u for u in func.insns() if u is not insn and any(s.is_var and s.key() == orig_val.key() for s in u.srcs)]

        if ld:
            if ld.dst:
                roots.add(ld.dst.key())
            for s in ld.srcs:
                if s.is_var:
                    roots.add(s.key())
        for shuf in shufs:
            if shuf.dst:
                roots.add(shuf.dst.key())
            for s in shuf.srcs:
                if s.is_var:
                    roots.add(s.key())
        for ctl, a, eff, slot in items:
            if ctl.dst:
                roots.add(ctl.dst.key())
            for s in ctl.srcs:
                if s.is_var:
                    roots.add(s.key())

        cur_mem = insn.srcs[0]
        blk = insn.block
        for i, (ctl, val, eff, slot) in enumerate(items):
            if eff is None:
                eff = _offset_addr(func, insn, base_addr, slot, defs)
            is_last = (i == len(items) - 1)
            dst_mem = insn.dst if is_last else Var(128, max_mem + 1)
            if not is_last:
                max_mem += 1
                new_st = Insn(Op.STORE, dst_mem, [_clone(cur_mem), eff, _clone(val)],
                              ew=ctl.ew, ea=insn.ea)
                new_st.block = blk
                if hazards:
                    assumed += 1
                    where = ", ".join("0x%X" % (h.ea or 0) for h in hazards[:3])
                    new_st.comment = ("was cwd/lqd/shufb/stqd; assumes no alias with "
                                      "the write at %s" % where)
                else:
                    new_st.comment = "was cwd/lqd/shufb/stqd"
                blk.insns.insert(blk.insns.index(insn), new_st)
            else:
                insn.op = Op.STORE
                insn.ew = ctl.ew
                insn.srcs = [_clone(cur_mem), eff, _clone(val)]
                if hazards:
                    assumed += 1
                    where = ", ".join("0x%X" % (h.ea or 0) for h in hazards[:3])
                    insn.comment = ("was cwd/lqd/shufb/stqd; assumes no alias with "
                                    "the write at %s" % where)
                else:
                    insn.comment = "was cwd/lqd/shufb/stqd"
            cur_mem = dst_mem
            n += 1

        if other_users:
            nv = max_ver.get(orig_val.reg, 0) + 1
            max_ver[orig_val.reg] = nv
            reload_var = Var(orig_val.reg, nv)
            reload_insn = Insn(Op.LOADQ, reload_var, [cur_mem, base_addr], ea=insn.ea)
            reload_insn.block = blk
            st_idx = blk.insns.index(insn)
            blk.insns.insert(st_idx + 1, reload_insn)
            for u in other_users:
                for s_idx, s in enumerate(u.srcs):
                    if s.is_var and s.key() == orig_val.key():
                        u.srcs[s_idx] = reload_var
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


def propagate_scalar_slots(func, demand):
    """
    For a shufb(val, old, genctl) where genctl inserts into slot k >= 4, word 0 is old.
    If slot == 0 and ew == EW.W, word 0 is val.
    Consumers that only read word 0 (preferred slot) can read that source directly,
    freeing the shufb and genctl to be eliminated as dead debris.
    """
    defs = _defmap(func)
    n = 0
    for insn in func.insns():
        if insn.op != Op.SHUFB or len(insn.srcs) != 3 or insn.dst is None:
            continue
        val, old, ctl = insn.srcs
        if not (ctl.is_var and ctl.key() in defs):
            continue
        ctl_def = defs[ctl.key()]
        if ctl_def.op != Op.GENCTL:
            continue
        slot = _slot_of(addr_expr(ctl_def.srcs[0], defs))
        if slot is None:
            continue
        target = None
        if slot >= 4 and old.is_var:
            target = old
        elif slot == 0 and ctl_def.ew == EW.W:
            target = val
        if target is None:
            continue

        for user in func.insns():
            if user is insn:
                continue
            for idx, u in enumerate(user.srcs):
                if u.is_var and u.key() == insn.dst.key():
                    if (user.op in INFIX and user.ew == EW.W) or \
                       (user.op in (Op.ADD, Op.SUB) and user.ew == EW.W) or \
                       lanes.is_scalar(demand.get(insn.dst.key(), ALL)):
                        user.srcs[idx] = _clone(target)
                        n += 1
    return n


def _callee():
    """
    The cross-procedural demand hook, or None outside a database.

    Imported lazily and defensively: scalarisation runs during
    :func:`spudec.decompile`, which :func:`spudec.param_demand` itself calls,
    and a unit test may have no database at all.
    """
    try:
        from . import param_demand
    except ImportError:
        return None
    return param_demand


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

    demand = lanes.compute_demand(func, callee=_callee())
    stats["loads"] = recover_loads(func, demand)

    demand = lanes.compute_demand(func, callee=_callee())
    stats["aligned_loads"] = recover_aligned_loads(func, demand)

    demand = lanes.compute_demand(func, callee=_callee())
    propagate_scalar_slots(func, demand)

    demand = lanes.compute_demand(func, callee=_callee())
    stats["scalars"] = mark_scalars(func, demand)
    func.demand = demand
    return stats

