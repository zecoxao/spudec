"""
Standalone tests for the IDA-free half of the pipeline.

Everything except ``cfg``, ``lifter`` and ``view`` runs without IDA -- SSA
construction, the optimiser, structuring, type recovery and rendering -- so
building the IR by hand here exercises most of the decompiler without opening
a database.

Run with::

    python tests/test_pipeline.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from spudec.ir import Op, EW, Const, Var, Insn, Function     # noqa: E402
from spudec import (regs, ssa, opt, structure, cgen, data, lanes,
                    outssa)     # noqa: E402


def word0(v):
    """A quadword whose preferred slot holds ``v``."""
    return (v & 0xFFFFFFFF) << 96


def build(name, blocks, edges):
    f = Function(blocks[0][0], name)
    for start, insns in blocks:
        b = f.new_block(start, start + 4 * max(len(insns), 1))
        for i in insns:
            b.add(i)
    for a, b in edges:
        f.blocks[a].succs.append(f.blocks[b])
        f.blocks[b].preds.append(f.blocks[a])
    return f


def render(f, str_of=None):
    ssa.to_ssa(f)
    problems = ssa.verify(f)
    assert not problems, "\n".join(problems)
    f.stats = opt.optimize(f)
    problems = ssa.verify(f)
    assert not problems, "after opt:\n" + "\n".join(problems)
    # As `spudec.decompile` does: leave SSA before rendering, so a web whose
    # members are live at once is split and its merges become real copies.
    outssa.lower_phis(f)
    problems = ssa.verify(f)
    assert not problems, "after out-of-ssa:\n" + "\n".join(problems)
    stmts, info = structure.structure(f)
    f.structure_info = info
    text = "\n".join(cgen.generate(f, stmts, info,
                                   name_of=lambda ea: "sub_%X" % ea,
                                   arity_of=lambda ea: None,
                                   str_of=str_of))
    audit(text)
    return text


def show(title, text):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(text)
    print()


def expect(text, *needles):
    for n in needles:
        assert n in text, "expected %r in:\n%s" % (n, text)


def abi_ret():
    """What a return hands back: memory, the channel chain and r3 onward."""
    return [Var(regs.R_MEM), Var(regs.R_CH)] + \
        [Var(r) for r in range(regs.ARG_FIRST, regs.ARG_FIRST + 4)]


def audit(text):
    """
    The declaration block and the body must agree about which names exist.

    A name the body mentions but never declares, or declares but never
    assigns, means the listing refers to a value it does not show being
    computed.  Both happened: the inlining depth cap used to print a bare name
    for a definition that had already been suppressed as a statement, and a
    muted clobber's name still appeared where a call's arguments were
    rendered.  Every rendered function in this file is checked, so a
    regression in either direction fails a test rather than quietly producing
    a listing that does not add up.

    Registers the function only reads are exempt when marked `// live in`:
    they arrive with a value, so there is nothing to assign.  Stack slots,
    marked `// stack`, are exempt for a different reason: they are storage,
    not values.  A slot whose address this function hands to a callee is
    filled by that callee, so `read_region_data(a1, &var_C0, ...)` followed
    by a read of `var_C0` is a correct listing with no assignment in it.
    """
    import re
    lines = text.splitlines()
    try:
        brace = next(i for i, l in enumerate(lines) if l == "{")
    except StopIteration:
        return
    sig = lines[brace - 1] if brace else ""
    params = set(re.findall(r"\ba\d+\b", sig))

    decl = re.compile(r"^    (?P<type>(?:unsigned |signed |const |vec_\w+ |"
                      r"char |short |int |long |float |double |bool |"
                      r"qword\w* |u8 |u16 |u32 |u64 |\w+ )+)\*?\s*"
                      r"(?P<names>[*\w, ]+);(?:\s*//\s*(?P<note>.*))?$")
    strlit = re.compile(r'"(?:[^"\\]|\\.)*"')
    annot = re.compile(r"<[^<>]*>")
    comment = re.compile(r"//.*$")
    regname = re.compile(r"\b(r\d+(?:_\d+)?|lr(?:_\d+)?|sp(?:_\d+)?|"
                         r"gp(?:_\d+)?|fp(?:_\d+)?|ra(?:_\d+)?)\b")

    names, code = {}, []
    for l in lines[brace + 1:]:
        m = decl.match(l)
        if m and " = " not in l and "(" not in l and "goto" not in l:
            note = m.group("note") or ""
            for n in m.group("names").split(","):
                n = n.strip().lstrip("*").strip()
                if re.fullmatch(r"[A-Za-z_]\w*", n):
                    names[n] = note
            continue
        code.append(annot.sub("", comment.sub("", strlit.sub('""', l))))
    code = "\n".join(code)

    for n, note in names.items():
        if "live in" in note or "stack" in note:
            continue
        assigned = re.search(r"(?:^|[^\w.>])\*?" + re.escape(n)
                             + r"\s*(?:=[^=]|\+\+|--)", code, re.M)
        assert assigned, "declared but never assigned: %s\n%s" % (n, text)

    for n in sorted(set(regname.findall(code))):
        assert n in names or n in params, \
            "used but never declared: %s\n%s" % (n, text)


# ---------------------------------------------------------------------------
# 1. a phi argument that is a constant must keep its assignment
# ---------------------------------------------------------------------------


def test_phi_constant_assignment():
    """
    Regression: constant propagation used to fold a constant into a phi
    argument, which killed the defining instruction and made DCE delete it --
    so the assignment vanished from the arm of the `if` that performed it and
    the listing silently dropped a store to a register.
    """
    R3, R4 = regs.ARG_FIRST, regs.ARG_FIRST + 1
    f = build("pick", [
        (0x100, [
            Insn(Op.CJMP, None, [Var(R4)], ea=0x100, aux=(0x120, "z")),
        ]),
        (0x104, [
            Insn(Op.CONST, Var(R3), [Const(word0(1))], ea=0x104, scalar=True),
            Insn(Op.JMP, None, [], ea=0x108, aux=0x130),
        ]),
        (0x120, [
            Insn(Op.CONST, Var(R3), [Const(word0(2))], ea=0x120, scalar=True),
            Insn(Op.JMP, None, [], ea=0x124, aux=0x130),
        ]),
        (0x130, [
            Insn(Op.RET, None, abi_ret(), ea=0x130),
        ]),
    ], [(0, 1), (0, 2), (1, 3), (2, 3)])

    text = render(f)
    show("1. both arms of a phi keep their assignment", text)
    # Both constants must survive; before the fix only the branch itself did.
    expect(text, "= 0x1;", "= 0x2;")


# ---------------------------------------------------------------------------
# 2. a constant that points at a string renders as one
# ---------------------------------------------------------------------------


def test_strings():
    """
    The resolver is a callback because answering it needs the database, so
    here it is a stub: 0x2C018 is a string, nothing else is.
    """
    def str_of(ea):
        return data.escape("INFO: %s\n") if ea == 0x2C018 else None

    R3, R4 = regs.ARG_FIRST, regs.ARG_FIRST + 1
    f = build("logger", [
        (0x200, [
            Insn(Op.CONST, Var(R3), [Const(word0(0x2C018))], ea=0x200,
                 scalar=True),
            # A store through the same constant must NOT print the literal.
            Insn(Op.STORE, Var(regs.R_MEM),
                 [Var(regs.R_MEM), Const(word0(0x2C018)), Var(R4)],
                 ew=EW.B, ea=0x204),
            Insn(Op.RET, None, abi_ret(), ea=0x208),
        ]),
    ], [])

    text = render(f, str_of=str_of)
    show("2. string constants", text)
    expect(text, '"INFO: %s\\n"')
    # Typed from the literal, so the declaration agrees with the assignment.
    expect(text, "char *")
    # ...but the store destination stays an address: writing through a string
    # literal would read as an assignment to a constant.
    expect(text, "*(u8 *)(0x2C018) =")

    # Escaping, and the sniffer's rejections.  The input holds a quote, a
    # backslash and a tab; all three have to come back out as C escapes.
    raw = 'a"b' + chr(92) + 'c' + chr(9) + 'd'
    assert data.escape(raw) == '"a\\"b\\\\c\\td"', data.escape(raw)
    assert data._looks_like_text(b"hello")
    assert not data._looks_like_text(b"ab"), "too short"
    assert not data._looks_like_text(b"    "), "no letter or digit"
    assert not data._looks_like_text(b"he\x01lo"), "not printable"
    # No database here, so the resolver itself must simply decline.
    assert data.Strings()(0x2C018) is None


# ---------------------------------------------------------------------------
# nothing may quietly bypass the demangling name resolver
# ---------------------------------------------------------------------------


def test_no_raw_name_of():
    """
    Regression: the viewer and the plugin each passed their own
    ``name_of=lambda ea: ida_name.get_name(ea) or "sub_%X" % ea``.  That is an
    *override*, so teaching the default resolver to demangle changed nothing
    for either of the two paths a user actually goes through -- the listing
    still showed ``_ZN2ss14ss_iso_dprintf5put_sEPKc(...)`` while the headless
    API showed ``ss::ss_iso_dprintf::put_s(...)``.

    A default that every real caller overrides is not a default, so this walks
    the source and fails if anything hands a bare ``get_name`` back as a name
    resolver again.
    """
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..")
    pattern = re.compile(r"name_of\s*=\s*(.*)")
    bad = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".git", "tests")]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            src = io.open(os.path.join(dirpath, fn), encoding="utf-8",
                          errors="replace").read()
            for m in pattern.finditer(src):
                rhs = m.group(1)
                if "get_name" in rhs:
                    bad.append("%s: name_of=%s" % (fn, rhs.strip()[:60]))
    assert not bad, ("these bypass the demangling resolver:\n  "
                     + "\n  ".join(bad))
    print("nothing bypasses the name resolver: ok")
    print()


def test_deep_inline_chain():
    """
    A chain longer than the inlining depth cap must still compute every value.

    ``operand`` only inlines while ``depth < MAX_INLINE_DEPTH``; past that it
    printed the definition's *name*, while ``insn_stmt`` had already dropped
    that definition as a statement because it was marked inlinable.  The
    listing then used a variable it never computed.  :func:`cgen._inlinable`
    now applies the cap when it picks candidates, so anything too deep stays
    a statement instead.

    Each link adds a different live-in register, so constant folding cannot
    collapse the chain and the cap really is reached.  ``audit`` inside
    :func:`render` is what checks the result; the assertion below states the
    shape so a silent change stays visible.
    """
    R3 = regs.ARG_FIRST
    n = 2 * cgen.MAX_INLINE_DEPTH
    t0 = regs.NREG
    insns = [Insn(Op.MOV, Var(t0), [Var(80)], ea=0x300, scalar=True)]
    for i in range(n):
        insns.append(Insn(Op.ADD, Var(t0 + i + 1),
                          [Var(t0 + i), Var(81 + i)],
                          ea=0x304 + 4 * i, scalar=True))
    insns.append(Insn(Op.MOV, Var(R3), [Var(t0 + n)], ea=0x400, scalar=True))
    insns.append(Insn(Op.RET, None, abi_ret(), ea=0x404))
    f = build("deep", [(0x300, insns)], [])

    text = render(f)
    show("deep inline chain (%d links, cap %d)"
         % (n, cgen.MAX_INLINE_DEPTH), text)
    body = [l.strip() for l in text.splitlines()
            if l.startswith("    ") and l.strip().endswith(";")]
    assert any(l.startswith("t") for l in body), (
        "expected a temporary to stay a statement:" + chr(10) + text)


def test_callee_demand_narrows_arguments():
    """
    A call argument's demand is what the callee reads, not all sixteen bytes.

    ``_contrib`` used to credit every operand of a call with ALL.  A call
    carries the whole argument register set so DCE cannot delete argument
    setup, so that one line made most values in a call-heavy function look
    like opaque quadwords -- and it fed back: an `a rX, sp, 0x20` computing an
    argument was credited ALL, which propagated into the stack pointer, so
    even plain address arithmetic stopped scalarising.

    Here the same function is analysed twice.  Without the hook the argument
    setup demands everything; with a callee that only reads the preferred slot
    of its first parameter, it narrows to the slot and the add becomes a
    scalar.  The conservative answer is still the default, which is what keeps
    an indirect call or a recursive cycle safe.
    """
    R3 = regs.ARG_FIRST
    t = regs.NREG
    call = Insn(Op.CALL, Var(regs.R_MEM),
                [Var(regs.R_MEM), Var(regs.R_CH), Var(regs.LR), Var(R3)],
                ea=0x108, aux=0x2000)
    f = build("caller", [
        (0x100, [
            Insn(Op.ADD, Var(R3), [Var(80), Const(word0(0x20))], ea=0x100),
            call,
            # Nothing but the call may read r3, or the return's own
            # conservative operand list would demand it whole and mask the
            # effect under test.
            Insn(Op.RET, None, [Var(regs.R_MEM), Var(regs.R_CH)], ea=0x10C),
        ]),
    ], [])
    ssa.to_ssa(f)

    add_key = next(i.defines().key() for i in f.insns() if i.op == Op.ADD)

    wide = lanes.compute_demand(f)
    assert wide[add_key] == lanes.ALL, (
        "without the hook a call argument must stay conservative, got 0x%04X"
        % wide[add_key])

    # A callee that only uses its first parameter as an address.
    def callee(ea):
        assert ea == 0x2000, ea
        return {R3: lanes.W0}

    narrow = lanes.compute_demand(f, callee=callee)
    assert narrow[add_key] == lanes.W0, (
        "the callee reads only the preferred slot, so the argument setup "
        "should too, got 0x%04X" % narrow[add_key])

    # The filler operands are classified by what they are, hook or no hook:
    # the link register a call reads is a return address, taken from the
    # preferred slot, never a quadword.
    for got in (wide.get((regs.LR, 0)), narrow.get((regs.LR, 0))):
        assert got == lanes.W0, \
            "the link register is a return address, got 0x%04X" % got
    print("cross-procedural demand narrows call arguments: ok")
    print()


# ---------------------------------------------------------------------------
# the calling convention belongs in the header, not in the body
# ---------------------------------------------------------------------------


def test_prologue_moves_to_the_header():
    """
    A callee-save store is stated in the header; a slot that is read back is
    not touched.

    Saving lr and r80..r127 and writing the back chain is a tenth of every
    line the corpus produces, and none of it says anything about what a
    function does.  What makes hiding it safe is the third store here: its
    slot is loaded from later, so it is carrying data whatever it looks like,
    and it must stay in the body.  The other two are stores of *live-in*
    callee-saved values, which at entry hold the caller's registers and so
    can only be being preserved.
    """
    from spudec import frame
    R3 = regs.ARG_FIRST
    SP, LR = regs.SP, regs.LR
    t = regs.NREG
    M = regs.R_MEM
    MASK = Const(word0(0x3FFF0))

    def slot(tmp, off):
        """`(sp + off) & ~0xF`, the address shape the lifter emits."""
        return [Insn(Op.ADD, Var(tmp), [Var(SP), Const(word0(off))],
                     ew=EW.W, ea=0x100),
                Insn(Op.AND, Var(tmp + 1), [Var(tmp), MASK], ew=EW.W,
                     ea=0x104)]

    insns = []
    insns += slot(t, -0x10)
    insns.append(Insn(Op.STOREQ, Var(M), [Var(M), Var(t + 1), Var(80)],
                      ew=EW.Q, ea=0x108))
    insns += slot(t + 2, 0x10)
    insns.append(Insn(Op.STOREQ, Var(M), [Var(M), Var(t + 3), Var(LR)],
                      ew=EW.Q, ea=0x10C))
    # A slot that is read back: data, not a save.
    insns += slot(t + 4, -0x30)
    insns.append(Insn(Op.STOREQ, Var(M), [Var(M), Var(t + 5), Var(81)],
                      ew=EW.Q, ea=0x110))
    insns += slot(t + 6, -0x30)
    insns.append(Insn(Op.LOADQ, Var(R3), [Var(M), Var(t + 7)], ew=EW.Q,
                      ea=0x114))
    insns.append(Insn(Op.RET, None, abi_ret(), ea=0x118))
    f = build("framed", [(0x100, insns)], [])

    ssa.to_ssa(f)
    fr = frame.analyse(f)
    assert sorted(r for _, r in fr.saves) == [LR, 80], \
        "expected lr and r80 to be recognised, got %r" % (fr.saves,)
    assert 81 not in [r for _, r in fr.saves], \
        "a slot that is read back is data, not a save"

    text = render(f)
    show("prologue stated in the header", text)
    expect(text, "// prologue: saves lr at sp+0x10, r80 at sp-0x10")
    assert "= r80;" not in text, "the r80 save should not be in the body"
    assert "= lr;" not in text, "the lr save should not be in the body"
    expect(text, "= r81;")          # ...but the one that is read back is

    # A register whose only appearance was the save must not be left
    # declared, or the listing declares a value it never mentions again.
    # `audit`, inside `render`, checks the general case; this states the one
    # the frame pass creates.
    body = text.split(chr(10) + "{" + chr(10), 1)[1]
    assert "r80" not in body, "r80 is still declared:" + chr(10) + text


# ---------------------------------------------------------------------------
# stack slots print as IDA's frame members
# ---------------------------------------------------------------------------


def test_stack_slots_print_as_names():
    """
    One slot, one name, whatever width each access uses.

    Identity comes from this decompiler -- ``addr_expr`` resolving an address
    to ``(base, offset)`` -- and the *name* comes from IDA, so a slot the user
    renamed shows the new name.  Grouping by IDA's name instead let one
    location print two ways, because a write through a form IDA had not
    labelled stayed an explicit dereference while the read became a name.

    Checked here: the widest access prints as the bare name; a narrower one
    casts through the slot's address rather than falling back to the raw
    expression; taking the address prints ``&var_30``; and the slot the
    prologue uses for the back chain is not named, since ``sp_2 = &var_20``
    would hide that sp_2 is the frame pointer.
    """
    R3 = regs.ARG_FIRST
    SP, M = regs.SP, regs.R_MEM
    t = regs.NREG
    MASK = Const(word0(0x3FFF0))

    def slot(tmp, off, base=SP, base_ver_of=None):
        return [Insn(Op.ADD, Var(tmp), [Var(base), Const(word0(off))],
                     ew=EW.W, ea=0x100),
                Insn(Op.AND, Var(tmp + 1), [Var(tmp), MASK], ew=EW.W,
                     ea=0x104)]

    insns = []
    # the back chain: sp is stored into sp-0x20, so -0x20 is the frame's own
    insns += slot(t, -0x20)
    insns.append(Insn(Op.STOREQ, Var(M), [Var(M), Var(t + 1), Var(SP)],
                      ew=EW.Q, ea=0x108))
    # a real slot at -0x30, written wide and read narrow
    insns += slot(t + 2, -0x30)
    insns.append(Insn(Op.STOREQ, Var(M), [Var(M), Var(t + 3), Var(R3)],
                      ew=EW.Q, ea=0x10C))
    insns += slot(t + 4, -0x30)
    insns.append(Insn(Op.LOAD, Var(R3 + 1), [Var(M), Var(t + 5)], ew=EW.W,
                      ea=0x110))
    # and its address, handed to a call -- in r3, the first argument
    # register, so the call really renders it as an argument
    insns.append(Insn(Op.ADD, Var(R3), [Var(SP), Const(word0(-0x30))],
                      ew=EW.W, ea=0x114))
    insns.append(Insn(Op.CALL, Var(M),
                      [Var(M), Var(regs.R_CH), Var(regs.LR), Var(R3)],
                      ea=0x118, aux=0x4000))
    insns.append(Insn(Op.RET, None, abi_ret(), ea=0x11C))
    f = build("slots", [(0x100, insns)], [])

    # The stub stands in for ida_frame: every access at these addresses is
    # the same frame member.  A real resolver answers per instruction.
    named = {0x10C: ("var_30", 16), 0x110: ("var_30", 16),
             0x108: ("var_20", 16)}

    ssa.to_ssa(f)
    problems = ssa.verify(f)
    assert not problems, chr(10).join(problems)
    f.stats = opt.optimize(f)
    stmts, info = structure.structure(f)
    f.structure_info = info
    text = chr(10).join(cgen.generate(
        f, stmts, info, name_of=lambda ea: "sub_%X" % ea,
        arity_of=lambda ea: 1, stk_of=lambda ea: named.get(ea)))
    audit(text)
    show("stack slots as frame members", text)

    expect(text,
           "var_30 = ",                      # the widest access, bare
           "*(u32 *)&var_30",                # a narrower one casts
           "= &var_30;",                     # computing its address
           "// stack")                       # declared, and marked as such
    assert "var_20" not in text, (
        "the back-chain slot is the frame's own, not a local:" + chr(10)
        + text)
    assert "0x3FFF0" not in text.split("{", 1)[1], (
        "every access to the slot should be named, none left explicit:"
        + chr(10) + text)


# ---------------------------------------------------------------------------
# one name never means two simultaneously live values
# ---------------------------------------------------------------------------


def test_interfering_web_is_split():
    """
    Coalescing a phi web is only sound when its members do not overlap.

    Found in sc_iso's `ss::sc_proxy_hdr::make_hdr`, where the IR said

        r9#19  = selb r3#22, r9#18, r11#16
        t60#1  = r9#1 & 0x3FFF0          <- the incoming pointer
        mem#20 = storeq mem#19, t60#1, r9#19

    and the listing said

        r9 = selb(..., *(qword *)(r9 & 0x3FFF0), ...);
        *(qword *)(r9 & 0x3FFF0) = r9;

    whose store address is the value assigned on the line above.  Not untidy
    -- wrong, and wrong in the way that is hardest to catch by reading.

    Here the phi takes the incoming r3 on one path, and the incoming r3 is
    still live at the phi because the store uses it as an address.  So the
    two cannot share a name, and the merge that used to be implicit has to
    become a real assignment on the edge that supplies it.
    """
    R3, R4, R5 = regs.ARG_FIRST, regs.ARG_FIRST + 1, regs.ARG_FIRST + 2
    M = regs.R_MEM
    # The address is kept in another register across the branch, exactly as
    # the compiler did it (`lr r12, r9`); copy propagation then folds r5 away
    # and the store addresses the incoming r3 directly, which is what makes
    # its range reach past the phi.
    f = build("split", [
        (0x100, [Insn(Op.MOV, Var(R5), [Var(R3)], ew=EW.Q, ea=0x100),
                 Insn(Op.CJMP, None, [Var(R4)], ea=0x102, aux=(0x108, "z"))]),
        (0x104, [Insn(Op.JMP, None, [], ea=0x104, aux=0x10C)]),
        (0x108, [Insn(Op.CONST, Var(R3), [Const(word0(2))], ea=0x108,
                      scalar=True),
                 Insn(Op.JMP, None, [], ea=0x10A, aux=0x10C)]),
        (0x10C, [
            Insn(Op.STOREQ, Var(M), [Var(M), Var(R5), Var(R3)], ew=EW.Q,
                 ea=0x10C),
            Insn(Op.RET, None, [Var(M), Var(regs.R_CH)], ea=0x110),
        ]),
    ], [(0, 1), (0, 2), (1, 3), (2, 3)])

    text = render(f)
    show("an interfering web is split", text)

    body = [l.strip() for l in text.split(chr(10) + "{" + chr(10), 1)[1]
            .splitlines() if l.strip().endswith(";")]
    store = [l for l in body if l.startswith("*")]
    assert store, "expected a store:" + chr(10) + text
    lhs, rhs = store[0].split(" = ", 1)
    assert "a1" in lhs, (
        "the address should still be the incoming r3:" + chr(10) + text)
    assert "a1" not in rhs, (
        "the stored value is the merged one, not the parameter:"
        + chr(10) + text)
    # ...and the path that used to supply the incoming value implicitly now
    # says so, in an arm the structurer had elided while it was empty.
    assert "r3 = a1;" in text, (
        "the split web needs its merge spelled out:" + chr(10) + text)


# ---------------------------------------------------------------------------
# every mnemonic IDA decodes must have a real semantic, not an intrinsic
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

# Where IDA's SPU processor module might live.  Absence only costs the
# cross-check; the coverage test itself runs from the checked-in list.
SPU_PY_CANDIDATES = (
    os.environ.get("SPU_PROC_PY"),
    os.path.join(os.environ.get("IDADIR", ""), "procs", "spu.py"),
    r"C:\ida94b1\procs\spu.py",
    r"C:\Program Files\IDA Professional 9.4\procs\spu.py",
)


def _listed_mnemonics():
    """The mnemonics recorded in ``tests/spu_mnemonics.txt``."""
    out = []
    for line in io.open(os.path.join(HERE, "spu_mnemonics.txt"),
                        encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _spu_py_mnemonics():
    """The same, read out of spu.py, or ``None`` if no install was found."""
    import re
    for path in SPU_PY_CANDIDATES:
        if not path or not os.path.exists(path):
            continue
        src = io.open(path, encoding="utf-8", errors="replace").read()
        # Two ways spu.py names an instruction: the itable_* constructors and
        # the few appended to Instructions by hand (`lr`, the ori shorthand).
        found = re.findall(r'idef_\w*\(\s*"([a-z0-9.]+)"', src)
        found += re.findall(
            r"""Instructions\.append\(\{\s*'name'\s*:\s*["']([a-z0-9.]+)["']""",
            src)
        return sorted(set(found)), path
    return None, None


def _covered_mnemonics():
    """
    Every mnemonic ``lifter.Lifter.lift`` dispatches without falling through.

    Read from the source rather than by importing: lifter.py imports IDA, and
    this file deliberately runs without a database.  ``lift`` resolves a
    mnemonic three ways -- a ``_i_<name>`` method, a class-body alias like
    ``_i_lqr = _i_lqa`` where one semantic serves several mnemonics, or
    membership of one of the ``self.<group>`` tables built in ``__init__`` --
    so all three are collected.  Leaving the aliases out reported `bra`,
    `brasl`, `lqr` and `stqr` as uncovered when they are handled.
    """
    import ast
    src = io.open(os.path.join(HERE, "..", "spudec", "lifter.py"),
                  encoding="utf-8").read()
    tree = ast.parse(src)
    covered = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name.startswith("_i_"):
            covered.add(node.name[3:])
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id.startswith("_i_"):
                covered.add(t.id[3:])
        if not any(isinstance(t, ast.Attribute)
                   and isinstance(t.value, ast.Name)
                   and t.value.id == "self" for t in node.targets):
            continue
        v = node.value
        # A dict's *keys* are mnemonics; its values are Op/EW enums.  A
        # frozenset/tuple/list holds mnemonics directly.
        items = []
        if isinstance(v, ast.Dict):
            items = v.keys
        elif isinstance(v, (ast.Set, ast.Tuple, ast.List)):
            items = v.elts
        elif isinstance(v, ast.Call) and v.args and isinstance(
                v.args[0], (ast.Set, ast.Tuple, ast.List)):
            items = v.args[0].elts          # frozenset((...))
        for k in items:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                covered.add(k.value)
    return covered


def test_every_mnemonic_has_semantics():
    """
    Anything without a hand-written semantic becomes an ``INTRINSIC``: sound,
    since its defs and uses stay right, but opaque -- the listing shows
    ``intr dftsv(...)`` and type recovery learns nothing.  That is invisible
    in the corpus numbers, because a mnemonic no PS3 module happens to use
    reports zero unmodelled hits while still being unmodelled.  `bisled`,
    `fscrrd`, `fscrwr` and `dftsv` sat that way unnoticed.

    So the gate is the decoder's whole instruction set, not the corpus: every
    mnemonic spu.py decodes must resolve to a handler or a dispatch table.
    """
    listed = _listed_mnemonics()
    covered = _covered_mnemonics()

    # `lift` looks up `_i_` + name.replace(".", "_"), so a dotted mnemonic is
    # covered by the underscored method name.
    missing = [m for m in listed
               if m not in covered and m.replace(".", "_") not in covered]
    assert not missing, (
        "%d of %d mnemonics fall through to an intrinsic: %s"
        % (len(missing), len(listed), ", ".join(missing)))

    # And the list itself must not drift away from the module it describes.
    actual, path = _spu_py_mnemonics()
    if actual is None:
        note = "list not cross-checked (no IDA install found)"
    else:
        stale = sorted(set(actual) - set(listed))
        gone = sorted(set(listed) - set(actual))
        assert not stale and not gone, (
            "tests/spu_mnemonics.txt disagrees with %s:%s%s%s"
            % (path,
               chr(10) + "  only in spu.py: " + ", ".join(stale) if stale
               else "",
               chr(10) + "  only in the list: " + ", ".join(gone) if gone
               else "",
               chr(10) + "regenerate the list if spu.py really changed."))
        note = "cross-checked against " + os.path.basename(path)
    print("all %d SPU mnemonics have semantics (%s)" % (len(listed), note))
    print()


def main():
    """
    Run every ``test_*`` in this file, in definition order.

    Listing them by hand drifted: `test_strings` and `test_deep_inline_chain`
    were both defined and never called, so they passed by not running.
    """
    import inspect
    mod = sys.modules[__name__]
    tests = [(n, f) for n, f in vars(mod).items()
             if n.startswith("test_") and inspect.isfunction(f)]
    tests.sort(key=lambda nf: nf[1].__code__.co_firstlineno)
    for name, fn in tests:
        fn()
    print("all %d tests passed" % len(tests))


if __name__ == "__main__":
    main()
