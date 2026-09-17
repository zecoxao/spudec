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
from spudec import regs, ssa, opt, structure, cgen, data     # noqa: E402


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
    they arrive with a value, so there is nothing to assign.
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
        if "live in" in note:
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
