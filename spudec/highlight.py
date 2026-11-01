"""
C syntax highlighting for the pseudocode viewers.

Token-level rather than line-level: the previous version coloured a whole line
by whatever it started with, which meant a line like

    r4 = *(unsigned int *)(r5 + 0x10);   // was lqd/rotqby

came out in one flat colour and the reader got nothing from it.

Colours use IDA's own `SCOLOR_*` tags rather than fixed RGB, so the output
follows whatever theme the user has set -- a dark theme does not end up with
black-on-black.

No IDA imports at module level beyond `ida_lines`, and nothing here depends on
a database, so it can be exercised from a plain script.
"""

import re

import ida_lines

C_KEYWORDS = frozenset((
    "if", "else", "while", "do", "for", "return", "break", "continue",
    "goto", "switch", "case", "default", "sizeof",
))

C_TYPES = frozenset((
    "void", "char", "short", "int", "long", "unsigned", "signed",
    "float", "double", "const", "struct", "union", "enum",
    # what this decompiler emits
    "qword", "qword_unaligned", "u8", "u16", "u32", "u64",
    "s8", "s16", "s32", "s64", "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "uintptr_t", "intptr_t",
))

# Tokens, in priority order.  `placeholder` deliberately requires a letter
# right after the '<' so that a shift operator (`a << b >> c`) cannot be
# swallowed as one.
_TOK = re.compile(r"""
      (?P<comment>//.*)
    | (?P<placeholder><[A-Za-z][^<>]*>)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<hex>-?\b0[xX][0-9A-Fa-f]+)
    | (?P<num>-?\b\d+\b)
    | (?P<ident>[A-Za-z_$][A-Za-z_0-9$]*(?:\.[A-Za-z0-9]+)?)
    | (?P<ws>\s+)
    | (?P<op>.)
""", re.VERBOSE)

_CALL_AHEAD = re.compile(r"\s*\(")


def _tag(text, color):
    return ida_lines.COLSTR(text, color)


def colorize(line, highlight_ident=None):
    """Return ``line`` with IDA colour tags around each token.
    
    If ``highlight_ident`` is specified, occurrences of that identifier
    are given an accent highlight tag (SCOLOR_EXTRA) to emphasize matching tokens.
    """
    if not line:
        return line

    stripped = line.lstrip()
    # Whole-line comments are the common case; skip the tokenizer if no token highlight.
    if stripped.startswith("//") and not highlight_ident:
        return _tag(line, ida_lines.SCOLOR_AUTOCMT)

    out = []
    pos = 0
    n = len(line)
    while pos < n:
        m = _TOK.match(line, pos)
        if m is None:                       # cannot happen, `op` matches any
            out.append(line[pos])
            pos += 1
            continue
        kind = m.lastgroup
        text = m.group()
        pos = m.end()

        if kind == "ws":
            out.append(text)
        elif kind == "comment":
            out.append(_tag(text, ida_lines.SCOLOR_AUTOCMT))
        elif kind == "placeholder":
            # `<clobbered by call>`, `<result in r3>` -- our own annotations,
            # not C, so they read better as markup than as identifiers.
            out.append(_tag(text, ida_lines.SCOLOR_MACRO))
        elif kind == "string":
            out.append(_tag(text, ida_lines.SCOLOR_STRING))
        elif kind in ("hex", "num"):
            out.append(_tag(text, ida_lines.SCOLOR_NUMBER))
        elif kind == "ident":
            if highlight_ident and (text == highlight_ident or text.split(".")[0] == highlight_ident):
                out.append(_tag(text, ida_lines.SCOLOR_EXTRA))
            else:
                out.append(_tag(text, _ident_color(text, line, pos)))
        else:
            out.append(_tag(text, ida_lines.SCOLOR_SYMBOL))
    return "".join(out)


def _ident_color(text, line, after):
    base = text.split(".", 1)[0]
    if base in C_KEYWORDS:
        return ida_lines.SCOLOR_KEYWORD
    if base in C_TYPES or base.startswith("vec_"):
        return ida_lines.SCOLOR_TYPE
    if base.startswith("loc_") or base.startswith("def_") \
            or base.startswith("jpt_") or base.startswith("sub_"):
        return ida_lines.SCOLOR_CREFTAIL
    # An identifier followed by '(' is being called.  This also picks up the
    # vector intrinsics (`shufb(`, `add.w(`), which is what we want: they are
    # operations, not variables.
    if _CALL_AHEAD.match(line, after):
        return ida_lines.SCOLOR_CNAME
    return ida_lines.SCOLOR_REG


def colorize_all(lines):
    return [colorize(ln) for ln in lines]
