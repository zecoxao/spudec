"""
IDA viewer for the SSA IR.

Double-click (or Enter) on a line jumps the disassembly to the address that
line was lifted from, so the IR stays anchored to what IDA shows.
"""

import ida_kernwin
import ida_lines
import idaapi

from .ir import Op


def _c(text, color):
    return ida_lines.COLSTR(text, color)


class IRViewer(ida_kernwin.simplecustviewer_t):
    """
    Shows the structured pseudocode by default; `i` toggles to the SSA IR it
    was rendered from, so any line of pseudocode can be checked against the
    three-address form and, from there, against the disassembly.
    """

    def __init__(self):
        super(IRViewer, self).__init__()
        self.line_ea = []
        self.func = None
        self.mode = "c"

    # -- construction ------------------------------------------------------

    def build(self, func, mode="c"):
        title = "SPU decompile - %s" % (func.name or "sub_%X" % func.start_ea)
        if not self.Create(title):
            return False
        self.func = func
        self.mode = mode
        self.refresh()
        return True

    def refresh(self):
        self.ClearLines()
        self.line_ea = []
        if self.mode == "c":
            self._render_c(self.func)
        else:
            self._render(self.func)
        self.Refresh()

    def _render_c(self, func):
        import spudec
        self._add(_c("; press i for the SSA IR this was rendered from",
                     ida_lines.SCOLOR_AUTOCMT))
        try:
            import ida_name
            lines = spudec.pseudocode(
                func, name_of=lambda ea: ida_name.get_name(ea) or
                ("sub_%X" % ea))
        except Exception:
            import traceback
            for ln in traceback.format_exc().splitlines():
                self._add(_c("; " + ln, ida_lines.SCOLOR_ERROR))
            return
        for ln in lines:
            ea = None
            if ln.startswith("//"):
                self._add(_c(ln, ida_lines.SCOLOR_AUTOCMT), func.start_ea)
                continue
            self._add(_colour_c(ln), ea)

    def _add(self, text, ea=None):
        self.AddLine(text)
        self.line_ea.append(ea)

    def _render(self, func):
        name = func.name or "sub_%X" % func.start_ea
        self._add(_c("; SPU SSA IR  --  %s  @ 0x%X" % (name, func.start_ea),
                     ida_lines.SCOLOR_AUTOCMT), func.start_ea)

        stats = getattr(func, "stats", {}) or {}
        if stats:
            self._add(_c("; opt: %d rounds, %d instructions eliminated"
                         % (stats.get("rounds", 0), stats.get("removed", 0)),
                         ida_lines.SCOLOR_AUTOCMT))

        sc = getattr(func, "scalar_stats", {}) or {}
        if sc:
            line = ("; scalarised: %d scalar stores, %d scalar loads, "
                    "%d aligned loads, %d scalar ops"
                    % (sc.get("stores", 0), sc.get("loads", 0),
                       sc.get("aligned_loads", 0), sc.get("scalars", 0)))
            self._add(_c(line, ida_lines.SCOLOR_AUTOCMT))
            if sc.get("assumed_noalias"):
                self._add(_c("; %d store(s) recovered assuming no aliasing -- "
                             "see the per-instruction comments"
                             % sc["assumed_noalias"],
                             ida_lines.SCOLOR_ERROR))

        unhandled = getattr(func, "unhandled", {}) or {}
        if unhandled:
            worst = sorted(unhandled.items(), key=lambda kv: -kv[1])
            self._add(_c("; unmodelled (lifted as intrinsics): %s"
                         % ", ".join("%s x%d" % (k, v) for k, v in worst),
                         ida_lines.SCOLOR_ERROR))

        problems = getattr(func, "problems", []) or []
        for p in problems:
            self._add(_c("; SSA PROBLEM: " + p, ida_lines.SCOLOR_ERROR))

        for b in func.blocks:
            self._add("")
            preds = ", ".join("B%d" % p.id for p in b.preds) or "-"
            succs = ", ".join("B%d" % s.id for s in b.succs) or "-"
            self._add(
                _c("B%d:" % b.id, ida_lines.SCOLOR_CREFTAIL) +
                _c("  ; 0x%X..0x%X  preds: %s  succs: %s"
                   % (b.start_ea, b.end_ea, preds, succs),
                   ida_lines.SCOLOR_AUTOCMT),
                b.start_ea)
            for insn in b.insns:
                self._add("    " + self._fmt(insn), insn.ea)

    def _fmt(self, insn):
        text = str(insn)
        if insn.op == Op.PHI:
            return _c(text, ida_lines.SCOLOR_MACRO)
        if insn.op.is_terminator:
            return _c(text, ida_lines.SCOLOR_KEYWORD)
        if insn.op in (Op.LOADQ, Op.STOREQ, Op.LOAD, Op.STORE, Op.LOADU):
            return _c(text, ida_lines.SCOLOR_DREF)
        if insn.op == Op.INTRINSIC:
            return _c(text, ida_lines.SCOLOR_ERROR)
        return text

    # -- navigation --------------------------------------------------------

    def _jump(self):
        n = self.GetLineNo()
        if n is None or n >= len(self.line_ea):
            return False
        ea = self.line_ea[n]
        if ea is None:
            return False
        idaapi.jumpto(ea)
        return True

    def OnDblClick(self, shift):
        return self._jump()

    def OnKeydown(self, vkey, shift):
        if vkey == 13:                    # Enter
            return self._jump()
        if vkey == 27:                    # Esc
            self.Close()
            return True
        if vkey == ord("I"):              # toggle pseudocode / SSA IR
            self.mode = "ir" if self.mode == "c" else "c"
            self.refresh()
            return True
        return False


_C_KEYWORDS = ("if ", "else", "while ", "do", "for ", "return", "break;",
               "continue;", "goto ")


def _colour_c(line):
    stripped = line.strip()
    if stripped.startswith("//"):
        return _c(line, ida_lines.SCOLOR_AUTOCMT)
    if stripped.startswith(_C_KEYWORDS) or stripped.startswith("} while"):
        return _c(line, ida_lines.SCOLOR_KEYWORD)
    if stripped.startswith("*(") or " = *(" in stripped:
        return _c(line, ida_lines.SCOLOR_DREF)
    if stripped.endswith(":"):
        return _c(line, ida_lines.SCOLOR_CREFTAIL)
    return line


def show(func, mode="c"):
    v = IRViewer()
    if not v.build(func, mode=mode):
        return None
    v.Show()
    return v
