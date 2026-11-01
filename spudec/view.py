"""
Interactive IDA custom viewer for spudec pseudocode.

Provides:
- Right-click context menu and hotkeys for:
  - Renaming variables and functions ('N') with IDB netnode persistence.
  - Setting variable and function types ('Y') with IDA type system integration.
  - Linking / synchronizing with IDA's disassembly view (follows cursor across functions).
  - Jumping to disassembly and clicking through to callee functions ('Enter' / 'Space' / double-click).
  - Refreshing decompilation ('F5').
  - Toggling between C pseudocode and SSA IR ('I').
- Reusable dockable window that updates in place during navigation.
"""

import re
import ida_kernwin
import ida_funcs
import ida_lines
import ida_typeinf
import idaapi

from .ir import Op
from . import highlight, db
from . import regs

_EA = re.compile(r"\b(?:loc_|sub_)([0-9A-Fa-f]+)\b")
_ADDR_HEX = re.compile(r"\b0x([0-9A-Fa-f]{3,8})\b")
_VIEW_TITLE = "spudec - Pseudocode"

ACTION_RENAME = "spudec:rename_var"
ACTION_SET_TYPE = "spudec:set_type"
ACTION_TOGGLE_SYNC = "spudec:toggle_sync"
ACTION_JUMP_DISASM = "spudec:jump_disasm"
ACTION_REFRESH = "spudec:refresh"
ACTION_TOGGLE_IR = "spudec:toggle_ir"

_active_viewer = None
_ui_hooks = None
_view_hooks = None


def _is_dark_theme():
    """Detect if IDA is currently using a dark theme."""
    try:
        from PySide6 import QtWidgets, QtGui
        app = QtWidgets.QApplication.instance()
        if app:
            pal = app.palette()
            role = QtGui.QPalette.ColorRole.Base if hasattr(QtGui.QPalette, "ColorRole") else QtGui.QPalette.Base
            return pal.color(role).lightness() < 128
    except Exception:
        pass
    try:
        from PyQt5 import QtWidgets, QtGui
        app = QtWidgets.QApplication.instance()
        if app:
            pal = app.palette()
            role = QtGui.QPalette.ColorRole.Base if hasattr(QtGui.QPalette, "ColorRole") else QtGui.QPalette.Base
            return pal.color(role).lightness() < 128
    except Exception:
        pass
    return False


def _get_highlight_colors():
    """Return (asm_highlight_color, decomp_highlight_bg) based on active IDA theme."""
    if _is_dark_theme():
        # Soft slate blue / dark cyan highlight in dark mode
        return 0x304855, 0x283848
    else:
        # Soft sky blue / cyan highlight in light mode
        return 0xC0E0FF, 0xD0E8FF


def _line_ea(line):
    """Extract code linear address from loc_XXXXX, sub_XXXXX, or 0xXXXXX in line."""
    clean = ida_lines.tag_remove(line) if hasattr(ida_lines, "tag_remove") else line
    m = _EA.search(clean)
    if m:
        try:
            return int(m.group(1), 16)
        except ValueError:
            pass
    m = _ADDR_HEX.search(clean)
    if m:
        try:
            return int(m.group(1), 16)
        except ValueError:
            pass
    return None


def _jumpto_disasm(ea):
    """Jump to ea in IDA disassembly view (IDA View-A)."""
    if ea is None:
        return False
    try:
        flags = getattr(ida_kernwin, "UIJMP_IDAVIEW", 0x0001)
        return ida_kernwin.jumpto(ea, -1, flags)
    except Exception:
        try:
            return idaapi.jumpto(ea)
        except Exception:
            return False


class _ActionHandler(ida_kernwin.action_handler_t):
    def __init__(self, view, callback):
        super().__init__()
        self.view = view
        self.callback = callback

    def activate(self, ctx):
        self.callback()
        return 1

    def update(self, ctx):
        if not self.view:
            return ida_kernwin.AST_DISABLE_FOR_WIDGET
        widget = self.view.GetWidget()
        if widget and ctx.widget == widget:
            return ida_kernwin.AST_ENABLE_FOR_WIDGET
        return ida_kernwin.AST_DISABLE_FOR_WIDGET


class _DecompilerView(ida_kernwin.simplecustviewer_t):
    def __init__(self):
        super().__init__()
        self.current_func = None
        self.current_func_ea = None
        self.current_func_name = None
        self.mode = "c"
        self.line_ea = []
        self.line_to_eas = {}
        self.ea_to_line = {}
        self.raw_lines = []
        self.header_count = 0
        self.var_types = {}
        self.params = []
        self.param_types = {}
        self.sig_line = None
        self.ret_type = "void"
        self._selected_line = None
        self._highlighted_asm_eas = set()
        self._sync_lock = False
        self.sync_with_disasm = db.get_sync_enabled()
        self._nav_history = []
        self._last_token = None
        self._highlighted_token_lines = set()

    def build_or_refresh(self, target, mode=None, preserve_cursor=False):
        """Populate or update lines with decompiled Function or EA."""
        import spudec

        if mode is not None:
            self.mode = mode

        saved_pos = self.GetPos() if preserve_cursor else None

        if isinstance(target, int):
            try:
                func = spudec.decompile(target, check=True)
            except Exception as exc:
                print("spudec: decompilation failed for 0x%X: %s" % (target, exc))
                return False
        else:
            func = target

        self.current_func = func
        self.current_func_ea = func.start_ea
        self.current_func_name = func.name or ("sub_%X" % func.start_ea)
        self.clear_asm_highlights()

        if self.mode == "c":
            try:
                lines = spudec.pseudocode(func)
            except Exception as exc:
                print("spudec: pseudocode failed for %s: %s" % (self.current_func_name, exc))
                return False
            self.header_count = getattr(lines, "header_count", 0)
            self.line_to_eas = getattr(lines, "line_to_eas", {}) or {}
            self.ea_to_line = getattr(lines, "ea_to_line", {}) or {}
            self.raw_lines = list(lines)
            self.var_types = getattr(lines, "var_types", {}) or {}
            self.params = getattr(lines, "params", []) or []
            self.param_types = getattr(lines, "param_types", {}) or {}
            self.sig_line = getattr(lines, "sig_line", None)
            self.ret_type = getattr(lines, "ret_type", "void")
            self._selected_line = None

            self.ClearLines()
            self.line_ea = []
            fn_line_eas = getattr(lines, "line_eas", None)
            for i, ln in enumerate(lines):
                self.AddLine(highlight.colorize(ln))
                ea = fn_line_eas[i] if (fn_line_eas and i < len(fn_line_eas)) else None
                self.line_ea.append(ea if ea is not None else _line_ea(ln))
            self.Refresh()

            initial_stmt_line = None
            try:
                screen_ea = ida_kernwin.get_screen_ea()
                if screen_ea is not None and screen_ea in self.ea_to_line:
                    cand = self.ea_to_line[screen_ea]
                    if cand >= self.header_count:
                        initial_stmt_line = cand
            except Exception:
                pass

            if saved_pos and 0 <= saved_pos[0] < len(self.raw_lines):
                self.select_line(saved_pos[0], jump=True, update_asm=True, jump_asm=False)
                try:
                    self.Jump(saved_pos[0], saved_pos[1])
                except Exception:
                    pass
            elif initial_stmt_line is not None and initial_stmt_line < len(self.raw_lines):
                self.select_line(initial_stmt_line, jump=True, update_asm=True, jump_asm=False)
            else:
                self._selected_line = None
                try:
                    self.Jump(0, 0)
                except Exception:
                    pass
        else:
            # SSA IR mode
            self.header_count = 0
            self.line_to_eas = {}
            self.ea_to_line = {}
            self.raw_lines = []
            self.var_types = {}
            self.params = []
            self.param_types = {}
            self.sig_line = None
            self.ret_type = "void"
            self._selected_line = None
            self.ClearLines()
            self.line_ea = []
            self._render_ir(func)
            self.Refresh()

        return True

    def _render_ir(self, func):
        def _c(text, color):
            return ida_lines.COLSTR(text, color)

        name = func.name or "sub_%X" % func.start_ea
        self._add_ir_line(_c("; SPU SSA IR  --  %s  @ 0x%X" % (name, func.start_ea),
                             ida_lines.SCOLOR_AUTOCMT), func.start_ea)

        stats = getattr(func, "stats", {}) or {}
        if stats:
            self._add_ir_line(_c("; opt: %d rounds, %d instructions eliminated"
                                 % (stats.get("rounds", 0), stats.get("removed", 0)),
                                 ida_lines.SCOLOR_AUTOCMT))

        sc = getattr(func, "scalar_stats", {}) or {}
        if sc:
            line = ("; scalarised: %d scalar stores, %d scalar loads, "
                    "%d aligned loads, %d scalar ops"
                    % (sc.get("stores", 0), sc.get("loads", 0),
                       sc.get("aligned_loads", 0), sc.get("scalars", 0)))
            self._add_ir_line(_c(line, ida_lines.SCOLOR_AUTOCMT))

        unhandled = getattr(func, "unhandled", {}) or {}
        if unhandled:
            worst = sorted(unhandled.items(), key=lambda kv: -kv[1])
            self._add_ir_line(_c("; unmodelled (lifted as intrinsics): %s"
                                 % ", ".join("%s x%d" % (k, v) for k, v in worst),
                                 ida_lines.SCOLOR_ERROR))

        problems = getattr(func, "problems", []) or []
        for p in problems:
            self._add_ir_line(_c("; SSA PROBLEM: " + p, ida_lines.SCOLOR_ERROR))

        for b in func.blocks:
            self._add_ir_line("")
            preds = ", ".join("B%d" % p.id for p in b.preds) or "-"
            succs = ", ".join("B%d" % s.id for s in b.succs) or "-"
            self._add_ir_line(
                _c("B%d:" % b.id, ida_lines.SCOLOR_CREFTAIL) +
                _c("  ; 0x%X..0x%X  preds: %s  succs: %s"
                   % (b.start_ea, b.end_ea, preds, succs),
                   ida_lines.SCOLOR_AUTOCMT),
                b.start_ea)
            for insn in b.insns:
                text = str(insn)
                if insn.op == Op.PHI:
                    c_text = _c(text, ida_lines.SCOLOR_MACRO)
                elif insn.op.is_terminator:
                    c_text = _c(text, ida_lines.SCOLOR_KEYWORD)
                elif insn.op in (Op.LOADQ, Op.STOREQ, Op.LOAD, Op.STORE, Op.LOADU):
                    c_text = _c(text, ida_lines.SCOLOR_DREF)
                elif insn.op == Op.INTRINSIC:
                    c_text = _c(text, ida_lines.SCOLOR_ERROR)
                else:
                    c_text = text
                self._add_ir_line("    " + c_text, insn.ea)

    def _add_ir_line(self, text, ea=None):
        lineno = len(self.raw_lines)
        self.raw_lines.append(text)
        self.AddLine(text)
        self.line_ea.append(ea)
        if ea is not None:
            self.line_to_eas[lineno] = [ea]
            if ea not in self.ea_to_line:
                self.ea_to_line[ea] = lineno

    def clear_asm_highlights(self):
        """Restore default background color for all previously highlighted asm instructions."""
        self._last_token = None
        self._highlighted_token_lines.clear()
        if not self._highlighted_asm_eas:
            return
        import ida_nalt
        for ea in list(self._highlighted_asm_eas):
            try:
                ida_nalt.del_item_color(ea)
            except Exception:
                pass
        self._highlighted_asm_eas.clear()
        try:
            ida_kernwin.refresh_idaview_anyway()
        except Exception:
            pass

    def select_line(self, lineno, jump=False, update_asm=True, jump_asm=True):
        """Move selection to lineno, visually highlight row, and sync asm highlight."""
        if lineno < 0 or lineno >= len(self.raw_lines):
            return
        prev_line = self._selected_line
        self._selected_line = lineno

        asm_color, decomp_bg = _get_highlight_colors()

        if prev_line is not None and prev_line != lineno and 0 <= prev_line < len(self.raw_lines):
            try:
                colored = highlight.colorize(self.raw_lines[prev_line]) if self.mode == "c" else self.raw_lines[prev_line]
                self.EditLine(prev_line, colored, bgcolor=0xFFFFFFFF)
            except Exception:
                pass

        if self.raw_lines[lineno].strip():
            try:
                colored = highlight.colorize(self.raw_lines[lineno]) if self.mode == "c" else self.raw_lines[lineno]
                self.EditLine(lineno, colored, bgcolor=decomp_bg)
                self.RefreshCurrent()
            except Exception:
                pass
        else:
            self._selected_line = None

        if jump:
            try:
                self.Jump(lineno, 0)
            except Exception:
                pass

        if update_asm:
            self._update_asm_highlight_for_line(lineno, jump_asm=jump_asm)

    def _update_asm_highlight_for_line(self, lineno, jump_asm=True):
        """Highlight assembly instructions in IDA disassembly corresponding to decompiled line."""
        import ida_nalt

        for ea in list(self._highlighted_asm_eas):
            try:
                ida_nalt.del_item_color(ea)
            except Exception:
                pass
        self._highlighted_asm_eas.clear()

        eas = self.line_to_eas.get(lineno, [])
        if not eas and lineno < len(self.line_ea) and self.line_ea[lineno] is not None:
            eas = [self.line_ea[lineno]]

        if not eas:
            if self.line_to_eas:
                nearest_line = min(self.line_to_eas.keys(), key=lambda l: abs(l - lineno))
                eas = self.line_to_eas[nearest_line]
            elif self.current_func_ea is not None:
                eas = [self.current_func_ea]

        if eas:
            asm_color, _ = _get_highlight_colors()
            for ea in eas:
                try:
                    ida_nalt.set_item_color(ea, asm_color)
                    self._highlighted_asm_eas.add(ea)
                except Exception:
                    pass
            try:
                ida_kernwin.refresh_idaview_anyway()
            except Exception:
                pass

            if self.sync_with_disasm and jump_asm:
                try:
                    self._sync_lock = True
                    _jumpto_disasm(eas[0])
                except Exception:
                    pass
                finally:
                    self._sync_lock = False
        else:
            try:
                ida_kernwin.refresh_idaview_anyway()
            except Exception:
                pass

    def sync_from_disasm_ea(self, ea):
        """Called when cursor moves in disassembly; syncs selection in decompiled view."""
        if self._sync_lock:
            return
        lineno = self.ea_to_line.get(ea)
        if lineno is None:
            lineno = self._find_closest_line(ea)

        if lineno is None or (self.mode == "c" and lineno < self.header_count):
            if self._selected_line is not None:
                if 0 <= self._selected_line < len(self.raw_lines):
                    try:
                        colored = highlight.colorize(self.raw_lines[self._selected_line]) if self.mode == "c" else self.raw_lines[self._selected_line]
                        self.EditLine(self._selected_line, colored, bgcolor=0xFFFFFFFF)
                    except Exception:
                        pass
                self._selected_line = None
            self.clear_asm_highlights()
            return

        if lineno == self._selected_line:
            return

        self._sync_lock = True
        try:
            self.select_line(lineno, jump=True, update_asm=True, jump_asm=False)
        finally:
            self._sync_lock = False

    def _find_closest_line(self, ea):
        """Find the decompiled line whose instruction EA is closest to ea."""
        if not self.ea_to_line:
            return None
        closest_ea = min(self.ea_to_line.keys(), key=lambda e: abs(e - ea))
        return self.ea_to_line[closest_ea]

    def _update_token_highlight(self):
        """Highlight all occurrences of the identifier under caret across the view."""
        if self.mode != "c" or not self.raw_lines:
            return
        word, lineno, col = self.get_word_under_cursor()
        active_token = None
        if word and (word[0].isalpha() or word[0] == "_"):
            if word not in highlight.C_KEYWORDS and word not in highlight.C_TYPES:
                active_token = word

        if active_token == self._last_token:
            return

        self._last_token = active_token
        _, decomp_bg = _get_highlight_colors()

        to_restore = set(self._highlighted_token_lines)
        self._highlighted_token_lines.clear()

        new_matching = set()
        if active_token:
            pat = re.compile(r"\b" + re.escape(active_token) + r"\b")
            for idx, raw in enumerate(self.raw_lines):
                if idx >= self.header_count and pat.search(raw):
                    new_matching.add(idx)

        affected = to_restore.union(new_matching)
        for idx in affected:
            if 0 <= idx < len(self.raw_lines):
                try:
                    colored = highlight.colorize(self.raw_lines[idx], highlight_ident=active_token)
                    bg = decomp_bg if idx == self._selected_line else 0xFFFFFFFF
                    self.EditLine(idx, colored, bgcolor=bg)
                    if idx in new_matching:
                        self._highlighted_token_lines.add(idx)
                except Exception:
                    pass
        if affected:
            try:
                self.RefreshCurrent()
            except Exception:
                pass

    def on_cursor_changed(self):
        """Called when user clicks or navigates in decompiler window."""
        if self._sync_lock:
            return
        pos = self.GetPos()
        if not pos:
            return
        lineno = pos[0]
        if lineno < 0 or lineno >= len(self.raw_lines):
            return

        self._sync_lock = True
        try:
            if lineno != self._selected_line:
                self.select_line(lineno, jump=False, update_asm=True, jump_asm=True)
            self._update_token_highlight()
            if self.current_func_ea is not None:
                db.save_window_state(True, self.current_func_ea, lineno=lineno, col=pos[1], mode=self.mode)
        finally:
            self._sync_lock = False

    def OnCursorPosChanged(self):
        self.on_cursor_changed()

    def OnClick(self, shift):
        self.on_cursor_changed()

    def Close(self):
        self.clear_asm_highlights()
        if self.current_func_ea is not None:
            db.save_window_state(False, self.current_func_ea, mode=self.mode)
        return super().Close()

    def show_func(self, ea, preserve_cursor=False):
        """Decompile function at ea and refresh view."""
        import spudec
        pfn = ida_funcs.get_func(ea)
        entry = pfn.start_ea if pfn else ea
        spudec.clear_caches()
        return self.build_or_refresh(entry, mode=self.mode, preserve_cursor=preserve_cursor)

    def get_word_under_cursor(self):
        """Return (word, lineno, col) under cursor."""
        pos = self.GetPos()
        if not pos:
            word = self.GetCurrentWord()
            return word or "", 0, 0
        lineno, col = pos[0], pos[1]
        if lineno < 0 or lineno >= len(self.raw_lines):
            return "", lineno, col
        line = self.raw_lines[lineno]
        if not line:
            return "", lineno, col
        if col < 0:
            col = 0
        elif col >= len(line):
            col = len(line) - 1

        if not (line[col].isalnum() or line[col] == "_"):
            if col > 0 and (line[col - 1].isalnum() or line[col - 1] == "_"):
                col -= 1
            elif col + 1 < len(line) and (line[col + 1].isalnum() or line[col + 1] == "_"):
                col += 1
            else:
                return "", lineno, col

        start = col
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
            start -= 1
        end = col
        while end < len(line) and (line[end].isalnum() or line[end] == "_"):
            end += 1
        return line[start:end], lineno, col

    def _validate_c_type(self, type_str):
        """Validate type_str against IDA type library (native SPU types, structs, typedefs)."""
        try:
            import ida_typeinf
            tif = ida_typeinf.tinfo_t()
            clean = type_str.strip().rstrip(";")
            if ida_typeinf.parse_decl(tif, None, clean + ";", ida_typeinf.PT_TYP):
                return True
            if ida_typeinf.parse_decl(tif, None, f"{clean} dummy;", ida_typeinf.PT_SIL) is not None:
                return True
            return False
        except Exception:
            return True

    def _update_param_type(self, param_name, new_type):
        """Update a single parameter's type in prototype and IDA type info."""
        data = db.get_func_data(self.current_func_ea)
        proto_str = data.get("proto")
        if not proto_str and self.sig_line is not None and self.sig_line < len(self.raw_lines):
            proto_str = self.raw_lines[self.sig_line].strip()
        if not proto_str:
            proto_str = f"{self.ret_type} {self.current_func_name}()"

        from . import abi
        parsed = abi._parse_proto(proto_str, self.current_func_name)
        if parsed:
            found = False
            for p in parsed.params:
                if p.name == param_name:
                    p.type_str = new_type if new_type else "qword"
                    found = True
                    break
            if not found and new_type:
                parsed.params.append(abi.Param(name=param_name, type_str=new_type))

            pstrs = []
            for p in parsed.params:
                sep = "" if p.type_str.endswith("*") else " "
                pstrs.append(f"{p.type_str}{sep}{p.name}")
            new_proto = f"{parsed.ret_type} {parsed.name}({', '.join(pstrs)})"
            db.set_func_proto(self.current_func_ea, new_proto)
            db.set_var_type(self.current_func_ea, param_name, new_type if new_type else None)
            try:
                tif = ida_typeinf.tinfo_t()
                if ida_typeinf.parse_decl(tif, None, new_proto + ";", ida_typeinf.PT_SIL):
                    ida_typeinf.apply_tinfo(self.current_func_ea, tif, ida_typeinf.TINFO_DEFINITE)
            except Exception:
                pass
        else:
            db.set_var_type(self.current_func_ea, param_name, new_type if new_type else None)

    def on_rename(self):
        """Rename variable, argument, or function under cursor and persist in IDB netnode."""
        import spudec
        if self.current_func_ea is None:
            return
        word, lineno, col = self.get_word_under_cursor()
        if not word:
            word = ida_kernwin.ask_str("", 0, "Enter identifier to rename")
            if not word:
                return

        # 1. Renaming the function itself
        if word == self.current_func_name or (self.sig_line is not None and lineno == self.sig_line and word == self.current_func_name):
            new_name = ida_kernwin.ask_str(word, 0, f"Rename Function '{word}'")
            if new_name is None:
                return
            new_name = new_name.strip()
            if new_name and new_name != word:
                import ida_name
                ida_name.set_name(self.current_func_ea, new_name)
                # If prototype override exists, update function name in prototype
                data = db.get_func_data(self.current_func_ea)
                if data.get("proto"):
                    from . import abi
                    parsed = abi._parse_proto(data["proto"])
                    if parsed:
                        pstrs = []
                        for p in parsed.params:
                            sep = "" if p.type_str.endswith("*") else " "
                            pstrs.append(f"{p.type_str}{sep}{p.name}")
                        new_proto = f"{parsed.ret_type} {new_name}({', '.join(pstrs)})"
                        db.set_func_proto(self.current_func_ea, new_proto)
                self.current_func_name = new_name
            self.show_func(self.current_func_ea, preserve_cursor=True)
            return

        # 2. Renaming a parameter
        is_param = (word in self.params) or (self.sig_line is not None and lineno == self.sig_line and "(" in self.raw_lines[lineno])
        prompt_label = f"parameter '{word}'" if is_param else f"variable '{word}'"
        new_name = ida_kernwin.ask_str(word, 0, f"Rename {prompt_label} in {self.current_func_name}")
        if new_name is None:
            return
        new_name = new_name.strip()

        # Update variable / parameter rename in persistent IDB netnode
        db.set_var_rename(self.current_func_ea, word, new_name)

        # If a prototype override exists in db and word was a parameter, keep prototype string in sync
        data = db.get_func_data(self.current_func_ea)
        if data.get("proto"):
            from . import abi
            parsed = abi._parse_proto(data["proto"])
            if parsed and parsed.params:
                modified = False
                for p in parsed.params:
                    if p.name == word:
                        p.name = new_name if new_name else word
                        modified = True
                        break
                if modified:
                    pstrs = []
                    for p in parsed.params:
                        sep = "" if p.type_str.endswith("*") else " "
                        pstrs.append(f"{p.type_str}{sep}{p.name}")
                    new_proto = f"{parsed.ret_type} {parsed.name}({', '.join(pstrs)})"
                    db.set_func_proto(self.current_func_ea, new_proto)
                    try:
                        tif = ida_typeinf.tinfo_t()
                        if ida_typeinf.parse_decl(tif, None, new_proto + ";", ida_typeinf.PT_SIL):
                            ida_typeinf.apply_tinfo(self.current_func_ea, tif, ida_typeinf.TINFO_DEFINITE)
                    except Exception:
                        pass

        # If word is a stack frame variable, also rename IDA's stack frame member
        self._rename_frame_member(word, new_name)

        # If word is a global symbol in IDA, rename it
        try:
            import ida_name, ida_idaapi
            ea = ida_name.get_name_ea(ida_idaapi.BADADDR, word)
            if ea != ida_idaapi.BADADDR and new_name:
                ida_name.set_name(ea, new_name)
        except Exception:
            pass

        self.show_func(self.current_func_ea, preserve_cursor=True)

    def _rename_frame_member(self, old_name, new_name):
        """Rename stack variable member in IDA's frame structure if present."""
        try:
            import ida_frame
            pfn = ida_funcs.get_func(self.current_func_ea)
            if not pfn:
                return
            tif = ida_typeinf.tinfo_t()
            if ida_frame.get_func_frame(tif, pfn):
                udt = ida_typeinf.udt_type_data_t()
                if tif.get_udt_details(udt):
                    for m in udt:
                        if m.name == old_name:
                            m.name = new_name
                            tif.create_udt(udt, ida_typeinf.BTF_STRUCT)
                            break
        except Exception:
            pass

    def on_set_type(self):
        """Set variable type, parameter type, or function prototype with IDA type integration."""
        import spudec
        if self.current_func_ea is None:
            return

        spudec.ensure_spu_types()
        word, lineno, col = self.get_word_under_cursor()
        is_sig_line = (self.sig_line is not None and lineno == self.sig_line) or \
                      (lineno == 0 and ("(" in self.raw_lines[0] and ")" in self.raw_lines[0] and not self.raw_lines[0].startswith("//")))

        # Check if cursor is on the signature line
        if is_sig_line:
            line_str = self.raw_lines[lineno]
            # 1. Determine if cursor is on a specific parameter in the signature
            param_match = None
            if word and word in self.params:
                param_match = word
            else:
                # Check if col falls within any parameter span
                if "(" in line_str and ")" in line_str:
                    lparen = line_str.index("(")
                    rparen = line_str.rindex(")")
                    if lparen < col <= rparen:
                        args_str = line_str[lparen + 1:rparen]
                        arg_parts = args_str.split(",")
                        curr_offset = lparen + 1
                        for idx, part in enumerate(arg_parts):
                            part_len = len(part)
                            if curr_offset <= col <= curr_offset + part_len + 1:
                                if idx < len(self.params):
                                    param_match = self.params[idx]
                                break
                            curr_offset += part_len + 1

            if param_match:
                # Editing a specific parameter's type!
                curr_type = self.param_types.get(param_match, "qword")
                new_type = ida_kernwin.ask_str(curr_type, 0, f"Set Type for parameter '{param_match}'")
                if new_type is None:
                    return
                new_type = new_type.strip()
                if new_type:
                    valid = self._validate_c_type(new_type)
                    if not valid:
                        ida_kernwin.warning(f"Warning: '{new_type}' could not be parsed by IDA's type library. Applying override anyway.")
                self._update_param_type(param_match, new_type)
                self.show_func(self.current_func_ea, preserve_cursor=True)
                return

            # 2. Check if cursor is on return type in signature
            if word and word == self.ret_type:
                new_ret = ida_kernwin.ask_str(self.ret_type, 0, f"Set Return Type for {self.current_func_name}")
                if new_ret is None:
                    return
                new_ret = new_ret.strip()
                if new_ret:
                    valid = self._validate_c_type(new_ret)
                    if not valid:
                        ida_kernwin.warning(f"Warning: '{new_ret}' could not be parsed by IDA. Applying override anyway.")
                    db.set_var_type(self.current_func_ea, "__return__", new_ret)
                    # Also update prototype if present
                    data = db.get_func_data(self.current_func_ea)
                    if data.get("proto"):
                        from . import abi
                        parsed = abi._parse_proto(data["proto"], self.current_func_name)
                        if parsed:
                            pstrs = []
                            for p in parsed.params:
                                sep = "" if p.type_str.endswith("*") else " "
                                pstrs.append(f"{p.type_str}{sep}{p.name}")
                            new_proto = f"{new_ret} {parsed.name}({', '.join(pstrs)})"
                            db.set_func_proto(self.current_func_ea, new_proto)
                            try:
                                tif = ida_typeinf.tinfo_t()
                                if ida_typeinf.parse_decl(tif, None, new_proto + ";", ida_typeinf.PT_SIL):
                                    ida_typeinf.apply_tinfo(self.current_func_ea, tif, ida_typeinf.TINFO_DEFINITE)
                            except Exception:
                                pass
                else:
                    db.set_var_type(self.current_func_ea, "__return__", None)
                self.show_func(self.current_func_ea, preserve_cursor=True)
                return

            # 3. Cursor is on the function name or general signature line -> edit full prototype
            data = db.get_func_data(self.current_func_ea)
            curr_proto = data.get("proto") or line_str.strip()
            new_proto = ida_kernwin.ask_str(curr_proto, 0, f"Set Function Prototype for {self.current_func_name}")
            if new_proto is None:
                return
            new_proto = new_proto.strip()
            if not new_proto:
                db.set_func_proto(self.current_func_ea, None)
            else:
                # Validate prototype with IDA
                tif = ida_typeinf.tinfo_t()
                valid = ida_typeinf.parse_decl(tif, None, new_proto + ";", ida_typeinf.PT_SIL)
                if not valid:
                    ida_kernwin.warning(f"Warning: Prototype '{new_proto}' could not be parsed by IDA. Storing override anyway.")
                db.set_func_proto(self.current_func_ea, new_proto)
                if valid:
                    try:
                        ida_typeinf.apply_tinfo(self.current_func_ea, tif, ida_typeinf.TINFO_DEFINITE)
                    except Exception:
                        pass

            self.show_func(self.current_func_ea, preserve_cursor=True)
            return

        # Cursor is on a local variable, live-in, or stack variable in the body
        if not word:
            return

        curr_type = self.var_types.get(word, "qword")
        new_type = ida_kernwin.ask_str(curr_type, 0, f"Set Type for variable '{word}'")
        if new_type is None:
            return
        new_type = new_type.strip()
        if new_type:
            valid = self._validate_c_type(new_type)
            if not valid:
                ida_kernwin.warning(f"Warning: '{new_type}' could not be parsed by IDA's type library. Applying override anyway.")

        db.set_var_type(self.current_func_ea, word, new_type if new_type else None)
        self.show_func(self.current_func_ea, preserve_cursor=True)

    def toggle_sync(self):
        """Toggle synchronization with IDA disassembly view."""
        self.sync_with_disasm = not self.sync_with_disasm
        db.set_sync_enabled(self.sync_with_disasm)
        state_str = "ENABLED" if self.sync_with_disasm else "DISABLED"
        print(f"spudec: Sync with Disassembly {state_str}")
        if self.sync_with_disasm:
            ea = ida_kernwin.get_screen_ea()
            pfn = ida_funcs.get_func(ea)
            if pfn and pfn.start_ea != self.current_func_ea:
                self.show_func(pfn.start_ea)

    def toggle_ir(self):
        """Toggle between C pseudocode and SSA IR views."""
        self.mode = "ir" if self.mode == "c" else "c"
        if self.current_func_ea is not None:
            self.show_func(self.current_func_ea)

    def jump_disasm(self):
        """Jump to the linear address or switch decompiler view to callee function."""
        word, lineno, col = self.get_word_under_cursor()
        if word:
            # 1. Check if word is a function in IDA -> click-through navigation!
            try:
                import ida_name, ida_idaapi
                target_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, word)
                if target_ea != ida_idaapi.BADADDR:
                    pfn = ida_funcs.get_func(target_ea)
                    if pfn and pfn.start_ea == target_ea:
                        _jumpto_disasm(target_ea)
                        if target_ea != self.current_func_ea:
                            pos = self.GetPos()
                            self._nav_history.append((self.current_func_ea, pos[0] if pos else 0, pos[1] if pos else 0))
                            self.show_func(target_ea)
                        return True
                    else:
                        _jumpto_disasm(target_ea)
                        return True
            except Exception:
                pass

            # 2. Check if word is sub_XXXXXX or loc_XXXXXX
            m = _EA.search(word)
            if m:
                try:
                    target_ea = int(m.group(1), 16)
                    pfn = ida_funcs.get_func(target_ea)
                    _jumpto_disasm(target_ea)
                    if pfn and pfn.start_ea == target_ea and target_ea != self.current_func_ea:
                        pos = self.GetPos()
                        self._nav_history.append((self.current_func_ea, pos[0] if pos else 0, pos[1] if pos else 0))
                        self.show_func(target_ea)
                    return True
                except ValueError:
                    pass

            # 3. Check hex address
            m = _ADDR_HEX.search(word)
            if m:
                try:
                    target_ea = int(m.group(1), 16)
                    _jumpto_disasm(target_ea)
                    return True
                except ValueError:
                    pass

        # Fallback: jump to line_ea or line_to_eas of current line
        target_ea = None
        if lineno is not None:
            if lineno in self.line_to_eas and self.line_to_eas[lineno]:
                target_ea = self.line_to_eas[lineno][0]
            elif lineno < len(self.line_ea) and self.line_ea[lineno] is not None:
                target_ea = self.line_ea[lineno]
            elif self.line_to_eas:
                nearest_line = min(self.line_to_eas.keys(), key=lambda l: abs(l - lineno))
                target_ea = self.line_to_eas[nearest_line][0]
            elif self.current_func_ea is not None:
                target_ea = self.current_func_ea

        if target_ea is not None:
            _jumpto_disasm(target_ea)
            return True
        return False

    def on_invert_if(self):
        """Invert if-condition and swap then/else branches (Hex-Rays 'I' hotkey)."""
        pos = self.GetPos()
        if not pos or pos[0] >= len(self.raw_lines):
            return
        lineno = pos[0]
        ea = self.line_ea[lineno] if lineno < len(self.line_ea) else None
        if ea is None and lineno in self.line_to_eas and self.line_to_eas[lineno]:
            ea = self.line_to_eas[lineno][0]
        if ea is not None:
            db.toggle_iflag(self.current_func_ea, ea)
            self.show_func(self.current_func_ea, preserve_cursor=True)

    def on_edit_comment(self):
        """Add or edit statement comment attached to linear address (Hex-Rays '/' hotkey)."""
        pos = self.GetPos()
        if not pos or pos[0] >= len(self.raw_lines):
            return
        lineno = pos[0]
        ea = self.line_ea[lineno] if lineno < len(self.line_ea) else None
        if ea is None and lineno in self.line_to_eas and self.line_to_eas[lineno]:
            ea = self.line_to_eas[lineno][0]
        if ea is not None:
            existing = db.get_statement_cmt(self.current_func_ea, ea) or ""
            new_cmt = ida_kernwin.ask_str(existing, 0, f"Comment for statement at 0x{ea:X}")
            if new_cmt is not None:
                db.set_statement_cmt(self.current_func_ea, ea, new_cmt)
                self.show_func(self.current_func_ea, preserve_cursor=True)

    def on_set_numform(self, form_str):
        """Set immediate number format ('hex', 'dec', 'char')."""
        pos = self.GetPos()
        if not pos or pos[0] >= len(self.raw_lines):
            return
        lineno = pos[0]
        ea = self.line_ea[lineno] if lineno < len(self.line_ea) else None
        if ea is not None:
            db.set_numform(self.current_func_ea, ea, 0, form_str)
            self.show_func(self.current_func_ea, preserve_cursor=True)

    def OnDblClick(self, shift):
        return self.jump_disasm()

    def OnKeydown(self, vkey, shift):
        if vkey in (ord('N'), ord('n')):
            self.on_rename()
            return True
        if vkey in (ord('Y'), ord('y')):
            self.on_set_type()
            return True
        if vkey in (ord('I'), ord('i')):
            if shift & 1:
                self.toggle_ir()
            else:
                self.on_invert_if()
            return True
        if vkey == ord('/'):
            self.on_edit_comment()
            return True
        if vkey in (ord('H'), ord('h')):
            self.on_set_numform('hex')
            return True
        if vkey in (ord('D'), ord('d')):
            self.on_set_numform('dec')
            return True
        if vkey in (ord('R'), ord('r')):
            self.on_set_numform('char')
            return True
        if vkey == 116:  # F5
            if self.current_func_ea is not None:
                self.show_func(self.current_func_ea)
            return True
        if vkey in (13, 32):  # Enter or Space
            return self.jump_disasm()
        if vkey == 9:  # Tab
            self.toggle_sync()
            return True
        if vkey == 27:  # Esc
            if self._nav_history:
                prev_ea, prev_line, prev_col = self._nav_history.pop()
                self.show_func(prev_ea)
                try:
                    self.select_line(prev_line, jump=True, update_asm=True, jump_asm=True)
                    self.Jump(prev_line, prev_col)
                except Exception:
                    pass
                return True
            global _active_viewer
            self.clear_asm_highlights()
            _active_viewer = None
            self.Close()
            return True
        # Navigation keys: Up(38), Down(40), PageUp(33), PageDown(34), Home(36), End(35)
        if vkey in (38, 40, 33, 34, 36, 35):
            try:
                ida_kernwin.execute_ui_requests([self.on_cursor_changed])
            except Exception:
                pass
        return False


class _UiHooks(ida_kernwin.UI_Hooks):
    def __init__(self, view):
        super().__init__()
        self.view = view

    def create_desktop_widget(self, title, cfg):
        if title == _VIEW_TITLE:
            global _active_viewer, _view_hooks
            if _active_viewer is None or _active_viewer.GetWidget() is None:
                v = _DecompilerView()
                if v.Create(_VIEW_TITLE):
                    _active_viewer = v
                    _register_actions(_active_viewer)
                    self.view = _active_viewer
                    if _view_hooks is None:
                        _view_hooks = _CustViewHooks(_active_viewer)
                        _view_hooks.hook()
                    else:
                        _view_hooks.view = _active_viewer
                    return _active_viewer.GetWidget()
            else:
                return _active_viewer.GetWidget()
        return None

    def desktop_applied(self, name, from_idb, type):
        self._restore_from_state()

    def database_inited(self, is_new_database):
        self._restore_from_state()

    def saving(self):
        self._record_current_state()

    def saved(self):
        self._record_current_state()

    def _record_current_state(self):
        global _active_viewer
        if _active_viewer is not None and _active_viewer.GetWidget() is not None:
            pos = _active_viewer.GetPos() or (0, 0)
            if _active_viewer.current_func_ea is not None:
                db.save_window_state(True, _active_viewer.current_func_ea,
                                     lineno=pos[0], col=pos[1], mode=_active_viewer.mode)

    def screen_ea_changed(self, ea, prev_ea):
        """Automatically follow current function in disassembly and sync line selection."""
        if not self.view or self.view._sync_lock:
            return
        if _active_viewer is not self.view:
            return
        pfn = ida_funcs.get_func(ea)
        if pfn is None:
            return
        if pfn.start_ea != self.view.current_func_ea:
            if self.view.sync_with_disasm:
                self.view.show_func(pfn.start_ea)
            else:
                return
        self.view.sync_from_disasm_ea(ea)

    def widget_invisible(self, widget):
        """Reset active viewer and persist closed state when the decompiler window is closed."""
        if self.view and widget == self.view.GetWidget():
            global _active_viewer
            self.view.clear_asm_highlights()
            if self.view.current_func_ea is not None:
                db.save_window_state(False, self.view.current_func_ea, mode=self.view.mode)
            _active_viewer = None

    def finish_populating_widget_popup(self, widget, popup_handle):
        """Attach decompiler context actions to the right-click popup menu."""
        if not self.view:
            return
        view_widget = self.view.GetWidget()
        if widget != view_widget:
            return

        ida_kernwin.attach_action_to_popup(widget, popup_handle, ACTION_RENAME)
        ida_kernwin.attach_action_to_popup(widget, popup_handle, ACTION_SET_TYPE)
        ida_kernwin.attach_action_to_popup(widget, popup_handle, ACTION_TOGGLE_SYNC)
        ida_kernwin.attach_action_to_popup(widget, popup_handle, ACTION_JUMP_DISASM)
        ida_kernwin.attach_action_to_popup(widget, popup_handle, ACTION_REFRESH)
        ida_kernwin.attach_action_to_popup(widget, popup_handle, ACTION_TOGGLE_IR)

    def _restore_from_state(self):
        global _active_viewer
        state = db.load_window_state()
        if not state.get("is_open") or not state.get("func_ea"):
            return
        func_ea = state["func_ea"]
        mode = state.get("mode", "c")
        lineno = state.get("lineno", 0)
        col = state.get("col", 0)
        if _active_viewer is not None and _active_viewer.GetWidget() is not None:
            if _active_viewer.current_func_ea != func_ea or not _active_viewer.raw_lines:
                _active_viewer.show_func(func_ea)
            if lineno:
                try:
                    _active_viewer.Jump(lineno, col)
                except Exception:
                    pass
        else:
            show(func_ea, mode=mode, auto_dock=False)
            if _active_viewer is not None and lineno:
                try:
                    _active_viewer.Jump(lineno, col)
                except Exception:
                    pass


class _CustViewHooks(ida_kernwin.View_Hooks):
    def __init__(self, view):
        super().__init__()
        self.view = view

    def view_curpos(self, view):
        """Called whenever cursor position changes in any view."""
        if self.view and view == self.view.GetWidget():
            self.view.on_cursor_changed()

    def view_click(self, view, event):
        """Called on mouse click in view."""
        if self.view and view == self.view.GetWidget():
            self.view.on_cursor_changed()

    def view_close(self, view):
        """Called when view is closed."""
        if self.view and view == self.view.GetWidget():
            self.view.clear_asm_highlights()


def _register_actions(view):
    """Register IDA actions for popup menu."""
    actions = [
        (ACTION_RENAME, "Rename variable/symbol... (N)", view.on_rename),
        (ACTION_SET_TYPE, "Set variable/function type... (Y)", view.on_set_type),
        (ACTION_TOGGLE_SYNC, "Sync with Disassembly (Tab)", view.toggle_sync),
        (ACTION_JUMP_DISASM, "Jump to Disassembly / Callee (Enter)", view.jump_disasm),
        (ACTION_REFRESH, "Refresh Decompilation (F5)", lambda: view.show_func(view.current_func_ea)),
        (ACTION_TOGGLE_IR, "Toggle Pseudocode / SSA IR (I)", view.toggle_ir),
    ]
    for action_id, label, callback in actions:
        ida_kernwin.unregister_action(action_id)
        desc = ida_kernwin.action_desc_t(
            action_id,
            label,
            _ActionHandler(view, callback),
            None,
            label
        )
        ida_kernwin.register_action(desc)


def show(target, mode="c", auto_dock=True):
    """Open or refresh the dockable decompiler view for the specified function or EA."""
    global _active_viewer, _ui_hooks, _view_hooks
    first_create = False
    if _active_viewer is None or _active_viewer.GetWidget() is None:
        v = _DecompilerView()
        if not v.Create(_VIEW_TITLE):
            return None
        _active_viewer = v
        _register_actions(_active_viewer)
        first_create = True

        if _ui_hooks is None:
            _ui_hooks = _UiHooks(_active_viewer)
            _ui_hooks.hook()
        else:
            _ui_hooks.view = _active_viewer

        if _view_hooks is None:
            _view_hooks = _CustViewHooks(_active_viewer)
            _view_hooks.hook()
        else:
            _view_hooks.view = _active_viewer

    _active_viewer.build_or_refresh(target, mode=mode)
    _active_viewer.Show()

    if first_create and auto_dock:
        try:
            ida_kernwin.set_dock_pos(_VIEW_TITLE, "IDA View-A", ida_kernwin.DP_RIGHT)
        except Exception:
            pass

    if _active_viewer.current_func_ea is not None:
        pos = _active_viewer.GetPos() or (0, 0)
        db.save_window_state(True, _active_viewer.current_func_ea, lineno=pos[0], col=pos[1], mode=_active_viewer.mode)

    return _active_viewer


def init_hooks():
    """Initialize UI hooks early (at plugin load) to support desktop/docking persistence."""
    global _ui_hooks
    if _ui_hooks is None:
        _ui_hooks = _UiHooks(None)
        _ui_hooks.hook()


class ListingViewer(ida_kernwin.simplecustviewer_t):
    """
    A plain scrollable listing, for the whole-database decompilation.

    Double-click on a `// name @ 0xEA` header line jumps there.
    """

    def __init__(self):
        super(ListingViewer, self).__init__()
        self.line_ea = []

    def build(self, title, lines):
        if not self.Create(title):
            return False
        self.line_ea = []
        for ln in lines:
            ea = None
            m = re.search(r"@\s*0x([0-9A-Fa-f]+)", ln)
            if m:
                try:
                    ea = int(m.group(1), 16)
                except ValueError:
                    ea = None
            self.AddLine(highlight.colorize(ln))
            self.line_ea.append(ea)
        return True

    def OnDblClick(self, shift):
        n = self.GetLineNo()
        if n is None or n >= len(self.line_ea):
            return False
        ea = self.line_ea[n]
        if ea is None:
            return False
        idaapi.jumpto(ea)
        return True

    def OnKeydown(self, vkey, shift):
        if vkey == 27:
            self.Close()
            return True
        return False


def show_listing(title, lines):
    v = ListingViewer()
    if not v.build(title, lines):
        return None
    v.Show()
    return v


def cleanup():
    """Unregister hooks and actions on plugin unload."""
    global _active_viewer, _ui_hooks, _view_hooks
    if _active_viewer is not None:
        try:
            _active_viewer.clear_asm_highlights()
        except Exception:
            pass
    if _ui_hooks is not None:
        try:
            _ui_hooks.unhook()
        except Exception:
            pass
        _ui_hooks = None
    if _view_hooks is not None:
        try:
            _view_hooks.unhook()
        except Exception:
            pass
        _view_hooks = None
    for action_id in [ACTION_RENAME, ACTION_SET_TYPE, ACTION_TOGGLE_SYNC, ACTION_JUMP_DISASM, ACTION_REFRESH, ACTION_TOGGLE_IR]:
        try:
            ida_kernwin.unregister_action(action_id)
        except Exception:
            pass
    _active_viewer = None
