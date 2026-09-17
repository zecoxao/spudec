"""
IDA plugin entry point for the SPU decompiler front end.

Install by copying this file *and* the ``spudec`` package directory next to
each other into IDA's user plugin directory:

    %APPDATA%\\Hex-Rays\\IDA Pro\\plugins\\

    Ctrl-Shift-S   decompile the function under the cursor
    Ctrl-F5        decompile everything in the database

The patched ``spu.py`` belongs in %APPDATA%\\Hex-Rays\\IDA Pro\\procs\\ .
"""

import os
import sys
import time
import traceback

import ida_idaapi
import ida_idp
import ida_kernwin
import ida_funcs
import ida_name
import idaapi

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


HOTKEY_ONE = "Ctrl-Shift-S"
HOTKEY_ALL = "Ctrl-F5"

# Viewers must outlive run(); IDA does not hold a reference for us.
_views = []
_hotkeys = []


def _check_spu():
    if ida_idp.ph_get_id() != ida_idp.PLFM_SPU:
        ida_kernwin.warning(
            "spudec: this database is not SPU "
            "(processor id %d, expected %d)."
            % (ida_idp.ph_get_id(), ida_idp.PLFM_SPU))
        return False
    return True


def _name_of(ea):
    return ida_name.get_name(ea) or ("sub_%X" % ea)


# ---------------------------------------------------------------------------
# one function
# ---------------------------------------------------------------------------


def _run_one():
    import spudec
    from spudec import view

    if not _check_spu():
        return

    ea = ida_kernwin.get_screen_ea()
    f = ida_funcs.get_func(ea)
    if f is None:
        ida_kernwin.warning("spudec: no function at 0x%X.\n"
                            "Create one with 'P' first." % ea)
        return

    try:
        func = spudec.decompile(f.start_ea, check=True)
    except Exception:
        ida_kernwin.warning("spudec failed:\n\n%s" % traceback.format_exc())
        return

    v = view.show(func)
    if v is not None:
        _views.append(v)

    n = sum(len(b.insns) for b in func.blocks)
    msg = "spudec: %s -> %d blocks, %d IR instructions" % (
        func.name or "sub_%X" % func.start_ea, len(func.blocks), n)
    info = getattr(func, "structure_info", None)
    if info:
        msg += ", %d loops, %d gotos" % (info.get("loops", 0),
                                         info.get("gotos", 0))
    if func.unhandled:
        msg += " (%d unmodelled mnemonics)" % len(func.unhandled)
    if func.problems:
        msg += " -- %d SSA problems, see the viewer" % len(func.problems)
    print(msg)


# ---------------------------------------------------------------------------
# everything
# ---------------------------------------------------------------------------

# Above this, the listing is offered as a file only: a custom viewer holding
# a few hundred thousand lines is painful to scroll and slow to build.
VIEWER_LINE_LIMIT = 60000


def _run_all():
    import spudec
    from spudec import view

    if not _check_spu():
        return

    total = ida_funcs.get_func_qty()
    if total == 0:
        ida_kernwin.warning("spudec: this database has no functions.")
        return

    state = {"t": 0.0}

    def progress(done, n, name):
        # Throttled: updating the wait box for every function costs more than
        # the decompilation does on small ones.
        now = time.time()
        if now - state["t"] > 0.1:
            state["t"] = now
            ida_kernwin.replace_wait_box(
                "Decompiling %d/%d\n%s" % (done + 1, n, name[:60]))
        return not ida_kernwin.user_cancelled()

    ida_kernwin.show_wait_box("Decompiling %d functions..." % total)
    t0 = time.time()
    try:
        lines, stats = spudec.decompile_all(name_of=_name_of,
                                            progress=progress)
    except Exception:
        ida_kernwin.hide_wait_box()
        ida_kernwin.warning("spudec: decompile all failed:\n\n%s"
                            % traceback.format_exc())
        return
    finally:
        try:
            ida_kernwin.hide_wait_box()
        except Exception:
            pass
    dt = time.time() - t0

    summary = ("spudec: %d functions in %.1fs (%d failed, %d SSA problems, "
               "%d loops, %d gotos)"
               % (stats["functions"], dt, stats["failed"], stats["problems"],
                  stats["loops"], stats["gotos"]))
    if stats["cancelled"]:
        summary += "  [CANCELLED -- partial output]"
    print(summary)

    # Offer to save, the way Hex-Rays' "Decompile all" does.
    import ida_nalt
    default = os.path.splitext(
        ida_nalt.get_input_file_path() or "spu")[0] + ".c"
    path = ida_kernwin.ask_file(True, default,
                                "Save the decompiled listing")
    if path:
        try:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write("\n".join(lines))
                fp.write("\n")
            print("spudec: wrote %d lines to %s" % (len(lines), path))
        except Exception as exc:
            ida_kernwin.warning("spudec: could not write %s:\n%s"
                                % (path, exc))

    if len(lines) > VIEWER_LINE_LIMIT:
        ida_kernwin.info(
            "spudec: %d lines is too many for the viewer.\n"
            "%s" % (len(lines),
                    "Saved to %s" % path if path
                    else "Re-run and choose a file to save it."))
        return

    v = view.show_listing("SPU decompilation - all functions", lines)
    if v is not None:
        _views.append(v)


# ---------------------------------------------------------------------------


class SpuDecPlugin(ida_idaapi.plugin_t):
    flags = 0
    comment = "SPU lifter + SSA IR + pseudocode (Hex-Rays has no SPU backend)"
    help = ("%s decompiles the current function, %s decompiles everything."
            % (HOTKEY_ONE, HOTKEY_ALL))
    wanted_name = "SPU decompiler (spudec)"
    wanted_hotkey = HOTKEY_ONE

    def init(self):
        if ida_idp.ph_get_id() != ida_idp.PLFM_SPU:
            return ida_idaapi.PLUGIN_SKIP
        ctx = ida_kernwin.add_hotkey(HOTKEY_ALL, _run_all)
        if ctx is not None:
            _hotkeys.append(ctx)
        print("spudec: loaded -- %s = this function, %s = decompile all"
              % (HOTKEY_ONE, HOTKEY_ALL))
        return ida_idaapi.PLUGIN_KEEP

    def run(self, arg):
        _run_one()

    def term(self):
        for ctx in _hotkeys:
            try:
                ida_kernwin.del_hotkey(ctx)
            except Exception:
                pass
        del _hotkeys[:]
        del _views[:]


def PLUGIN_ENTRY():
    return SpuDecPlugin()
