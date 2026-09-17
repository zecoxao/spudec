"""
IDA plugin entry point for the SPU decompiler front end.

Install by copying this file *and* the ``spudec`` package directory next to
each other into IDA's user plugin directory:

    %APPDATA%\\Hex-Rays\\IDA Pro\\plugins\\

Then press Ctrl-Shift-S inside an SPU function.
"""

import os
import sys
import traceback

import ida_idaapi
import ida_idp
import ida_kernwin
import ida_funcs
import idaapi

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


ACTION_HOTKEY = "Ctrl-Shift-S"

# The viewer must outlive run(); IDA does not hold a reference for us.
_views = []


def _run():
    import spudec
    from spudec import view

    if ida_idp.ph_get_id() != ida_idp.PLFM_SPU:
        ida_kernwin.warning(
            "spudec: this database is not SPU "
            "(processor id %d, expected %d)."
            % (ida_idp.ph_get_id(), ida_idp.PLFM_SPU))
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


class SpuDecPlugin(ida_idaapi.plugin_t):
    flags = 0
    comment = "SPU lifter + SSA IR (standalone; Hex-Rays has no SPU backend)"
    help = "Press %s inside an SPU function." % ACTION_HOTKEY
    wanted_name = "SPU decompiler (spudec)"
    wanted_hotkey = ACTION_HOTKEY

    def init(self):
        if ida_idp.ph_get_id() != ida_idp.PLFM_SPU:
            return ida_idaapi.PLUGIN_SKIP
        print("spudec: loaded (%s)" % ACTION_HOTKEY)
        return ida_idaapi.PLUGIN_KEEP

    def run(self, arg):
        _run()

    def term(self):
        del _views[:]


def PLUGIN_ENTRY():
    return SpuDecPlugin()
