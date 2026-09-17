"""
spudec -- an SPU (Cell BE Synergistic Processor Unit) decompiler front end for
IDA Pro 9.x.

Hex-Rays cannot be extended to new architectures from the public SDK: microcode
generation lives inside the closed per-processor backends (hexarm/hexppc/...),
and the exposed hooks (microcode_filter_t, optinsn_t, udc_filter_t) only act on
microcode an existing backend already produced.  So this is a standalone
pipeline that uses IDA's SPU processor module purely as a decoder:

    procs/spu.py  ->  lifter  ->  CFG  ->  SSA  ->  opt  ->  viewer

Stage 1 (this code) gets you an optimised SSA IR for a function.  Structuring
and C output build on top of it.

Usage from the IDA console::

    import spudec
    print(spudec.decompile(here()))
"""

__version__ = "0.1.0"

from .ir import Op, EW, Const, Var, Insn, Block, Function     # noqa: F401


def decompile(ea, optimize_ir=True, scalarize=True, drop_hints=True,
              check=False):
    """
    Lift the function containing ``ea`` and return the IR ``Function``.

    The pipeline is lift -> SSA -> optimise -> scalarise -> optimise.  The
    second optimise matters: scalarisation replaces four-instruction memory
    idioms with one instruction, and it is DCE that then removes the
    `loadq`/`shufb`/`genctl` and their address arithmetic.

    ``check=True`` additionally runs the SSA verifier; the problem list is
    attached as ``func.problems``.
    """
    from . import cfg, ssa, opt
    from . import scalarize as _scalarize

    func, lifter = cfg.build(ea, drop_hints=drop_hints)
    ssa.to_ssa(func)
    func.stats = {}
    func.scalar_stats = {}
    if optimize_ir:
        func.stats = opt.optimize(func)
    if scalarize:
        func.scalar_stats = _scalarize.scalarize(func)
        if optimize_ir:
            func.stats = opt.optimize(func)
            # Marking is cheap and the demand masks sharpen once the dead
            # idiom remnants are gone.
            from . import lanes
            func.demand = lanes.compute_demand(func)
            func.scalar_stats["scalars"] = _scalarize.mark_scalars(
                func, func.demand)
    func.unhandled = dict(lifter.unhandled)
    func.problems = ssa.verify(func) if check else []
    return func


def dump(ea, **kw):
    """Convenience: decompile and return the textual SSA IR."""
    return decompile(ea, **kw).dump()


def structured(func):
    """Structure an already-decompiled function: returns (stmts, info)."""
    from . import structure
    stmts, info = structure.structure(func)
    func.structure_info = info
    return stmts, info


def decompile_all(name_of=None, progress=None, funcs=None, **kw):
    """
    Decompile every function in the database into one listing -- the
    equivalent of Hex-Rays' "Decompile all" (Ctrl+F5).

    ``progress(done, total, name)`` is called before each function and may
    return False to cancel; whatever was produced so far is still returned.

    A function that fails is reported inline as a comment rather than
    silently skipped: a listing that quietly omits code is worse than one
    that admits a gap.

    Returns ``(lines, stats)``.
    """
    import ida_funcs
    import ida_name
    import ida_nalt

    if name_of is None:
        def name_of(ea):
            return ida_name.get_name(ea) or ("sub_%X" % ea)

    if funcs is None:
        funcs = sorted(ida_funcs.getn_func(i).start_ea
                       for i in range(ida_funcs.get_func_qty()))

    stats = dict(functions=0, failed=0, problems=0, blocks=0, insns=0,
                 loops=0, gotos=0, stores=0, loads=0, scalars=0,
                 unreachable=0, cancelled=False)
    unhandled = {}
    failures = []
    body = []

    for n, ea in enumerate(funcs):
        nm = name_of(ea)
        if progress is not None and not progress(n, len(funcs), nm):
            stats["cancelled"] = True
            break
        try:
            func = decompile(ea, check=True, **kw)
            lines = pseudocode(func, name_of=name_of)
        except Exception as exc:
            stats["failed"] += 1
            failures.append((ea, nm, "%s: %s" % (type(exc).__name__, exc)))
            body.append("")
            body.append("//" + "-" * 74)
            body.append("// %s @ 0x%X -- DECOMPILATION FAILED" % (nm, ea))
            body.append("//   %s: %s" % (type(exc).__name__, exc))
            body.append("//" + "-" * 74)
            continue

        info = func.structure_info
        sc = func.scalar_stats
        stats["functions"] += 1
        stats["problems"] += len(func.problems)
        stats["blocks"] += len(func.blocks)
        stats["insns"] += sum(len(b.insns) for b in func.blocks)
        stats["loops"] += info.get("loops", 0)
        stats["gotos"] += info.get("gotos", 0)
        stats["stores"] += sc.get("stores", 0)
        stats["loads"] += sc.get("loads", 0) + sc.get("aligned_loads", 0)
        stats["scalars"] += sc.get("scalars", 0)
        stats["unreachable"] += len(getattr(func, "unreachable", ()))
        for k, v in func.unhandled.items():
            unhandled[k] = unhandled.get(k, 0) + v

        body.append("")
        body.append("//" + "-" * 74)
        for ln in lines:
            body.append(ln)
        for u in getattr(func, "unreachable", ()):
            body.append("// NOT DECOMPILED: 0x%X..0x%X is unreachable from "
                        "the function entry" % u)

    stats["unmodelled"] = unhandled
    stats["failures"] = failures

    head = [
        "//",
        "// Decompiled by spudec %s" % __version__,
        "//   input      : %s" % (ida_nalt.get_input_file_path() or "?"),
        "//   functions  : %d decompiled, %d failed%s"
        % (stats["functions"], stats["failed"],
           "  (CANCELLED)" if stats["cancelled"] else ""),
        "//   blocks     : %d,  IR instructions: %d"
        % (stats["blocks"], stats["insns"]),
        "//   structure  : %d loops, %d gotos"
        % (stats["loops"], stats["gotos"]),
        "//   scalarised : %d stores, %d loads, %d scalar ops"
        % (stats["stores"], stats["loads"], stats["scalars"]),
    ]
    if stats["problems"]:
        head.append("//   SSA verifier problems: %d -- the affected functions "
                    "are suspect" % stats["problems"])
    if unhandled:
        head.append("//   unmodelled mnemonics: %s"
                    % ", ".join("%s x%d" % kv for kv in
                                sorted(unhandled.items(),
                                       key=lambda kv: -kv[1])))
    if stats["unreachable"]:
        head.append("//   %d unreachable range(s) reported inline and NOT "
                    "decompiled" % stats["unreachable"])
    head.append("//")

    if failures:
        head.append("// Functions that failed:")
        for ea, nm, msg in failures[:40]:
            head.append("//   0x%-8X %s  (%s)" % (ea, nm, msg))
        if len(failures) > 40:
            head.append("//   ... and %d more" % (len(failures) - 40))
        head.append("//")

    return head + body, stats


def pseudocode(ea, name_of=None, **kw):
    """
    Full pipeline: lift, SSA, optimise, scalarise, structure, render.

    Returns the pseudocode as a list of lines.  The SSA IR it was rendered
    from is untouched, so ``dump()`` on the same address still shows the
    verifiable three-address form.
    """
    from . import cgen
    func = ea if isinstance(ea, Function) else decompile(ea, **kw)
    stmts, info = structured(func)
    return cgen.generate(func, stmts, info, name_of=name_of)
