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
