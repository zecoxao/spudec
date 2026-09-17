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
            func.demand = lanes.compute_demand(func,
                                               callee=param_demand)
            func.scalar_stats["scalars"] = _scalarize.mark_scalars(
                func, func.demand)
    func.unhandled = dict(lifter.unhandled)
    func.problems = ssa.verify(func) if check else []
    return func


def dump(ea, **kw):
    """Convenience: decompile and return the textual SSA IR."""
    return decompile(ea, **kw).dump()


_ARITY_CACHE = {}
_ARITY_ACTIVE = set()
_ARITY_CUTS = 0        # cycles cut while answering the current query


def clear_arity_cache():
    """Forget recovered arities -- call after changing function boundaries."""
    _ARITY_CACHE.clear()
    _ARITY_ACTIVE.clear()
    global _ARITY_CUTS
    _ARITY_CUTS = 0


def arity_of(ea):
    """
    How many parameters the function at ``ea`` takes, or None if unknown.

    Needed to show a call's arguments: the operand list of a call names every
    argument register because a callee *might* read them, so only the callee
    itself can say how many of those are real.

    Demand-driven and recursive: working out one function's arity consults its
    own callees, so a parameter that is merely forwarded down a chain is still
    recovered.  Recursion terminates on an in-progress set rather than by
    caching a placeholder -- caching one would make the answer depend on which
    function happened to be asked about first.  Memoisation keeps the total
    work linear in the number of functions.

    The cache is keyed on address, so clear it if function boundaries change.
    """
    global _ARITY_CUTS
    if ea in _ARITY_CACHE:
        return _ARITY_CACHE[ea]
    if ea in _ARITY_ACTIVE:
        # A cycle.  Report "unknown" for this query and record that the answer
        # now being computed rests on a cut edge.
        _ARITY_CUTS += 1
        return None

    cuts_before = _ARITY_CUTS
    _ARITY_ACTIVE.add(ea)
    try:
        import ida_funcs
        f = ida_funcs.get_func(ea)
        if f is None or f.start_ea != ea:
            return None
        from . import cgen
        func = decompile(ea, scalarize=False)
        n = len(cgen._params(func, arity_of))
    except Exception:
        return None
    finally:
        _ARITY_ACTIVE.discard(ea)

    # Only cache an answer that did not depend on a cut cycle.  A result
    # computed while some caller up the stack was still in progress is a
    # *lower bound*, not the answer -- caching it would freeze in whatever
    # the traversal order happened to produce, and the same function would
    # then get different signatures depending on what was decompiled first.
    if _ARITY_CUTS == cuts_before:
        _ARITY_CACHE[ea] = n
    return n


_PARAM_DEMAND_CACHE = {}
_PARAM_DEMAND_ACTIVE = set()


def clear_param_demand():
    """Forget recovered parameter demand -- after boundaries change."""
    _PARAM_DEMAND_CACHE.clear()
    _PARAM_DEMAND_ACTIVE.clear()


def param_demand(ea):
    """
    What the function at ``ea`` reads of each of its parameter registers.

    ``{reg: byte mask}``, the mask being the same 16-bit lane mask
    :mod:`lanes` uses.  This is what stops a call's conservative operand list
    from telling the demand analysis that every argument register is read
    whole; see :func:`lanes._abi_contrib`.

    Demand-driven and recursive, like :func:`arity_of`: working out what one
    function reads of a parameter consults its own callees, so a parameter
    merely forwarded down a chain is resolved.  A cycle, an unresolvable
    address, or any failure answers ALL for every register -- the direction
    that cannot make the output claim more than it knows.
    """
    from . import lanes, regs
    wide = {r: lanes.ALL for r in range(regs.ARG_FIRST, regs.ARG_LAST + 1)}
    if ea in _PARAM_DEMAND_CACHE:
        return _PARAM_DEMAND_CACHE[ea]
    if ea in _PARAM_DEMAND_ACTIVE:
        return wide                      # a cycle: stay conservative
    _PARAM_DEMAND_ACTIVE.add(ea)
    try:
        import ida_funcs
        f = ida_funcs.get_func(ea)
        if f is None or f.start_ea != ea:
            return wide
        func = decompile(ea, scalarize=False)
        dem = lanes.compute_demand(func, callee=param_demand)
        out = {}
        for r in range(regs.ARG_FIRST, regs.ARG_LAST + 1):
            out[r] = dem.get((r, 0), lanes.NONE)
    except Exception:
        return wide
    finally:
        _PARAM_DEMAND_ACTIVE.discard(ea)
    _PARAM_DEMAND_CACHE[ea] = out
    return out


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
    import ida_nalt

    if name_of is None:
        name_of = names()

    if funcs is None:
        funcs = sorted(ida_funcs.getn_func(i).start_ea
                       for i in range(ida_funcs.get_func_qty()))

    stats = dict(functions=0, failed=0, problems=0, blocks=0, insns=0,
                 loops=0, gotos=0, stores=0, loads=0, scalars=0,
                 unreachable=0, strings=0, cancelled=False)
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
        stats["strings"] += (getattr(func, "type_stats", None)
                             or {}).get("strings", 0)
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
        "//   strings    : %d constant(s) resolved to string literals"
        % stats["strings"],
        "//   names      : %d C++ symbol(s) demangled" % names().demangled,
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


def strings():
    """The shared constant-address-to-string-literal resolver (see data.py)."""
    from . import data
    return data.strings()


def names():
    """The shared function-name resolver: demangled and unambiguous."""
    from . import data
    return data.names()


def clear_caches():
    """
    Forget recovered strings and names.

    Call after retyping data or renaming functions in the database; the
    resolvers cache per session so the whole-database pass stays fast.
    """
    from . import data
    data.clear()
    clear_arity_cache()
    clear_param_demand()


def pseudocode(ea, name_of=None, arity=None, str_of=None, **kw):
    """
    Full pipeline: lift, SSA, optimise, scalarise, structure, render.

    Returns the pseudocode as a list of lines.  The SSA IR it was rendered
    from is untouched, so ``dump()`` on the same address still shows the
    verifiable three-address form.

    ``str_of`` resolves a constant address to a C string literal; it defaults
    to the shared database-backed resolver, and passing ``lambda ea: None``
    turns the feature off.
    """
    from . import cgen
    func = ea if isinstance(ea, Function) else decompile(ea, **kw)
    stmts, info = structured(func)
    if arity is None:
        arity = arity_of
    if str_of is None:
        str_of = strings()
    if name_of is None:
        name_of = names()
    return cgen.generate(func, stmts, info, name_of=name_of, arity_of=arity,
                         str_of=str_of)
