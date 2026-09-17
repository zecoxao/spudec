"""
Stack slots as named locals, taken from IDA's own frame analysis.

`spu.py` tracks the stack pointer (``add_func_auto_stkpnt``) and creates a
frame member for every sp-relative displacement it decodes, so an SPU database
already knows which slot each ``lqd``/``stqd`` touches -- including any name
the user gave it by hand.  Rebuilding that here would mean re-deriving IDA's
sp-delta bookkeeping and then disagreeing with the disassembly the moment the
user renamed something, so this asks IDA instead::

    *(qword *)((sp_2 + 0x30) & 0x3FFF0) = *(qword *)((sp_2 + 0x30) & 0x3FFF0) & r5;

becomes::

    var_30 = var_30 & r5;

Naming a slot asserts that the address really is that slot and nothing else
reaches it.  Two rules keep that honest:

* the operand has to be one IDA itself resolved to a frame member.  A computed
  address that merely happens to land in the frame is not named.
* every access to a slot must be the same width.  Real code does read a
  quadword slot as a word, and one name cannot stand for both without
  implying a type the code never uses, so a slot accessed two ways keeps the
  explicit dereference everywhere -- see :meth:`CGen._stack_name`.

Imports IDA lazily, so the rest of the pipeline stays testable without a
database.
"""


class StackVars(object):
    """
    Resolves an instruction address to the frame member it addresses.

    ``__call__(ea)`` answers ``(name, size)`` or None.  Cached per address:
    the whole-database pass renders every instruction of every function, and
    decoding each one again to ask IDA would dominate the run.
    """

    def __init__(self):
        self._cache = {}
        self._members = {}      # function start -> {byte offset: (name, size)}
        self.named = 0
        self._off = False       # no database, or a bug already reported
        self.error = None

    def _frame(self, pfn):
        """IDA's frame members for ``pfn``, by byte offset."""
        key = pfn.start_ea
        if key in self._members:
            return self._members[key]
        out = {}
        import ida_frame
        import ida_typeinf
        tif = ida_typeinf.tinfo_t()
        # A function without a frame is ordinary -- a leaf that needs no
        # locals has none -- so an empty answer is a fact, not a failure.
        if ida_frame.get_func_frame(tif, pfn):
            udt = ida_typeinf.udt_type_data_t()
            if tif.get_udt_details(udt):
                # Member offsets are in bits, and in the same space as
                # `calc_stkvar_struc_offset` answers in.
                for m in udt:
                    out[m.offset // 8] = (m.name, m.size // 8)
        self._members[key] = out
        return out

    def __call__(self, ea):
        if ea in self._cache:
            return self._cache[ea]
        if self._off:
            return None
        try:
            out = self._lookup(ea)
        except ImportError:
            # No database: the feature is simply unavailable.
            self._off = True
            return None
        except AttributeError as exc:
            # A missing IDA symbol is a bug here, not a fact about the
            # database, and swallowing it once cost a whole feature: this
            # module looked for UA_MAXOP in ida_ua, where it does not live,
            # so every lookup raised and every slot went unnamed while the
            # listing quietly rendered as though there were no frame at all.
            # Fail loudly the first time, then stay out of the way.
            self._off = True
            self.error = "spudec/stack.py: %s" % exc
            raise
        self._cache[ea] = out
        if out is not None:
            self.named += 1
        return out

    def _lookup(self, ea):
        import ida_funcs
        import ida_ida
        import ida_ua
        import ida_frame
        from ida_idaapi import BADADDR

        pfn = ida_funcs.get_func(ea)
        if pfn is None:
            return None
        insn = ida_ua.insn_t()
        if ida_ua.decode_insn(insn, ea) <= 0:
            return None
        members = self._frame(pfn)
        for n in range(ida_ida.UA_MAXOP):
            op = insn.ops[n]
            if op.type == ida_ua.o_void:
                break
            if op.type != ida_ua.o_displ:
                continue
            soff = ida_frame.calc_stkvar_struc_offset(pfn, insn, n)
            if soff == BADADDR:
                continue
            hit = members.get(soff)
            if hit is not None and hit[0]:
                return hit
            # IDA resolved the operand to a frame offset but the frame has no
            # member there (it can lag a reanalysis).  Its own generated name
            # is still the name the disassembly shows.
            name = ida_frame.build_stkvar_name(pfn, soff)
            if name:
                return (name, 0)
        return None


_STACK = None


def stackvars():
    global _STACK
    if _STACK is None:
        _STACK = StackVars()
    return _STACK


def clear():
    global _STACK
    _STACK = None
