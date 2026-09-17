"""
Recognise the ABI frame boilerplate, so the listing can state it instead of
spelling it out.

Every non-leaf SPU function opens by saving the link register and whichever of
r80..r127 it intends to use, and by writing the caller's stack pointer into
the new frame's back-chain slot.  In the 23-module corpus that is a tenth of
every line of output::

    *(qword *)(((char *)sp - 0x10) & 0x3FFF0) = r80;
    *(qword *)(((char *)sp - 0x20) & 0x3FFF0) = r81;
    ...
    *(qword *)((sp + 0x10) & 0x3FFF0) = lr;
    *(qword *)((sp - 0x13B0) & 0x3FFF0) = sp;

None of it says anything about what the function does -- it is the calling
convention, repeated once per function -- and it is noise in exactly the place
a reader starts reading.

The matching reloads are usually gone already: nothing reads a restored
callee-saved register, so DCE deletes them, which is why the saves dominate.

What makes a store recognisable as a save rather than as data:

* it stores the *live-in* value of a callee-saved register, or of lr.  At
  entry those hold the caller's values, which this function has no other
  use for, so storing one can only be preserving it.
* its address is built from the incoming stack pointer.
* nothing in the function loads from that slot.  A slot that is read back is
  carrying data, whatever it looks like, and stays in the listing.

That last condition is what keeps this from hiding real work, and it is
checked against every load in the function, not just the obvious ones.  It
does not cover a pointer *into* the save area handed to a callee, which could
read a slot from the outside.

Two things make that acceptable.  Suppression is presentational only -- the IR
keeps the stores, so anything reasoning about memory still sees them -- and
:func:`describe` states every save in the function's header rather than
summarising a count, so a reader is told exactly what was hidden and where.
"""

from .ir import Op
from . import regs
from .scalarize import addr_expr


class Frame(object):
    """What the prologue of one function does."""

    def __init__(self):
        self.saves = []          # [(offset, register number)]
        self.back_chain = None   # offset of the back-chain slot, or None
        self.size = None         # frame size in bytes, or None
        self.hidden = set()      # id() of every instruction not to print

    def __bool__(self):
        return bool(self.saves) or self.back_chain is not None

    __nonzero__ = __bool__       # py2 spelling, harmless here

    def describe(self):
        """
        The header line, or None when there was no boilerplate to hide.

        Offsets are printed as the source spells them -- relative to the
        incoming stack pointer -- because that is what the disassembly shows.
        A run of consecutive registers in consecutive slots collapses to
        ``r80-r87 at sp-0x10..sp-0x80``: spelling out eight of those was
        longer than the prologue it replaced.
        """
        if not self:
            return None
        parts = []
        if self.size:
            parts.append("frame 0x%X" % self.size)
        if self.saves:
            parts.append("saves " + ", ".join(self._runs()))
        if self.back_chain is not None:
            parts.append("back chain at %s" % _slot(self.back_chain))
        return "prologue: " + "; ".join(parts)

    def _runs(self):
        """The saves, with consecutive register/slot runs collapsed."""
        out = []
        for run in _group(sorted(self.saves, key=lambda s: s[1])):
            (o0, r0), (o1, r1) = run[0], run[-1]
            if len(run) == 1:
                out.append("%s at %s" % (regs.reg_name(r0), _slot(o0)))
            else:
                out.append("%s-%s at %s..%s"
                           % (regs.reg_name(r0), regs.reg_name(r1),
                              _slot(o0), _slot(o1)))
        return out


def _slot(off):
    """An offset as ``sp-0x10`` / ``sp+0x10``."""
    return "sp%s0x%X" % ("-" if off < 0 else "+", abs(off))


def _group(saves):
    """
    Split ``(offset, reg)`` pairs, sorted by register, into runs.

    A run is consecutive registers in consecutive quadword slots, in the same
    direction -- which is what a compiler-emitted save sequence looks like and
    what makes ``r80-r87`` a truthful abbreviation.  Anything else stays its
    own group rather than being folded into a range that would imply slots
    nothing was saved in.
    """
    runs = []
    for item in saves:
        if runs:
            po, pr = runs[-1][-1]
            if item[1] == pr + 1 and item[0] - po == -0x10:
                runs[-1].append(item)
                continue
        runs.append([item])
    return runs


def signed_offset(off):
    """An offset as the address arithmetic means it, not as 32 bits."""
    off &= 0xFFFFFFFF
    return off - (1 << 32) if off >= (1 << 31) else off


def analyse(func):
    """
    Find the frame boilerplate in ``func``.  Returns a :class:`Frame`.

    Runs on the SSA IR after optimisation, so ``addr_expr`` can resolve a
    store's address through the align mask and the offset adds the lifter
    emits for stqd.
    """
    fr = Frame()
    defs = {}
    for i in func.insns():
        d = i.defines()
        if d is not None:
            defs[d.key()] = i

    sp_in = (regs.SP, 0)

    # Every slot any load reads.  A slot that is read back holds data, not a
    # saved register, however much the store looks like a save.
    read = set()
    for i in func.insns():
        if i.op in (Op.LOAD, Op.LOADQ, Op.LOADU) and len(i.srcs) > 1:
            base, off, _ = addr_expr(i.srcs[1], defs)
            read.add((base, off))

    # The frame size, from the one instruction that moves the stack pointer.
    for i in func.insns():
        d = i.defines()
        if d is None or d.reg != regs.SP or d.ver == 0:
            continue
        if i.op == Op.ADD and len(i.srcs) == 2:
            a, b = i.srcs
            k = b.word0 if b.is_const else (a.word0 if a.is_const else None)
            other = a if b.is_const else b
            if k is not None and other.is_var and other.key() == sp_in:
                delta = signed_offset(k)
                if delta < 0:
                    fr.size = -delta

    for i in func.insns():
        if i.op != Op.STOREQ or len(i.srcs) < 3:
            continue
        val = i.srcs[2]
        if not val.is_var or val.ver != 0:
            continue
        base, off, _ = addr_expr(i.srcs[1], defs)
        if base != sp_in or (base, off) in read:
            continue
        off = signed_offset(off)
        if val.reg == regs.SP:
            fr.back_chain = off
        elif val.reg == regs.LR or regs.is_callee_saved(val.reg):
            fr.saves.append((off, val.reg))
        else:
            continue
        fr.hidden.add(id(i))

    # A register saved on two paths is still one saved register.
    fr.saves = sorted(set(fr.saves), key=lambda s: s[0])
    return fr
