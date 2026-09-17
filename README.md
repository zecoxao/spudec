# spudec — SPU lifter + SSA IR for IDA Pro 9.x

Stage 1 of an SPU decompiler: IDA's SPU processor module is used purely as a
decoder, and everything above it is ours.

```
procs/spu.py -> lifter -> CFG -> SSA -> opt -> scalarise -> opt
             -> structure -> pseudocode
```

## Why not Hex-Rays

Hex-Rays cannot be extended to a new architecture from the public SDK.
Microcode generation lives inside the closed per-processor backends
(`hexarm.dll`, `hexppc.dll`, `hexmips.dll`, `hexarc.dll`, `hexrv.dll`,
`hexv850.dll`, `hexx64.dll`). The exposed hooks — `microcode_filter_t`,
`optinsn_t`/`optblock_t`, `udc_filter_t` — all operate on microcode an existing
backend already produced, and `init_hexrays_plugin()` refuses outright on a
processor with no backend. There is no `register_backend_for_processor()`.

So this is standalone. `F5` will never light up on an SPU database; this gives
you the IR that a decompiler is built on, in its own viewer.

## Install

Copy `spudec_plugin.py` **and** the `spudec/` directory side by side into:

```
%APPDATA%\Hex-Rays\IDA Pro\plugins\
```

| key | what |
|---|---|
| <kbd>Ctrl-Shift-S</kbd> | decompile the function under the cursor |
| <kbd>Ctrl-F5</kbd> | decompile everything in the database |
| <kbd>i</kbd> (in the viewer) | toggle pseudocode / the SSA IR it came from |
| double-click | jump the disassembly to that line's address |

Both viewers syntax-highlight the C (`highlight.py`), token by token rather
than line by line — a line like

```c
r4 = *(unsigned int *)(r5 + 0x10);   // was lqd/rotqby
```

has a type, a variable, a number and a comment in it, and colouring the whole
thing one colour tells the reader nothing. Colours are IDA's own `SCOLOR_*`
tags rather than fixed RGB, so the output follows whatever theme is set instead
of turning black-on-black in a dark one. Verified over the whole metldr
listing: 11859 lines, every one balanced and byte-identical after
`tag_remove`.

<kbd>Ctrl-F5</kbd> mirrors Hex-Rays' "Decompile all": every function into one
listing, with a progress box you can cancel (a cancelled run still returns
what it produced), then a prompt to save it as a `.c` file. Past 60k lines it
offers the file only — a custom viewer holding a few hundred thousand lines is
slow to build and painful to scroll.

From the IDA console:

```python
import spudec
print("\n".join(spudec.pseudocode(here())))   # structured pseudocode
print(spudec.dump(here()))                    # optimised SSA IR
print(spudec.dump(here(), optimize_ir=False)) # raw lifting
func = spudec.decompile(here(), check=True)   # func.problems = verifier output

lines, stats = spudec.decompile_all()         # the whole database
open("out.c", "w").write("\n".join(lines))
```

The whole-database listing opens with a header giving the counts that matter —
functions decompiled and failed, loops, gotos, what was scalarised, any SSA
verifier problems, any unmodelled mnemonics. A function that fails to decompile
is reported inline as a comment rather than skipped, and unreachable ranges are
marked where they would have appeared: a listing that quietly omits code is
worse than one that admits a gap. metldr (189 functions) takes 1.8s and comes
out at about 10k lines.

## Patched `spu.py`

Two fixes to IDA's SPU processor module ship alongside this. Install the
patched copy to `%APPDATA%\Hex-Rays\IDA Pro\procs\spu.py` — IDA reads
processor modules from there, so it needs no administrator rights and leaves
the stock file in Program Files untouched.

**Endianness.** The SPU is big-endian, and `spu.py` tried to say so from
`notify_init()`. That never ran: `processor_t` forwards the legacy `notify_*`
names for only a few events (`newprc`, `newfile`, `oldfile`), and `init` is not
one of them. The call was also written `ida_ida.cvar.inf_set_be(True)`, which
is IDA 7.x spelling and would have raised even if it had been reached. And
setting it at init alone would not stick anyway, because the loader runs
afterwards. Now set from `ev_init`, `ev_newprc`, `ev_newfile` and `ev_oldfile`.
Flat SPU binaries went from ~57% of words decoding to a valid opcode to 100%,
with no manual forcing.

**Switch recognition.** A `bi` through a jump table produced no code
references, so every case body was orphaned — and the table itself got swept
into code, since a table of local-store addresses has 0x00 in its top byte and
opcode 0 is `stop`. The dispatch GCC emits is:

```
clgti  $c, $idx, N        bounds check
brnz   $c, default
shli   $i, $idx, 2
ila    $b, TABLE
a      $a, $i, $b
lqx    $x, $i, $b
rotqby $t, $x, $a
bi     $t
```

matched backwards from the `bi`. Two things this needed: the kernel does not
call `ev_is_switch` for this module, so the table is built from `ev_emu_insn`
via `create_switch_table`/`create_switch_xrefs`; and the lookup has to be
position-aware, because the idiom reuses registers (`rotqby r2, r2, r3`) and a
plain register→definition map resolves an instruction's own operand to itself.

Matching is deliberately strict — no bounds check means no recognised switch.
A false positive would invent code references and corrupt the flow graph,
where a false negative only leaves things as they were.

Across the corpus this recovered **+1057 blocks and +12490 IR instructions**
of code that was previously dropped without a word.

## Two things to know about `spu.py`

**`itype_*` constants are not stable.** They are assigned by iterating dict
values at module init, so they shift if the tables are edited and cannot be
hardcoded. The lifter resolves mnemonics through `ida_idp.ph_get_instruc()` at
runtime and dispatches on the name.

**Stock SPU databases come up little-endian** on flat binary loads — see the
patched `spu.py` above, which fixes it. Without the patch, work around it
before analysing:

```python
import ida_ida, ida_bytes, ida_auto
ida_ida.inf_set_be(True)
ida_bytes.del_items(0, ida_bytes.DELIT_SIMPLE, 0x400)   # then re-create code
ida_auto.auto_wait()
```

## Design

**128-bit values throughout.** The SPU has no scalar register file. Every IR
value is a 128-bit bitvector and every operation carries an element-width tag.
Byte 0 is the most significant; word 0 (bits 127..96) is the preferred slot.
Recovering "this quadword is really an `int`" is a later pass's job, not the
lifter's.

**Memory and channels are registers.** `mem`, `ch` and `spr` are pseudo
registers threaded through SSA, so a load names exactly which store it is
ordered after and the normal SSA machinery, propagation and DCE apply to them
for free.

**Everything lifts to something.** Instructions without hand-written semantics
become `INTRINSIC` with correct defs and uses. Silently dropping one would
corrupt the SSA. Currently 195/199 mnemonics are modelled; `bisled`, `dftsv`,
`fscrrd` and `fscrwr` fall back.

**Semi-pruned SSA.** Only registers with an upward-exposed use get phis; the
lifter emits many single-block temporaries and full placement would bury the
output. The cost is the occasional redundant phi, which DCE removes.

**Immediate shift counts are normalised.** SPU encodes right shifts with a
negated count (`rotmi rt,ra,-3` shifts right by 3). With a literal count that is
proved away at lift time and emitted as a plain `shr`/`sar`.

**Calls and returns carry the ABI register set** (`r3..r74`, `lr`, `mem`, `ch`)
as operands, so DCE cannot delete argument setup or return-value computation.
They print as `[abi: N regs]`.

## Scalarisation

The SPU can only load and store aligned quadwords, so compilers build scalar
accesses out of idioms. `scalarize.py` recovers them, driven by the byte-level
demand analysis in `lanes.py`.

**Demand analysis.** For every SSA value, a 16-bit mask of which *bytes*
anything reads. Bitwise ops are byte-exact; arithmetic expands to the whole
element; `shufb` and the quadword rotates are exact byte permutations whenever
their control is a known constant, which after folding is the usual case.
A value whose demand fits in `0x000F` is a scalar in the preferred slot, and
the operations producing it print as ordinary arithmetic (`r90#2 = r34#3 & 0xFF`
rather than `and.w`).

**Scalar store** — `cwd`/`lqd`/`shufb`/`stqd` collapses to one `store.ew`.

**Scalar load** — `lqd`/`rotqby` collapses to one `load.ew`, with the width
taken from demand: the rotate leaves the value at byte 0, and how many bytes
anything reads from there *is* the access size. When demand is wider than any
scalar (the result feeds a runtime shuffle, say) it stays an honest `loadu`.

Two things real code forced, neither of which a textbook pattern would have:

- **The control-mask address may differ from the access address.** `cwd`/`cbd`
  only consume `addr & 0xF`, so a compiler emits `lqd r40, 0x400(r33)` beside
  `cbd r41, 0(r33)` — 0x400 is a multiple of 16, so the byte position is
  identical. Requiring the offsets to be equal misses every one of those.
- **The read-modify-write may not be airtight.** The load and the write-back
  can sit on different memory states. `scalarize` walks the memory chain and
  proves the intervening writes disjoint where it can (same base, offsets ≥ 16
  apart). Where it cannot, it still recovers the store — such a sequence would
  be *discarding* the other write, which is a miscompile if they really alias —
  but says so in the instruction comment rather than assuming silently.
  `scalarize(func, strict=True)` refuses those instead.

**Byte placement is asymmetric below word width**, and the IR says so rather
than pretending otherwise: `LOAD.ew` leaves the value in bytes 0..ew-1 (where
`rotqby` actually puts it), `STORE.ew` takes it from the low ew bytes of the
preferred slot (where `cwd`+`shufb` actually reads it). At word width — the
common case — the two coincide.

**Calling convention.** Calls read r3..r10 and clobber memory; returns hand
back r3..r10. Both are knobs (`Lifter(n_arg_regs=…, n_ret_regs=…)`). The full
ABI range is r3..r74, but assuming a call reads all 72 makes every scratch
register in the function immortal and buries the listing — eight registers is
128 bytes of arguments, past which the ABI passes by hidden pointer.

## Control-flow structuring

Structural analysis in the Cifuentes tradition: loops come from back edges in
the dominator tree, conditionals end at their immediate post-dominator.
Anything that does not fit becomes an explicit `goto` and a label — structuring
never silently drops an edge, and the goto count is reported so you know when
it happened.

**Shared tails are duplicated rather than jumped to.** Measured across the
corpus, 94% of gotos came from one cause: a block reached a second time. The
commonest shape by far is a small tail that one arm of a conditional emitted
inline and the other can only jump to, because their join point sits further
out than the tail does. Emitting such a block twice is semantically free —
only one path runs it — so `may_duplicate` does that, bounded three ways
(small blocks only, a cap per block, a budget per function) so it cannot blow
up. Loop headers are never duplicated; that is what `continue` is for. See the
measured trade-off at the top of `structure.py`.

**Every loop is emitted as endless first**, with `break`/`continue` inserted
wherever control reaches the follow or the header, and a refinement pass then
recognises `while` and `do-while`. Doing it in that order means a self-loop, a
header that carries statements, and a multi-latch loop all fall out of the same
code instead of needing three special cases. When the test is not isolated at a
boundary the loop just stays endless with an explicit break — less pretty, still
correct.

Two things worth knowing, both found by running against real code:

- **A self-loop's body is the header alone.** Seeding the backward walk at the
  latch when latch *is* the header drags the loop's entry block, and everything
  before it, into the body — which puts the follow in the wrong place and emits
  the code *after* the loop inside it.
- **A loop header almost always carries a phi.** Treating that as a statement
  stops every top-tested loop from being recognised as a `while`, so phi-only
  blocks count as markers, like labels.

## Pseudocode

`cgen.py` renders the structured AST. It is a *rendering* — the SSA IR is
untouched, so `dump()` still shows the verifiable three-address form and the
verifier still applies to it.

- **Phi webs are coalesced.** Structured control flow already expresses the
  selection, so phis disappear. Distinct webs of one register get `_2`/`_3`
  suffixes rather than being conflated, so two unrelated uses of `r33` never
  silently become one variable.
- **Single-use pure definitions inline into their use site**, which is what
  turns `t17 = r33 + 0x400` / `store.b mem, t17, r34` into
  `*(u8 *)(r33 + 0x400) = r34;`. Loads only inline when no memory write sits
  between definition and use, so nothing reorders across a store.
- **Vector operations print as calls**, not as expressions. Pretending `shufb`
  is C would be a lie; scalar operations — the ones demand analysis proved live
  only in the preferred slot — print as ordinary arithmetic.

## Tests

The harnesses live in `spudec-dev/` beside this repository, so the repo holds
only what you install into IDA. They locate the package in the sibling repo
automatically and write their output beside themselves, never in here.

```
cd ../spudec-dev
python selftest.py                       # ir / sem / ssa / opt, no IDA needed
python romscan.py <blob>                 # endianness probe + lifter coverage
```

`romscan.py` parses the opcode tables straight out of `procs/spu.py`, so the
mnemonic mapping cannot drift from what IDA actually decodes. It probes both
endiannesses at all four offsets — useful, since SPU dumps are often
byte-swapped and the giveaway is the valid-opcode fraction (real SPU code hits
100%; the big-endian-misread baseline is ~57%).

Headless integration test over a real blob:

```
idat.exe -c -A -pspu -T"Binary file" -S"ida_rom_test.py" rom_be.bin
```

Last run — 1 KB SPU boot ROM, 256 instructions, 6 functions, 48 blocks,
275 IR instructions: **0 SSA verifier problems, 0 unmodelled mnemonics**,
50 operations scalarised, 8 loops structured, **0 gotos**.

## Cross-check against GhidraSPU

The [GhidraSPU](https://github.com/aerosoul94/GhidraSPU) SLEIGH module is an
independent encoding of the same ISA, so it is a useful second opinion. Of its
204 instruction definitions: **38 have empty bodies** (every float operation —
`fa`, `fs`, `fm`, `fma`, all doubles, all int/float conversions — plus hints
and nops) and **81 are opaque `pcodeop`s** the decompiler cannot fold, including
`shufb`, every `cwd`/`cbd`/`chd`/`cdd` control generator, every quadword
rotate, all the multiplies, `clz`, `cntb`, `gb` and the byte/halfword compares.
85 (42%) carry semantics Ghidra can reason about.

Because `cwd` and `shufb` are both opaque there, the scalar-store idiom cannot
be recovered at all — which is presumably why their `LSA` operand drops the
`& 0xFFFFFFF0` alignment mask outright, with the comment *"we know that the LSA
must be 16 bytes aligned, but it f\*s up the decompiler output"*. That trades
correctness for readability. Keeping the mask and recovering the idiom, as
`scalarize.py` does, gets both.

Points where the two agree, which is worth something: `sf`/`sfi` operand order,
`addx` reading RT, `iohl` reading RT, `rotmi`'s negated-count normalisation,
and `brhz` testing the *low* halfword. Where they differ, mine was checked
against the ISA document and a round-trip property test — their `fsm` inverts
its conditional for three of four words and shifts past the register width.

Three real improvements came out of the comparison:

- **Call clobbering** (`abi.py`). Their `spu.cspec` lists `<unaffected>` as
  exactly r80..r127, confirming the volatile set. A register written before a
  call and read after it used to keep the same SSA name, so constant
  propagation folded the stale value across the call — confidently wrong
  output. Now each call defines the volatile registers it destroys.
- **Local-store address masking.** Effective addresses wrap within the 256 KB
  local store, so the mask is `0x3FFF0`, not `~0xF`.
- Their cspec lists arguments and returns as r3..r74. Measured on the ROM,
  using the full range costs **+25% IR** (389 vs 310 instructions) with no
  readability gain, so the default stays at r3..r10 with `n_arg_regs` /
  `n_ret_regs` to widen it. The wide set is *correct*; the narrow one is a
  documented heuristic.

Two bugs of my own surfaced while doing this: `CALL`/`ICALL` were marked as
terminators (they fall through), so a call last in a block was sliced off with
the terminator and vanished from the output; and `biz`/`binz`/`bihz`/`bihnz`
were lifted as an *unconditional* indirect jump, turning a conditional branch
into an unconditional one. Both are fixed and covered by tests.

## Type recovery

The SPU hands over an unusual amount of type evidence, so `types.py` is
constraint propagation over the SSA graph rather than guesswork. Every rule is
anchored to an instruction that can only mean one thing:

| evidence | from |
|---|---|
| **width** | demand analysis — byte 3 alone is a `char`, bytes 0..3 an `int` |
| **signedness** | the opcode: `cgt` vs `clgt`, `rotma` vs `rotm`, `xsbh`/`xshw`/`xswd` |
| **float** | `fa`/`fm`/`dfa` and the four conversion instructions |
| **pointers** | use as an address — the operand of a `load.w` points at 4 bytes |

Width from demand is the part no other architecture gives you for free. It
comes straight out of the analysis the scalarisation pass already needed.

Two judgement calls worth stating:

**A demand mask of `ALL` is the default, not evidence.** Every call and return
carries an ABI operand list that demands all sixteen bytes, because a callee
might read them. Treating that as "this is a vector" made every value reaching
a return look like one. Only a mask *narrower* than ALL says anything.

**Scalar or vector is already answered.** `insn.scalar` means demand analysis
proved nothing reads outside the preferred slot. Re-deriving it here would be
strictly worse, so the inference just uses it.

Conflicting evidence is recorded, not resolved: the same bits read signed in
one place and unsigned in another is ordinary, so that is marked *ambiguous*
and rendered unsigned. A real contradiction — float evidence meeting pointer
evidence — is counted and reported in the function header, because it usually
means a lifting bug and hiding it wastes the signal.

That distinction mattered. The first version counted any mismatched kinds as a
contradiction and flagged 26 of metldr's 189 functions. They were all vector
meeting scalar — which on a machine where *every* register is 128 bits is not a
conflict at all: "the whole quadword is used" and "the preferred slot holds an
address" are routinely both true. Narrowing the rule to float-versus-pointer
took it to zero, leaving 11 genuinely ambiguous signedness cases.

### Parameters

The SPU ABI is positional — r3 is the first argument, r4 the second, and so on
— so arity comes from the *highest* argument register with a real use, and
every register below it is a parameter whether or not the function reads it.
An unused parameter is ordinary; taking only the registers actually read gave
signatures like `sub_0(qword r6)`, which cannot be what the caller sees.

Parameters print positionally (`a1`, `a2`, …) the way a real decompiler shows
them, with the register each came from in a header comment so nothing is lost:

```c
// parameters: a1 = r3  a2 = r4  a3 = r5
vec_uchar16 lv0::signed_elf::check_extended_header(qword a1, unsigned int a2,
                                                   vec_uint4 a3)
```

**What counts as a real use** is the whole difficulty. A use inside a call or
return's ABI operand list proves nothing: those lists name every argument
register because a callee *might* read them. Neither does a phi argument on its
own — it is only real if the phi's result is, which has to be propagated
backwards. Without that filter, `check_manu_revoked_version(void)` came out
with eight parameters because r9 and r10 were merely in scope across a call.

Measured against the 95 mangled C++ names in metldr, which state the true
prototype (and `this` for member functions): **75 exact, 19 under, 1 over.**
The under-counts are parameters a function only forwards to another call, whose
sole use is therefore an ABI list. Closing that needs the callee's arity —
an interprocedural fixpoint this per-function pipeline does not do. Erring
low is the safe direction: it shows fewer parameters than exist rather than
inventing ones, and nothing in the body is hidden either way.

### Everything else

Output gains a declarations block, a typed signature, `*p` instead of
`*(u32 *)(p)` where the pointer type is known, and scalar-spelled constants.
Registers the function reads but never writes, outside the range the ABI calls
arguments, are declared too and marked `// live in` — otherwise the listing
would use names it never declares. And each name in a pointer group carries its
own star, since `T *a, b;` makes only `a` a pointer:

```c
void __vector_Reset(void)
{
    unsigned int r14;

    r14 = 0x630;
    goto *r14;   // indirect / tail call
}
```

On metldr: 2102 integers, 2056 pointers, 5242 vectors across 189 functions,
129 of which get a non-`void` return type.

## Checked against a hand-written reference

`rom_pseudo_code.c` — a human annotation of this same boot ROM — was used as
ground truth. Three things came out of the comparison:

**Channels are named** (`channels.py`). The reference annotates every channel,
and it is the right call: channel I/O is how an SPU does DMA, mailboxes,
signals and synchronisation, so `wrch(21, 0x40)` hides exactly what a reader
needs. The numbers are architectural (CBEA SPU channel map), so naming them is
data, not a guess. PS3 isolation channels (64+) are labelled separately because
they are *not* architectural. A literal written to `MFC_Cmd` gets its DMA
opcode named too. Output now reads `rdch(SPU_RdMachStat)`,
`wrch(MFC_Cmd, 0x40); // GET` — matching the reference's annotations on
channels 3, 4, 13, 16–22, 64 and 66.

**Jumps out of a function were rendered as `return`.** IDA's flow chart emits a
zero-length stub block for a branch target outside the function, and these were
being given an implicit `return`. So the reference's `goto cleanup` at 0x018
came out as `return;` — wrong, and the kind of wrong that quietly changes what
the reader concludes. Now rendered as `goto loc_3A0; // outside this function`.

**Unreachable blocks were dropped silently.** Dominator computation needs a
single reachable entry, so blocks unreachable from it must be pruned — but
pruning them without a word means the listing omits real code. In this ROM the
DMA helper at 0x220 is only ever entered through an indirect branch, so it
vanished. `cfg.py` now records what it pruned on `func.unreachable`, and the
harness reports both the orphan blocks and a byte-level coverage figure.

It deliberately does *not* fabricate functions for them: splitting every orphan
block turned 0x1DC..0x23C into eighteen one-block "functions", several of which
are data (0x210 is the ShiftRows permutation table). Which orphans are real
entry points is a judgement for IDA and the analyst; the decompiler's job is to
say clearly that the code exists and was not analysed.

## Validated on real compiler output

`metldr.356.sym.elf` — 49 KB of C++ SPU code, 189 named functions
(`lv0::signed_elf`, `lv0::meta_secure_loader`, AES/ECDSA/SHA1) — is the real
workout, because compiler output leans on the scalar memory idioms far harder
than hand-written ROM code does. The ELF loader sets endianness from the
header, so the `notify_init` bug above only ever bites flat binaries.

```
189 functions, 972 blocks, 15486 IR instructions, 1.8s
0 failures, 0 SSA problems, 0 unmodelled mnemonics
145 loops, 84 gotos
201 scalar stores, 210 scalar loads, 21 aligned loads, 3592 scalar ops
```

Four things this found that the ROM never exercised:

- **`simplify_phis` stranded surviving phis.** Collapsing `phi(x, x)` to a
  `MOV` in place leaves later phis sitting behind a non-phi, breaking the
  invariant that phis come first — which structuring depends on to recognise
  top-tested loops. Touched blocks are re-canonicalised now. This was the
  source of all four SSA verifier problems.
- **The insertion mask often comes from a different register.** Compilers
  generate `cwd` from whatever known-aligned register is cheapest — real code
  pairs `stqd`/`lqd` on `r3` with `cwd $5, 0($sp)`, since the ABI keeps `sp`
  quadword-aligned so the mask is identical either way. `_slot_of` proves the
  byte position for literals and for anything built on `sp`.
- **The slot is often non-zero.** `lqd q, 0(r3)` + `rotqby v, q, $sp+8` reads
  the scalar at `align(r3) + 8`. That is exact — it is what the hardware
  reads, independent of whether `r3` is aligned — so the pass materialises the
  offset. Loads recovered went 127 → 210.
- **An `if` arm holding only a label** rendered as `if (c) { loc_97C: }`. The
  label belongs after the `if`, which is where that arm leads, so a jump to it
  still lands correctly and the condition inverts to give a non-empty arm.

The 777 leftover `storeq` are not misses: 704 are genuine full-quadword vector
stores and 29 store a constant quadword. What remains addressable is 19 chained
`shufb` (two fields spliced into one quadword before a single store) and 16
masks computed at runtime.

## Corpus run

23 PS3 SPU binaries across two firmware versions (`isoldr`, `lv1ldr`,
`lv2ldr`, `appldr`, `rvkldr`, `sc_iso`, the verifiers and the iso modules),
run end to end — lift, SSA, optimise, scalarise, structure, render:

```
3889 functions, 28363 blocks, 451760 IR instructions   (~2.5 min total)
0 failures, 0 SSA verifier problems, 0 unmodelled mnemonics
6662 scalar stores, 6307 scalar loads, 654 aligned loads, 104292 scalar ops
3789 loops, 1993 gotos, 257076 lines of pseudocode
```

The interesting number was the unreachable-block count, which pointed straight
at a real limitation and got both of the `spu.py` fixes above written:
**IDA's SPU processor module did not recognise switch dispatch**, so a `bi`
through a jump table produced no edges and every case body was orphaned.
`inflate` in `lv2ldr` was 350 blocks, 339 of which were four-byte jump-table
cases with no predecessors — and the table itself had been swept into bogus
`stop` instructions.

Against the run before those fixes (27306 blocks, 439270 IR instructions,
3202 gotos), switch recognition added **+1057 blocks and +12490 IR
instructions** of previously-dropped code, and tail duplication cut gotos by
**38%** at a cost of about 5% more output.

`_prune_unreachable` also merges adjacent ranges now, so one unresolved table
reads as one range rather than 339 — without that the corpus figure reads
1962 instead of 190.

## Known limitations

- Two `rdch` on the same channel are ordered by the `ch` chain, but the chain
  bump is a separate `sync` instruction rather than a second destination.
- Callee-clobbered registers are not modelled: a value in a volatile register
  appears to survive a call.
- Liveness is per-function. A register handed between functions (common in
  hand-written ROM code, where IDA's function boundaries are approximate) can
  be eliminated as dead.
- Phi-web coalescing assumes the members of a web are never simultaneously
  live with different values. That is the standard decompiler assumption and
  holds for compiler output, but it is an assumption — the SSA view (`i` in
  the viewer) is the ground truth if a rendering ever looks wrong.
- Type recovery stops at scalars, pointers and vectors. No structs, no arrays,
  no typedefs: `p->field` and `a[i]` still read as pointer arithmetic.
- Parameter recovery is per-function, so a parameter that is only forwarded to
  another call is missed (19 of 95 on metldr). Fixing it properly means
  computing arity for the whole program to a fixpoint, so a call site can be
  told how many of its argument registers the callee actually reads.
- Switch dispatch is resolved only for the GCC jump-table idiom the patched
  `spu.py` matches. Other shapes still arrive as unreachable blocks; they are
  reported, not decompiled.
- Code unreachable from a function's entry is reported, not decompiled. Create
  a function at the address and it will be.
- Analysis is per-function, so cross-function idioms are not recovered. The
  reference's `dma_get` is shared code entered both inline and by link
  register, with `binz $71,$71` as its return — nothing here infers that.
