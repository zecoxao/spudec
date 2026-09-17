# spudec — SPU lifter + SSA IR for IDA Pro 9.x

Stage 1 of an SPU decompiler: IDA's SPU processor module is used purely as a
decoder, and everything above it is ours.

```
procs/spu.py -> lifter -> CFG -> SSA -> opt -> scalarise -> opt
             -> structure -> pseudocode
```

Place the plugin and the folder under the ida instalation directory / plugins folder

## C++ names

Mangled symbols are demangled to the qualified name without its parameter list
(IDA's `MNG_NODEFINIT`), since the real arguments are printed instead, so a
call reads as `ss::ss_iso_dprintf::put_s(r3, "In main: ")`. Two functions can
demangle to one name -- a constructor emits both a complete-object and a
base-object body, differing only in the mangling -- so a name shared by more
than one function keeps its address (`ss::crypto::crypto_1510`). Compiler
helpers whose demangled form is not expression-safe (`` `global constructor
keyed to' ``) are sanitised into identifiers.

## What the listing promises

The declaration block and the body are kept in agreement: every name the body
mentions is declared, and every declared local is assigned somewhere (a
register the function only reads is marked `// live in` instead). That sounds
obvious, but three separate things used to break it, and each produced a
listing that referred to a value it never showed being computed:

* A clobber read only by a phi whose own result went nowhere but another
  call's conservative ABI operand list was printed as `rN = <result in rN>`.
  On call-heavy code that was 14% of all output lines.
* The inlining depth cap printed a bare *name* for a definition that had
  already been dropped as a statement, so a deeply nested expression used an
  undeclared variable. The cap now applies when inlinability is decided, so
  anything too deep simply stays a statement.
* A callee-saved register's name is printed by the prologue save (which reads
  the incoming value) while the epilogue restore prints nothing, so the name
  was classed as a local that is never assigned rather than as live-in.

`tests/test_pipeline.py` audits every function it renders for exactly this
agreement, so a regression in either direction fails a test.

## Tests

    python tests/test_pipeline.py

Builds IR by hand, so it needs no IDA: covers the phi-argument regression and
the string rendering.
