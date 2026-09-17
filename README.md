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

## Tests

    python tests/test_pipeline.py

Builds IR by hand, so it needs no IDA: covers the phi-argument regression and
the string rendering.
