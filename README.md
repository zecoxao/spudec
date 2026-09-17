# spudec — SPU lifter + SSA IR for IDA Pro 9.x

Stage 1 of an SPU decompiler: IDA's SPU processor module is used purely as a
decoder, and everything above it is ours.

```
procs/spu.py -> lifter -> CFG -> SSA -> opt -> scalarise -> opt
             -> structure -> pseudocode
```

Place the plugin and the folder under the ida instalation directory / plugins folder