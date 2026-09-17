"""
SPU channel map and MFC command decoding.

Channel I/O is how an SPU does everything outside its own local store -- DMA,
mailboxes, signals, decrementer, events -- so a listing full of `wrch(21, ...)`
hides exactly the part a reader cares about.  The numbers are architectural and
documented (Cell Broadband Engine Architecture, SPU channel map), so naming
them costs nothing and is not a guess.

Channels 64-> are not in the CBEA: they are the PS3 isolation-mode channels
used by the SPU boot ROM.  They are labelled as such rather than silently
presented as if they were architectural.

No IDA imports.
"""

# Architectural SPU channels (CBEA).  (name, direction)
CHANNELS = {
    0:  ("SPU_RdEventStat", "r"),
    1:  ("SPU_WrEventMask", "w"),
    2:  ("SPU_WrEventAck", "w"),
    3:  ("SPU_RdSigNotify1", "r"),
    4:  ("SPU_RdSigNotify2", "r"),
    7:  ("SPU_WrDec", "w"),
    8:  ("SPU_RdDec", "r"),
    9:  ("MFC_WrMSSyncReq", "w"),
    11: ("SPU_RdEventMask", "r"),
    12: ("MFC_RdTagMask", "r"),
    13: ("SPU_RdMachStat", "r"),
    14: ("SPU_WrSRR0", "w"),
    15: ("SPU_RdSRR0", "r"),
    16: ("MFC_LSA", "w"),
    17: ("MFC_EAH", "w"),
    18: ("MFC_EAL", "w"),
    19: ("MFC_Size", "w"),
    20: ("MFC_TagID", "w"),
    21: ("MFC_Cmd", "w"),
    22: ("MFC_WrTagMask", "w"),
    23: ("MFC_WrTagUpdate", "w"),
    24: ("MFC_RdTagStat", "r"),
    25: ("MFC_RdListStallStat", "r"),
    26: ("MFC_WrListStallAck", "w"),
    27: ("MFC_RdAtomicStat", "r"),
    28: ("SPU_WrOutMbox", "w"),
    29: ("SPU_RdInMbox", "r"),
    30: ("SPU_WrOutIntrMbox", "w"),
}

# PS3 isolation-mode channels -- not architectural.
ISO_CHANNELS = {
    64: "SPU_IsoCntl",
    65: "SPU_IsoStat",
    66: "SPU_IsoRootKey",       # undocumented; hardware root key, 8 x 32 bits
    67: "SPU_IsoKey",
    68: "SPU_IsoIdent",
}


def name(n):
    """A readable name for channel ``n``, or None if unknown."""
    if n in CHANNELS:
        return CHANNELS[n][0]
    return ISO_CHANNELS.get(n)


def label(n):
    """``name`` if known, otherwise the bare number."""
    return name(n) or str(n)


def is_iso(n):
    return n in ISO_CHANNELS


# ---------------------------------------------------------------------------
# MFC commands (the value written to MFC_Cmd)
# ---------------------------------------------------------------------------

MFC_CMDS = {
    0x20: "PUT", 0x21: "PUTS", 0x22: "PUTR",
    0x24: "PUTB", 0x25: "PUTBS", 0x26: "PUTRB",
    0x28: "PUTF", 0x29: "PUTFS", 0x2A: "PUTRF",
    0x40: "GET", 0x41: "GETS",
    0x44: "GETB", 0x45: "GETBS",
    0x48: "GETF", 0x49: "GETFS",
    0x50: "PUTL", 0x54: "PUTLB", 0x58: "PUTLF",
    0x60: "GETL", 0x64: "GETLB", 0x68: "GETLF",
    0xA0: "SNDSIG", 0xA4: "SNDSIGB", 0xA8: "SNDSIGF",
    0xB0: "GETLLAR", 0xB4: "PUTLLC", 0xB8: "PUTLLUC", 0xBC: "PUTQLLUC",
    0xCC: "BARRIER", 0xC0: "MFCEIEIO", 0xC8: "MFCSYNC",
}


def mfc_cmd(value):
    """Name of an MFC command code, or None."""
    return MFC_CMDS.get(value & 0xFF)
