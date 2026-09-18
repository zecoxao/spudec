"""
Recovering what a constant address points at.

Once `ila`/`ilhu`+`iohl` have been folded, an address is a plain literal in the
preferred slot, and a great many of those point at C strings -- SPU code is as
full of format strings as anything else.  A line reading

    sub_1C5A4(0x2C018, r4, 0x1F4);

says nothing; the same line reading

    sub_1C5A4("INFO: %s(%d) err %d\\n", r4, 0x1F4);

says what the function is and what the other arguments must be.  So a constant
that points at a NUL-terminated char array is rendered as that string.

Two sources of truth, in order:

1. **A string literal IDA already has defined there.**  That is the strongest
   signal available: it carries IDA's own analysis and any markup the user
   added by hand, and honouring it means the decompiler and the disassembly
   never disagree about what a byte range is.
2. **A sniff**, for the very common case of an address pointing into the
   *middle* of a defined string (a shared `"%d\\n"` tail) or at data IDA never
   got round to typing.  Deliberately strict -- printable, terminated inside a
   bounded window, long enough, and containing at least one letter or digit --
   because a false positive here does not merely look untidy, it asserts
   something false about the program.

This module imports IDA lazily, so the rest of the pipeline stays testable
without a database.
"""

MIN_LEN = 4          # shorter runs are far more often coincidence than text
MAX_LEN = 96         # how much of a long string to show before eliding
WINDOW = 4096        # how far to look for the terminator

# Printable ASCII plus the whitespace that really occurs in format strings.
_OK = frozenset(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D}
_ALNUM = frozenset(range(0x30, 0x3A)) | frozenset(range(0x41, 0x5B)) \
    | frozenset(range(0x61, 0x7B))

_ESCAPES = {0x09: "\\t", 0x0A: "\\n", 0x0D: "\\r",
            0x22: '\\"', 0x5C: "\\\\"}


def escape(text):
    """The C spelling of ``text``, quotes included."""
    out = []
    for ch in text:
        e = _ESCAPES.get(ord(ch))
        out.append(e if e is not None else ch)
    return '"%s"' % "".join(out)


def _is_progression(body):
    """
    True when the bytes form a constant-stride run -- a table, not text.

    Byte tables land in printable ASCII surprisingly often: metldr's SPU
    shuffle-control masks are `20 22 24 26 ... 3E` and `40 44 48 4C ... 7C`,
    both of which are printable, NUL-terminated and full of digits, so every
    other rule here waves them through.  Real text is not an arithmetic
    progression.

    A stride of 1 is deliberately allowed: `"0123456789"` is a progression but
    it is also a genuine string, used by every hand-written itoa.  A stride of
    0 is a run of one repeated byte, which is padding once it is long.
    """
    if len(body) < 4:
        return False
    stride = body[1] - body[0]
    if stride == 1:
        return False
    if not all(body[i + 1] - body[i] == stride for i in range(len(body) - 1)):
        return False
    return stride != 0 or len(body) >= 8


def _looks_like_text(body):
    if len(body) < MIN_LEN:
        return False
    if not all(b in _OK for b in body):
        return False
    if _is_progression(body):
        return False
    # A run of spaces, dashes or digits-as-padding is not a string.  Requiring
    # a letter or digit is enough to reject the padding patterns that actually
    # occur without rejecting anything real.
    return any(b in _ALNUM for b in body)


class Strings(object):
    """
    Resolves constant addresses to string literals, with a cache.

    One instance per decompilation run; the cache matters because the same
    format string is referenced from many call sites and the whole-database
    pass asks about every constant in every function.
    """

    def __init__(self, min_len=MIN_LEN, max_len=MAX_LEN):
        self.min_len = min_len
        self.max_len = max_len
        self._cache = {}
        self.hits = 0

    def __call__(self, ea):
        """The C literal for the string at ``ea``, or None."""
        if ea in self._cache:
            return self._cache[ea]
        text = self._lookup(ea)
        lit = None
        if text is not None:
            self.hits += 1
            if len(text) > self.max_len:
                # Elide rather than print a screenful.  The ellipsis is inside
                # the quotes so it reads as "and it goes on", not as content.
                lit = escape(text[:self.max_len])[:-1] + '..."'
            else:
                lit = escape(text)
        self._cache[ea] = lit
        return lit

    def text(self, ea):
        """The decoded string at ``ea`` without C quoting, or None."""
        return self._lookup(ea)

    # -- the two sources ---------------------------------------------------

    def _lookup(self, ea):
        try:
            import ida_bytes
        except ImportError:
            return None
        if ea <= 0 or not ida_bytes.is_loaded(ea):
            return None

        defined = self._defined(ea)
        if defined is not None:
            return defined
        return self._sniff(ea)

    @staticmethod
    def _defined(ea):
        """
        The string literal IDA has at -- or *covering* -- this address.

        The address need not be the literal's head.  A compiler pools the
        shared tail of one string as another, so `"buf_addr is NULL\n"`
        defined at 0x19780 is also where the bare `"\n"` at 0x19790 lives,
        and a `put_s(0x19790)` really does print just the newline.  Resolving
        through the item head is what makes that read as a string, and it
        keeps the decompiler agreeing with the disassembly about what the byte
        range is.

        Length and content are not questioned here: this tier is IDA's own
        analysis plus whatever the user marked up by hand, which outranks any
        guess.  The sniffer below is the tier that has to be strict, because
        it is the one guessing.
        """
        import ida_bytes
        import ida_nalt
        head = ida_bytes.get_item_head(ea)
        if not ida_bytes.is_strlit(ida_bytes.get_flags(head)):
            return None
        # Read from `ea`, not from the head: the tail is what the code points
        # at, and reporting the whole string would be wrong.
        raw = ida_bytes.get_strlit_contents(ea, -1, ida_nalt.STRTYPE_C)
        if not raw:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    def _sniff(self, ea):
        import ida_bytes
        raw = ida_bytes.get_bytes(ea, WINDOW)
        if not raw:
            # Near the end of local store a full window is not readable; fall
            # back to something that certainly is.
            raw = ida_bytes.get_bytes(ea, self.max_len + 1)
        if not raw:
            return None
        n = raw.find(b"\0")
        if n < 0:
            return None                 # not terminated -- not a C string
        body = raw[:n]
        if not _looks_like_text(body):
            return None
        return body.decode("latin-1")


# ---------------------------------------------------------------------------
# function names
# ---------------------------------------------------------------------------
#
# C++ code compiled for these cores keeps its mangled symbols, and a listing
# full of `_ZN2ss6cryptoC1ENS_16crypto_algorithmE(...)` is barely readable.
# IDA can demangle, and `MNG_NODEFINIT` gives exactly the form a call site
# wants: the qualified name with no parameter list, since the real arguments
# are printed instead.

# Characters that may appear in a name without making the expression it sits
# in unreadable.  `~` is a destructor, `::` a scope, `<>,` a template.
_NAME_OK = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:~<>,")


def _sanitise(name):
    """
    Make a demangled name safe to drop into an expression.

    Most demangled names already are.  The exceptions are the compiler's own
    helpers -- "`global constructor keyed to'ss::foo" -- whose backticks,
    apostrophes and spaces would read as broken syntax in a call.
    """
    if all(ch in _NAME_OK for ch in name):
        return name
    out = "".join(ch if ch in _NAME_OK else "_" for ch in name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or name


def demangle(name):
    """The qualified C++ name without its parameter list, or None."""
    if not name:
        return None
    try:
        import ida_name
    except ImportError:
        return None
    d = ida_name.demangle_name(name, ida_name.MNG_NODEFINIT)
    if not d or d == name:
        return None
    # Defensive: if a parameter list survived, cut it.  `operator()` is the
    # one name where the parentheses are part of the name itself.
    if "(" in d and "operator" not in d:
        d = d.split("(", 1)[0].rstrip()
    return _sanitise(d)


class Names(object):
    """
    Function names for the listing: demangled, and never ambiguous.

    Two functions can demangle to one name -- a C++ constructor emits both a
    complete-object and a base-object body, and they differ only in the
    mangling -- so a name shared by more than one function keeps its address.
    Printing one name for two functions would quietly merge them, which is
    precisely the kind of confident wrongness this decompiler tries not to
    produce.
    """

    def __init__(self):
        self._cache = {}
        self._shared = None
        self.demangled = 0

    def _shared_names(self):
        if self._shared is not None:
            return self._shared
        self._shared = set()
        try:
            import ida_funcs
            import ida_name
        except ImportError:
            return self._shared
        seen = {}
        for i in range(ida_funcs.get_func_qty()):
            f = ida_funcs.getn_func(i)
            if f is None:
                continue
            d = demangle(ida_name.get_name(f.start_ea))
            if d:
                seen[d] = seen.get(d, 0) + 1
        self._shared = {n for n, c in seen.items() if c > 1}
        return self._shared

    def __call__(self, ea):
        if ea in self._cache:
            return self._cache[ea]
        raw = None
        try:
            import ida_name
            raw = ida_name.get_name(ea)
        except ImportError:
            pass
        d = demangle(raw)
        if d:
            self.demangled += 1
            out = "%s_%X" % (d, ea) if d in self._shared_names() else d
        else:
            out = raw or ("sub_%X" % ea)
        self._cache[ea] = out
        return out


# ---------------------------------------------------------------------------
# C++ member functions and the implicit `this`
# ---------------------------------------------------------------------------
#
# A non-static member function receives the object as a hidden first argument in
# r3, but the mangled name does not say so: `lv0::appli_secure_loader::start()`
# mangles its parameter list as `v` (void) exactly like the free function
# `lv0::some_ns::start()` does, so the name alone cannot tell a member from a
# namespaced free/static function.  What settles it is auxiliary evidence that
# the enclosing scope is a *type*: a vtable, a constructor or a destructor for
# it.  A scope with any of those is a class, and a function directly inside a
# class scope is one of its members -- whose r3 is `this`.


class Members(object):
    """
    Resolves a function to the class it is a member of, or None.

    Builds the set of class scopes once from the whole symbol table (vtables,
    constructors, destructors), then answers per function by matching the
    enclosing scope of its demangled name against that set.  Caller confirms
    the r3 is actually used as a pointer before trusting it as `this`, so a
    static member -- a class scope with no object -- is not mislabelled.
    """

    def __init__(self):
        self._classes = None
        self._cache = {}

    def _class_set(self):
        if self._classes is not None:
            return self._classes
        classes = set()
        try:
            import ida_name
            import idautils
        except ImportError:
            self._classes = classes
            return classes
        for ea, raw in idautils.Names():
            if raw.startswith("_ZTV"):
                d = ida_name.demangle_name(raw, ida_name.MNG_NODEFINIT)
                if d:
                    m = d.split("for'", 1)
                    if len(m) == 2:
                        classes.add(_sanitise(m[1].strip().rstrip(">").strip()))
                continue
            d = demangle(raw)
            if not d or "::" not in d:
                continue
            scope, last = d.rsplit("::", 1)
            cls = scope.rsplit("::", 1)[-1]
            # A constructor is `A::B::B`; a destructor `A::B::~B`.  Either way
            # the scope `A::B` names a class.
            if last == cls or last == "~" + cls:
                classes.add(scope)
        self._classes = classes
        return classes

    def __call__(self, ea):
        """The class `ea` is a member of, or None -- keyed by address."""
        if ea in self._cache:
            return self._cache[ea]
        cls = None
        try:
            import ida_name
            d = demangle(ida_name.get_name(ea))
        except ImportError:
            d = None
        if d and "::" in d:
            scope = d.rsplit("::", 1)[0]
            if scope in self._class_set():
                cls = scope
        self._cache[ea] = cls
        return cls


# ---------------------------------------------------------------------------
# shared, per-session resolvers
# ---------------------------------------------------------------------------
#
# One of each per session rather than one per function: the same format string
# and the same callee are referenced from many places, and the whole-database
# pass asks about every constant and every call in every function, so the
# caches are what keep the lookups from dominating the run.

_STRINGS = None
_NAMES = None
_MEMBERS = None


def strings():
    global _STRINGS
    if _STRINGS is None:
        _STRINGS = Strings()
    return _STRINGS


def names():
    global _NAMES
    if _NAMES is None:
        _NAMES = Names()
    return _NAMES


def members():
    global _MEMBERS
    if _MEMBERS is None:
        _MEMBERS = Members()
    return _MEMBERS


def clear():
    """Forget everything -- call after retyping data or renaming functions."""
    global _STRINGS, _NAMES, _MEMBERS
    _STRINGS = None
    _NAMES = None
    _MEMBERS = None
