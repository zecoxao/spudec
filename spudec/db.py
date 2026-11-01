"""
Database-backed persistence layer for spudec using IDA netnodes.

Stores and retrieves per-function:
- Resilient local variable locators (vdloc + defea -> name, type, flags)
- Variable renames (old_name -> new_name, backward-compatible)
- Variable type overrides (var_name -> type_str)
- Function prototype overrides (prototype string)
- Statement comments (stmt_ea -> comment_str)
- Number format overrides (stmt_ea:opnum -> 'hex' | 'dec' | 'char')
- Inverted if-conditions (set of branch_ea)
- Global UI preferences (sync with disassembly enabled)

When run outside IDA (headless tests), falls back to an in-memory dictionary.
"""

import json

try:
    import ida_netnode
except ImportError:
    ida_netnode = None

NODE_NAME = "$ spudec"
_MEM_STORAGE = {}
_MEM_SYNC = True


def _get_node():
    if ida_netnode is None:
        return None
    try:
        node = ida_netnode.netnode()
        node.create(NODE_NAME)
        return node
    except Exception:
        return None


def get_func_data(func_ea):
    """Retrieve persistent renames, types, locators, comments, and prototype for a function."""
    node = _get_node()
    default_data = {
        "renames": {},
        "types": {},
        "proto": None,
        "lvars": {},
        "cmts": {},
        "numforms": {},
        "iflags": []
    }
    if node is None:
        data = _MEM_STORAGE.get(func_ea)
        if not data:
            return default_data
        for k, v in default_data.items():
            data.setdefault(k, v)
        return data

    try:
        val = node.supstr(func_ea)
        if val:
            data = json.loads(val)
            for k, v in default_data.items():
                data.setdefault(k, v)
            return data
    except Exception:
        pass
    return default_data


def set_func_data(func_ea, data):
    """Save persistent data for a function."""
    node = _get_node()
    if node is None:
        _MEM_STORAGE[func_ea] = data
        return
    try:
        node.supset(func_ea, json.dumps(data))
    except Exception:
        _MEM_STORAGE[func_ea] = data


def set_var_rename(func_ea, old_name, new_name):
    """Save a variable rename (e.g. r3_3 -> counter, or counter -> new_counter).
    
    If new_name is empty or None, the rename is cleared, reverting to default.
    """
    data = get_func_data(func_ea)
    renames = data.setdefault("renames", {})

    canonical = None
    for k, v in renames.items():
        if v == old_name:
            canonical = k
            break
    key = canonical if canonical is not None else old_name

    if not new_name or new_name == key:
        renames.pop(key, None)
    else:
        renames[key] = new_name
    set_func_data(func_ea, data)


def set_var_type(func_ea, var_name, type_str):
    """Save a variable type override (e.g. r3 -> uint32_t *).
    
    If type_str is empty or None, the override is cleared.
    """
    data = get_func_data(func_ea)
    types = data.setdefault("types", {})
    if not type_str:
        types.pop(var_name, None)
    else:
        types[var_name] = type_str.strip()
    set_func_data(func_ea, data)


def set_var_locator(func_ea, locator, name=None, type_str=None, flags=0):
    """Save user modifications keyed by robust definition-site locator.
    
    Locator example: 'reg:3@0x2548' or 'stk:-0x20@0x2550'.
    """
    data = get_func_data(func_ea)
    lvars = data.setdefault("lvars", {})
    if not name and not type_str and not flags:
        lvars.pop(locator, None)
    else:
        entry = lvars.setdefault(locator, {})
        if name is not None:
            entry["name"] = name
        if type_str is not None:
            entry["type"] = type_str.strip()
        if flags:
            entry["flags"] = flags
    set_func_data(func_ea, data)


def get_var_locator(func_ea, locator):
    """Retrieve saved info for a locator."""
    data = get_func_data(func_ea)
    return data.get("lvars", {}).get(locator)


def set_statement_cmt(func_ea, stmt_ea, cmt_str):
    """Save a statement or line comment attached to instruction linear address."""
    data = get_func_data(func_ea)
    cmts = data.setdefault("cmts", {})
    ea_key = hex(stmt_ea) if isinstance(stmt_ea, int) else str(stmt_ea)
    if not cmt_str:
        cmts.pop(ea_key, None)
    else:
        cmts[ea_key] = cmt_str.strip()
    set_func_data(func_ea, data)


def get_statement_cmt(func_ea, stmt_ea):
    """Retrieve saved comment for a statement."""
    data = get_func_data(func_ea)
    ea_key = hex(stmt_ea) if isinstance(stmt_ea, int) else str(stmt_ea)
    return data.get("cmts", {}).get(ea_key)


def set_numform(func_ea, stmt_ea, op_idx, form_str):
    """Save immediate constant display format ('hex', 'dec', 'char')."""
    data = get_func_data(func_ea)
    numforms = data.setdefault("numforms", {})
    key = f"{hex(stmt_ea) if isinstance(stmt_ea, int) else str(stmt_ea)}:{op_idx}"
    if not form_str or form_str == "default":
        numforms.pop(key, None)
    else:
        numforms[key] = form_str.lower()
    set_func_data(func_ea, data)


def get_numform(func_ea, stmt_ea, op_idx=0):
    """Retrieve saved format for an immediate constant."""
    data = get_func_data(func_ea)
    key = f"{hex(stmt_ea) if isinstance(stmt_ea, int) else str(stmt_ea)}:{op_idx}"
    return data.get("numforms", {}).get(key)


def toggle_iflag(func_ea, branch_ea):
    """Toggle inverted condition flag for an if statement."""
    data = get_func_data(func_ea)
    iflags = data.setdefault("iflags", [])
    ea_val = branch_ea if isinstance(branch_ea, int) else int(branch_ea, 16)
    if ea_val in iflags:
        iflags.remove(ea_val)
        inverted = False
    else:
        iflags.append(ea_val)
        inverted = True
    set_func_data(func_ea, data)
    return inverted


def is_iflag_inverted(func_ea, branch_ea):
    """Check if if-condition at branch_ea is marked inverted."""
    data = get_func_data(func_ea)
    ea_val = branch_ea if isinstance(branch_ea, int) else int(branch_ea, 16)
    return ea_val in data.get("iflags", [])


def set_func_proto(func_ea, proto_str):
    """Save a function prototype override (e.g. int sub_488(qword a1, int a2)).
    
    If proto_str is empty or None, the override is cleared.
    """
    data = get_func_data(func_ea)
    if not proto_str:
        data["proto"] = None
    else:
        data["proto"] = proto_str.strip()
    set_func_data(func_ea, data)


def get_sync_enabled():
    """Retrieve global sync-with-disassembly preference."""
    node = _get_node()
    if node is None:
        return _MEM_SYNC
    try:
        val = node.altval(0)
        return val != 2
    except Exception:
        return True


def set_sync_enabled(enabled):
    """Save global sync-with-disassembly preference."""
    global _MEM_SYNC
    _MEM_SYNC = bool(enabled)
    node = _get_node()
    if node is None:
        return
    try:
        node.altset(0, 1 if enabled else 2)
    except Exception:
        pass


WIN_STATE_KEY = 0x7FFFFFFF
_MEM_WIN_STATE = {"is_open": False, "func_ea": None, "mode": "c", "lineno": 0, "col": 0}


def save_window_state(is_open, func_ea=None, lineno=0, col=0, mode="c"):
    """Persist decompiler window state (open status, active function, cursor position)."""
    global _MEM_WIN_STATE
    state = {
        "is_open": bool(is_open),
        "func_ea": func_ea,
        "lineno": lineno,
        "col": col,
        "mode": mode
    }
    _MEM_WIN_STATE = state
    node = _get_node()
    if node is not None:
        try:
            node.supset(WIN_STATE_KEY, json.dumps(state))
        except Exception:
            pass


def load_window_state():
    """Retrieve persisted decompiler window state."""
    node = _get_node()
    if node is not None:
        try:
            val = node.supstr(WIN_STATE_KEY)
            if val:
                return json.loads(val)
        except Exception:
            pass
    return dict(_MEM_WIN_STATE)


def clear_all():
    """Clear in-memory and database storage (useful in tests)."""
    global _MEM_STORAGE, _MEM_SYNC, _MEM_WIN_STATE
    _MEM_STORAGE.clear()
    _MEM_SYNC = True
    _MEM_WIN_STATE = {"is_open": False, "func_ea": None, "mode": "c", "lineno": 0, "col": 0}
    node = _get_node()
    if node is not None:
        try:
            node.kill()
        except Exception:
            pass
