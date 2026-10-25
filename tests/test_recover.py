"""
Headless tests for the spudec persistence, prototype parsing, and user override layer.

    python tests/test_recover.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spudec.ir import Op, EW, Const, Var, Insn, Function
from spudec import regs, ssa, opt, structure, cgen, abi, db, outssa, scalarize


def _build_simple_func(name="my_func", ea=0x100):
    f = Function(ea, name)
    b = f.new_block(ea, ea + 0x10)
    # r3 = r3 + r4; ret
    b.add(Insn(Op.ADD, Var(regs.ARG_FIRST), [Var(regs.ARG_FIRST), Var(regs.ARG_FIRST + 1)], EW.W, ea=ea))
    b.add(Insn(Op.RET, None, [Var(regs.ARG_FIRST)], ea=ea + 4))
    return f


def test_db_persistence():
    db.clear_all()
    ea = 0x1000

    assert db.get_sync_enabled() is True
    db.set_sync_enabled(False)
    assert db.get_sync_enabled() is False
    db.set_sync_enabled(True)
    assert db.get_sync_enabled() is True

    data = db.get_func_data(ea)
    assert data["renames"] == {}
    assert data["types"] == {}
    assert data["proto"] is None

    # Variable rename
    db.set_var_rename(ea, "r3", "result")
    assert db.get_func_data(ea)["renames"]["r3"] == "result"

    # Canonical renaming chain
    db.set_var_rename(ea, "result", "final_val")
    assert db.get_func_data(ea)["renames"]["r3"] == "final_val"

    # Revert rename
    db.set_var_rename(ea, "final_val", None)
    assert "r3" not in db.get_func_data(ea)["renames"]

    # Variable type
    db.set_var_type(ea, "r3", "int *")
    assert db.get_func_data(ea)["types"]["r3"] == "int *"
    db.set_var_type(ea, "r3", "")
    assert "r3" not in db.get_func_data(ea)["types"]

    # Prototype
    db.set_func_proto(ea, "int sub_1000(int a1, void *a2)")
    assert db.get_func_data(ea)["proto"] == "int sub_1000(int a1, void *a2)"
    db.set_func_proto(ea, None)
    assert db.get_func_data(ea)["proto"] is None

    db.clear_all()
    print("ok  test_db_persistence")


def test_proto_parser():
    sig = abi._parse_proto("int *calculate(int a, void *buf, unsigned int size)")
    assert sig is not None
    assert sig.name == "calculate"
    assert sig.ret_type == "int *"
    assert len(sig.params) == 3
    assert sig.params[0].name == "a" and sig.params[0].type_str == "int" and sig.params[0].location == "r3"
    assert sig.params[1].name == "buf" and sig.params[1].type_str == "void *" and sig.params[1].location == "r4"
    assert sig.params[2].name == "size" and sig.params[2].type_str == "unsigned int" and sig.params[2].location == "r5"
    assert sig.format_signature() == "int *calculate(int a, void *buf, unsigned int size)"

    # Void return & void params
    sig_void = abi._parse_proto("void reset(void)")
    assert sig_void is not None
    assert sig_void.ret_type == "void"
    assert len(sig_void.params) == 0
    assert sig_void.format_signature() == "void reset(void)"

    print("ok  test_proto_parser")


def test_cgen_user_overrides():
    f = _build_simple_func()
    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)

    # 1. Default render
    lines = cgen.generate(f, stmts, info)
    assert isinstance(lines, cgen.CodeLines)
    assert lines.header_count > 0
    assert f.start_ea in lines.ea_to_line
    default_text = "\n".join(lines)
    assert "vec_uint4 my_func(vec_uint4 a1, vec_uint4 a2)" in default_text

    # 2. Render with user renames and types
    user_renames = {"my_func": "add_numbers", "a1": "left", "a2": "right"}
    user_types = {"__return__": "int", "left": "int", "right": "int"}
    lines2 = cgen.generate(f, stmts, info, user_renames=user_renames, user_types=user_types)
    text2 = "\n".join(lines2)
    assert "int add_numbers(int left, int right)" in text2
    assert lines2.header_count > 0
    assert lines2.line_to_eas
    assert lines2.ea_to_line

    # 3. Render with user prototype override
    user_proto = "int add_numbers(int x, int y)"
    lines3 = cgen.generate(f, stmts, info, user_proto=user_proto)
    text3 = "\n".join(lines3)
    assert "int add_numbers(int x, int y)" in text3

    print("ok  test_cgen_user_overrides")


def test_codelines_mapping():
    f = _build_simple_func(ea=0x200)
    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info)

    assert hasattr(lines, "header_count")
    assert hasattr(lines, "line_eas")
    assert hasattr(lines, "line_to_eas")
    assert hasattr(lines, "ea_to_line")

    assert lines.header_count > 0
    assert len(lines.line_eas) == len(lines)
    # 0x200 should map to the function header or instruction
    assert 0x200 in lines.ea_to_line
    line_for_200 = lines.ea_to_line[0x200]
    assert 0x200 in lines.line_to_eas[line_for_200]

    print("ok  test_codelines_mapping")


def test_stack_local_variables():
    # Build a function with a stack frame and address-of slot calculation
    ea = 0x500
    R3 = regs.ARG_FIRST
    SP, M = regs.SP, regs.R_MEM
    t = regs.NREG
    def word0(v):
        return (v & 0xFFFFFFFF) << 96
    MASK = Const(word0(0x3FFF0))

    f = Function(ea, "func_with_frame")
    b = f.new_block(ea, ea + 0x40)

    # saves lr at sp+0x10
    b.add(Insn(Op.ADD, Var(t), [Var(SP), Const(word0(0x10))], EW.W, ea=ea))
    b.add(Insn(Op.AND, Var(t + 1), [Var(t), MASK], EW.W, ea=ea + 2))
    b.add(Insn(Op.STOREQ, Var(M), [Var(M), Var(t + 1), Var(regs.LR)], EW.Q, ea=ea + 4))

    # back chain at sp-0x30
    b.add(Insn(Op.ADD, Var(t + 2), [Var(SP), Const(word0(-0x30))], EW.W, ea=ea + 6))
    b.add(Insn(Op.AND, Var(t + 3), [Var(t + 2), MASK], EW.W, ea=ea + 8))
    b.add(Insn(Op.STOREQ, Var(M), [Var(M), Var(t + 3), Var(SP)], EW.Q, ea=ea + 10))

    # sp = sp - 0x30
    b.add(Insn(Op.ADD, Var(SP), [Var(SP), Const(word0(-0x30))], EW.W, ea=ea + 12))

    # r3 = sp + 0x20  (which is sp_in - 0x10 -> var_10)
    b.add(Insn(Op.ADD, Var(R3), [Var(SP), Const(word0(0x20))], EW.W, ea=ea + 16))

    # call callee(r3)
    b.add(Insn(Op.CALL, Var(M), [Var(M), Var(regs.R_CH), Var(regs.LR), Var(R3)], aux=0x600, ea=ea + 20))

    # ret
    b.add(Insn(Op.RET, None, [Var(M), Var(regs.R_CH), Var(regs.LR), Var(R3)], ea=ea + 24))

    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)

    # callee taking 1 arg
    def mock_arity(aux):
        return 1 if aux == 0x600 else 0

    lines = cgen.generate(f, stmts, info, arity_of=mock_arity)
    text = "\n".join(lines)

    # Verify:
    # 1. No sp live-in
    assert "vec_uint4 sp;   // live in" not in text
    # 2. No sp adjust statement
    assert "sp - 0x30" not in text
    # 3. Local variable var_10 declared
    assert "var_10;   // stack" in text
    # 4. Slot address taken as &var_10
    assert "&var_10" in text

    print("ok  test_stack_local_variables")




def test_condition_idioms():
    from spudec.structure import Cond
    # Test function with ceqbi + xsbh + brhnz
    f = Function(0x1000, "test_cond")
    b0 = f.new_block(0x1000, 0x1020)
    b_then = f.new_block(0x1020, 0x1030)
    b_else = f.new_block(0x1030, 0x1040)
    b_exit = f.new_block(0x1040, 0x1050)

    R3, R5, R2 = regs.ARG_FIRST, regs.ARG_FIRST + 2, regs.ENV
    word0 = lambda w: (w & 0xFFFFFFFF) << 96

    # ceqbi r5, r3, 0
    b0.add(Insn(Op.CMPEQ, Var(R5), [Var(R3), Const(word0(0))], EW.B, ea=0x1000))
    # xsbh r2, r5
    b0.add(Insn(Op.EXTS, Var(R2), [Var(R5)], EW.H, ea=0x1004))
    # cjmp r2 hnz -> b_else
    b0.add(Insn(Op.CJMP, None, [Var(R2)], aux=(0x1030, "hnz"), ea=0x1008))

    b0.succs = [b_then, b_else]
    b_then.preds = [b0]; b_else.preds = [b0]

    b_then.add(Insn(Op.JMP, None, [], aux=0x1040, ea=0x1020))
    b_then.succs = [b_exit]; b_exit.preds.append(b_then)

    b_else.add(Insn(Op.JMP, None, [], aux=0x1040, ea=0x1030))
    b_else.succs = [b_exit]; b_exit.preds.append(b_else)

    b_exit.add(Insn(Op.RET, None, [Var(R3)], ea=0x1040))

    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info)
    code = "\n".join(lines)

    # Verify that the condition is simplified to "a1 == 0" (cmpeq with 0)
    assert "a1 == 0" in code, f"Expected 'a1 == 0' in:\n{code}"
    assert "cmpeq.b" not in code, f"Found unsimplified cmpeq.b in:\n{code}"
    assert "exts.h" not in code, f"Found unsimplified exts.h in:\n{code}"

    print("ok  test_condition_idioms")


def test_symbol_resolution():
    # Test global data symbol resolution
    def mock_sym(addr):
        if addr == 0x232A0:
            return "xmmword_232A0"
        if addr == 0x303B0:
            return "g_auth"
        return None

    f = Function(0x2000, "test_sym")
    b = f.new_block(0x2000, 0x2020)
    R3, R4 = regs.ARG_FIRST, regs.ARG_FIRST + 1
    word0 = lambda w: (w & 0xFFFFFFFF) << 96

    # r3 = *(qword *)(0x232A0)
    b.add(Insn(Op.LOADQ, Var(R3), [Var(regs.R_MEM), Const(word0(0x232A0))], EW.Q, ea=0x2000))
    # r4 = 0x303B0
    b.add(Insn(Op.CONST, Var(R4), [Const(word0(0x303B0))], EW.W, ea=0x2004, scalar=True))
    # store r4 into r3 address
    b.add(Insn(Op.STORE, Var(regs.R_MEM), [Var(regs.R_MEM), Var(R3), Var(R4)], EW.W, ea=0x2008))
    b.add(Insn(Op.RET, None, [Var(regs.R_MEM), Var(R3)], ea=0x200C))

    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info, sym_of=mock_sym)
    code = "\n".join(lines)

    assert "&xmmword_232A0" in code, f"Expected '&xmmword_232A0' in:\n{code}"
    assert "&g_auth" in code, f"Expected '&g_auth' in:\n{code}"

    print("ok  test_symbol_resolution")


def test_active_call_args():
    f = Function(0x1000, "test_active")
    b = f.new_block(0x1000, 0x1020)
    R3, R4, R5, R10, R80 = 3, 4, 5, 10, 80
    word0 = lambda w: (w & 0xFFFFFFFF) << 96

    # mfc_put call at 0x1000 with clobbers
    abi_ops = [Var(regs.R_MEM), Var(R3), Var(R4)] + [Var(r) for r in range(5, 75)]
    b.add(Insn(Op.CALL, None, abi_ops, ea=0x1000, aux=0x2000))
    b.add(Insn(Op.UNDEF, Var(R3), [], ea=0x1000))
    b.add(Insn(Op.UNDEF, Var(R4), [], ea=0x1000))
    b.add(Insn(Op.UNDEF, Var(R5), [], ea=0x1000))
    b.add(Insn(Op.UNDEF, Var(R10), [], ea=0x1000))

    # r3 = r80
    b.add(Insn(Op.MOV, Var(R3), [Var(R80)], EW.W, ea=0x1004, scalar=True))

    # call sub_FD80 with unknown arity
    abi_ops2 = [Var(regs.R_MEM), Var(R3), Var(R4), Var(R5), Var(R10)] + [Var(r) for r in range(11, 75)]
    b.add(Insn(Op.CALL, None, abi_ops2, ea=0x1008, aux=0xFD80))
    b.add(Insn(Op.UNDEF, Var(R3), [], ea=0x1008))

    # r10 = 0
    b.add(Insn(Op.CONST, Var(R10), [Const(word0(0))], EW.W, ea=0x100C, scalar=True))
    b.add(Insn(Op.RET, None, [Var(regs.R_MEM), Var(R3)], ea=0x1010))

    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info)
    code = "\n".join(lines)

    assert "<result in r4>" not in code, f"Unexpected '<result in r4>' in:\n{code}"
    assert "<result in r10>" not in code, f"Unexpected '<result in r10>' in:\n{code}"
    assert "sub_FD80(r80)" in code or "sub_FD80(a1)" in code or "sub_FD80(" in code, f"Call not found in:\n{code}"
    assert "sub_FD80(r80, " not in code and "sub_FD80(a1, " not in code, f"Spurious args in call:\n{code}"

    print("ok  test_active_call_args")


def test_extended_arithmetic_simplification():
    f = Function(0x3000, "test_arith")
    b = f.new_block(0x3000, 0x3030)
    R3, R4, R13, R14, R16, R17, R19 = 3, 4, 13, 14, 16, 17, 19
    R80 = 80
    word0 = lambda w: (w & 0xFFFFFFFF) << 96

    # r14 = 0x40000
    b.add(Insn(Op.CONST, Var(R14), [Const(word0(0x40000))], EW.W, ea=0x3000, scalar=True))
    # r19 = bg(r3, r14)
    b.add(Insn(Op.BG, Var(R19), [Var(R3), Var(R14)], EW.W, ea=0x3004))
    # r16_mask = const
    mask_val = 0x04050607C0C0C0C00C0D0E0FC0C0C0C0
    b.add(Insn(Op.CONST, Var(R80), [Const(mask_val)], EW.Q, ea=0x3008))
    # r16_rot = shufb(r19, r19, mask)
    b.add(Insn(Op.SHUFB, Var(R16), [Var(R19), Var(R19), Var(R80)], EW.B, ea=0x300C))
    # r16 = subx(r14, r3, r16_rot)
    b.add(Insn(Op.SUBX, Var(R16), [Var(R14), Var(R3), Var(R16)], EW.W, ea=0x3010))

    # r17 = cg(r16, r4)
    b.add(Insn(Op.CG, Var(R17), [Var(R16), Var(R4)], EW.W, ea=0x3014))
    # r13_rot = shufb(r17, r17, mask)
    b.add(Insn(Op.SHUFB, Var(R13), [Var(R17), Var(R17), Var(R80)], EW.B, ea=0x3018))
    # r13 = addx(r16, r4, r13_rot)
    b.add(Insn(Op.ADDX, Var(R13), [Var(R16), Var(R4), Var(R13)], EW.W, ea=0x301C))

    b.add(Insn(Op.RET, None, [Var(regs.R_MEM), Var(R13)], ea=0x3020))

    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info)
    code = "\n".join(lines)

    assert "bg.w" not in code, f"Unsimplified bg.w in:\n{code}"
    assert "cg.w" not in code, f"Unsimplified cg.w in:\n{code}"
    assert "subx.w" not in code, f"Unsimplified subx.w in:\n{code}"
    assert "addx.w" not in code, f"Unsimplified addx.w in:\n{code}"
    assert "-" in code and "+" in code, f"Expected '-' and '+' in:\n{code}"

    print("ok  test_extended_arithmetic_simplification")


def test_instruction_translation_to_c():
    f = Function(0x4000, "test_insn_trans")
    b = f.new_block(0x4000, 0x4050)
    R3, R4, R5, R6, R7, R8 = 3, 4, 5, 6, 7, 8
    word0 = lambda w: (w & 0xFFFFFFFF) << 96

    # 1. Test selb with constants (-5, 0) and compare condition
    # r4 = (r3 == 0)
    b.add(Insn(Op.CMPEQ, Var(R4), [Var(R3), Const(0)], EW.W, ea=0x4000))
    # r5 = selb(-5, 0, r4)
    # -5 replicated in words
    neg5_w4 = 0xFFFFFFFBFFFFFFFBFFFFFFFBFFFFFFFB
    b.add(Insn(Op.SELB, Var(R5), [Const(neg5_w4), Const(0), Var(R4)], EW.W, ea=0x4004))

    # 2. Test qshlby(r3, 4) -> (u32)r3
    b.add(Insn(Op.QSHLBY, Var(R6), [Var(R3), Const(word0(4))], EW.Q, ea=0x4008))

    # 3. Test replicated 64-bit mask 0xF (2 doublewords of 64 bits each)
    mask_0xf_d2 = (0xF << 64) | 0xF
    b.add(Insn(Op.AND, Var(R7), [Var(R3), Const(mask_0xf_d2)], EW.Q, ea=0x400C))

    # 4. Test cmpgt
    b.add(Insn(Op.CMPGT, Var(R8), [Var(R7), Const(word0(11))], EW.W, ea=0x4010))

    b.add(Insn(Op.RET, None, [Var(regs.R_MEM), Var(R5), Var(R6), Var(R7), Var(R8)], ea=0x4014))

    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info)
    code = "\n".join(lines)

    # Verify no assembly-style literals (#0, :w4, :d2, etc.)
    assert "#" not in code, f"Unexpected '#' assembly constant in:\n{code}"
    assert ":w4" not in code and ":d2" not in code, f"Unexpected lane suffix in:\n{code}"
    # Verify selb translated to C ternary (? :)
    assert "selb(" not in code, f"Untranslated selb in:\n{code}"
    assert "?" in code and ":" in code, f"Expected ternary expression in:\n{code}"
    assert "-0x5" in code, f"Expected -0x5 in:\n{code}"
    # Verify qshlby translated to (u32)
    assert "qshlby(" not in code, f"Untranslated qshlby in:\n{code}"
    assert "(u32)" in code, f"Expected (u32) cast in:\n{code}"
    # Verify 0xF mask
    assert "0xF" in code, f"Expected 0xF in:\n{code}"
    # Verify cmpgt translated to >
    assert "cmpgt" not in code, f"Untranslated cmpgt in:\n{code}"
    assert ">" in code, f"Expected '>' operator in:\n{code}"

    print("ok  test_instruction_translation_to_c")


def test_chained_store_and_fsm_recovery():
    # 1. Test chained store recovery
    f = Function(0x5000, "test_chained_store")
    b = f.new_block(0x5000, 0x5050)
    word0 = lambda w: (w & 0xFFFFFFFF) << 96
    R_MEM = regs.R_MEM
    SP = regs.SP
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    R8 = 8
    R9 = 9
    R10 = 10
    R11 = 11

    # Stack pointer setup: sp_1 = sp + 0x20
    b.add(Insn(Op.ADD, Var(155), [Var(SP), Const(word0(0x20))], EW.W, ea=0x5000))
    # Align mask: t15 = t13 & 0x3FFF0
    b.add(Insn(Op.AND, Var(156), [Var(155), Const(word0(0x3FFF0))], EW.W, ea=0x5004))
    # lqd r9, t15
    b.add(Insn(Op.LOADQ, Var(R9), [Var(R_MEM), Var(156)], ea=0x5008))
    # genctl.w c1
    b.add(Insn(Op.GENCTL, Var(R11), [Var(155)], EW.W, ea=0x500C))
    # shufb r8, 0, r9, c1 (store word 0)
    b.add(Insn(Op.SHUFB, Var(R8), [Const(0), Var(R9), Var(R11)], ea=0x5010))
    # genctl.b c2
    b.add(Insn(Op.GENCTL, Var(R6), [Var(155)], EW.B, ea=0x5014))
    # shufb r4, r3, r8, c2 (store byte 0 = r3)
    b.add(Insn(Op.SHUFB, Var(R4), [Var(R3), Var(R8), Var(R6)], ea=0x5018))
    # stqd r4, t15
    b.add(Insn(Op.STOREQ, Var(R_MEM), [Var(R_MEM), Var(156), Var(R4)], ea=0x501C))

    # FSM test: r10 = fsm.w(r3 == 0)
    b.add(Insn(Op.CMPEQ, Var(R5), [Var(R3), Const(0)], EW.W, ea=0x5020))
    b.add(Insn(Op.FSM, Var(R10), [Var(R5)], EW.W, ea=0x5024))
    # selb r7, 10, 20, r10
    b.add(Insn(Op.SELB, Var(R7), [Const(word0(10)), Const(word0(20)), Var(R10)], EW.W, ea=0x5028))
    b.add(Insn(Op.RET, None, [Var(R_MEM), Var(R7)], ea=0x502C))

    ssa.to_ssa(f)
    opt.optimize(f)
    scalar_stats = scalarize.scalarize(f)
    assert scalar_stats["stores"] >= 2, f"Expected at least 2 scalar stores, got {scalar_stats}"
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info, name_of=lambda ea: "mfc_put" if ea == 0xCF0 else f"sub_{ea:X}")
    code = "\n".join(lines)

    # Verify shufb and genctl are eliminated
    assert "shufb" not in code, f"Unexpected shufb in:\n{code}"
    assert "genctl" not in code, f"Unexpected genctl in:\n{code}"
    assert "fsm" not in code, f"Unexpected fsm in:\n{code}"
    # Verify C ternary
    assert "?" in code and ":" in code, f"Expected ternary in:\n{code}"

    print("ok  test_chained_store_and_fsm_recovery")


def test_primitive_elimination_to_native_c():
    f = Function(0x6000, "test_primitives")
    b = f.new_block(0x6000, 0x6080)
    word0 = lambda w: (w & 0xFFFFFFFF) << 96
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    R8 = 8
    R9 = 9
    R10 = 10
    R11 = 11

    # 1. CLZ: r5 = clz(r3)
    b.add(Insn(Op.CLZ, Var(R5), [Var(R3)], EW.W, ea=0x6000))
    # 2. HALT: halt_if(r3 == 0)
    b.add(Insn(Op.HALT, None, [Var(R3), Const(0)], ea=0x6004, aux="eq"))
    # 3. CG: r6 = cg(r3, r4)
    b.add(Insn(Op.CG, Var(R6), [Var(R3), Var(R4)], EW.W, ea=0x6008))
    # 4. BG: r7 = bg(r3, r4)
    b.add(Insn(Op.BG, Var(R7), [Var(R3), Var(R4)], EW.W, ea=0x600C))
    # 5. ROTM: r8 = rotm(r3, -4) -> logical right shift
    neg_4 = (-4) & 0xFFFFFFFF
    b.add(Insn(Op.ROTM, Var(R8), [Var(R3), Const(word0(neg_4))], EW.W, ea=0x6010))
    # 6. QROTBI: r9 = qrotbi(r3, 5) -> _rotl(r3, 5)
    b.add(Insn(Op.QROTBI, Var(R9), [Var(R3), Const(word0(5))], ea=0x6014))
    # 7. MPYH: r10 = mpyh(r3, r4)
    b.add(Insn(Op.MPYH, Var(R10), [Var(R3), Var(R4)], EW.W, ea=0x6018))
    # 8. Halfword ADD: r11 = add.h(r3, r4)
    b.add(Insn(Op.ADD, Var(R11), [Var(R3), Var(R4)], EW.H, ea=0x601C))
    # Ret
    b.add(Insn(Op.RET, None, [Var(regs.R_MEM), Var(R5), Var(R6), Var(R7), Var(R8), Var(R9), Var(R10), Var(R11)], ea=0x6020))

    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info)
    code = "\n".join(lines)

    # Verify all non-C primitives are eliminated
    assert "__builtin_clz(" in code, f"Expected __builtin_clz in:\n{code}"
    assert "clz(" not in code.replace("__builtin_clz(", ""), f"Unexpected clz intrinsic in:\n{code}"
    assert "__builtin_trap()" in code, f"Expected __builtin_trap in:\n{code}"
    assert "halt_if(" not in code, f"Unexpected halt_if in:\n{code}"
    assert "cg.w" not in code, f"Unexpected cg.w in:\n{code}"
    assert "bg.w" not in code, f"Unexpected bg.w in:\n{code}"
    assert "rotm.w" not in code, f"Unexpected rotm.w in:\n{code}"
    assert "qrotbi" not in code, f"Unexpected qrotbi in:\n{code}"
    assert "mpyh(" not in code, f"Unexpected mpyh in:\n{code}"
    assert "add.h" not in code, f"Unexpected add.h in:\n{code}"
    assert "_rotl" in code, f"Expected _rotl in:\n{code}"
    assert ">> 4" in code, f"Expected '>> 4' in:\n{code}"
    assert "(u16)" in code, f"Expected '(u16)' in:\n{code}"

    print("ok  test_primitive_elimination_to_native_c")


def test_interactive_ux_renaming_and_types():
    """Verify interactive UX: variable renames, argument types, and persistence."""
    import spudec
    from spudec import db, cgen, structure, ssa, opt, outssa

    def word0(v):
        return (v & 0xFFFFFFFF) << 96

    f = Function(0x2548, "authenticator_read_program_header")
    b = f.new_block(0x2548, 0x2560)

    # a1 = r3, a2 = r4, a3 = r5
    # r12 = -4
    # r3 = r12
    # ret
    b.add(Insn(Op.CONST, Var(12), [Const(word0(-4))], EW.W, ea=0x2548, scalar=True))
    b.add(Insn(Op.MOV, Var(3), [Var(12)], EW.W, ea=0x254C, scalar=True))
    b.add(Insn(Op.RET, None, [Var(regs.R_MEM), Var(3), Var(4), Var(5)], ea=0x2550))

    ssa.to_ssa(f)
    opt.optimize(f)
    outssa.lower_phis(f)

    # Set user renames and types in db
    db.set_var_rename(0x2548, "r3", "result")
    db.set_var_rename(0x2548, "a3", "phdr")
    db.set_var_type(0x2548, "result", "int")
    db.set_func_proto(0x2548, "qword authenticator_read_program_header(vec_uint4 *a1, vec_uint4 a2, Elf64_Phdr *phdr)")

    # Call pseudocode() on f directly without arguments - it should auto-query db!
    lines = spudec.pseudocode(f)
    code = "\n".join(lines)

    # Verify prototype
    assert "Elf64_Phdr *phdr" in code, f"Expected 'Elf64_Phdr *phdr' in:\n{code}"
    # Verify declaration of renamed and typed variable
    assert "int result;" in code, f"Expected 'int result;' in:\n{code}"
    # Verify return
    assert "return result;" in code, f"Expected 'return result;' in:\n{code}"

    # Verify CodeLines metadata
    assert getattr(lines, "sig_line", None) is not None
    assert getattr(lines, "func_name", None) == "authenticator_read_program_header"
    assert getattr(lines, "ret_type", None) == "qword"
    assert "phdr" in getattr(lines, "params", [])
    assert getattr(lines, "param_types", {}).get("phdr") == "Elf64_Phdr *"
    assert getattr(lines, "var_types", {}).get("result") == "int"

    print("ok  test_interactive_ux_renaming_and_types")


def test_tail_call_argument_preservation():
    # Build IR model of boot_lv0 calling mfc_put
    f = Function(0xD60, "boot_lv0")
    b0 = f.new_block(0xD60, 0xDD0)
    b1 = f.new_block(0xCF0, 0xCF0)
    b0.succs.append(b1)
    b1.preds.append(b0)

    # b0: r3 holds a1
    # Save a1 in r80, r3 = 0, call mfc_put(0, 0x3600)
    b0.add(Insn(Op.MOV, Var(80), [Var(3)], EW.W, ea=0xD70))
    b0.add(Insn(Op.CONST, Var(3), [Const(0)], EW.W, ea=0xD74))
    b0.add(Insn(Op.CONST, Var(4), [Const(0x3600 << 96)], EW.W, ea=0xD60))
    b0.add(Insn(Op.CALL, Var(regs.R_MEM), [Var(regs.R_MEM), Var(regs.R_CH), Var(3), Var(4)], ea=0xD7C, aux=0xCF0))

    # Restore r3 = r80, call sub_FD80(a1)
    b0.add(Insn(Op.MOV, Var(3), [Var(80)], EW.W, ea=0xD80))
    b0.add(Insn(Op.CALL, Var(regs.R_MEM), [Var(regs.R_MEM), Var(regs.R_CH), Var(3)], ea=0xD84, aux=0xFD80))
    # Return value from call in r3
    b0.add(Insn(Op.UNDEF, Var(3), [], ea=0xD84))

    # Stack variable var_20 address
    b0.add(Insn(Op.ADD, Var(24), [Var(regs.SP), Var(80)], EW.W, ea=0xD88)) # slot var_20

    # Stores into var_20
    b0.add(Insn(Op.STORE, Var(regs.R_MEM), [Var(regs.R_MEM), Var(24), Var(3)], EW.B, ea=0xDAC))

    # Compute EA 4 into r3
    b0.add(Insn(Op.CONST, Var(3), [Const(4 << 96)], EW.W, ea=0xDB8))

    # Load from var_20 into r4
    b0.add(Insn(Op.LOADQ, Var(4), [Var(regs.R_MEM), Var(24)], ea=0xDC4))

    # Jump to b1
    b0.add(Insn(Op.JMP, None, [], ea=0xDCC, aux=0xCF0))

    # b1: Tail call to mfc_put
    tail_abi = [Var(regs.R_MEM), Var(regs.R_CH), Var(3), Var(4)]
    b1.add(Insn(Op.TCALL, None, tail_abi, ea=0xCF0, aux=0xCF0))

    ssa.to_ssa(f)
    opt.optimize(f)
    stmts, info = structure.structure(f)
    names = {0xD60: "boot_lv0", 0xCF0: "mfc_put", 0xFD80: "sub_FD80"}
    arities = {0xCF0: 2, 0xFD80: 1}
    lines = cgen.generate(f, stmts, info, name_of=lambda ea: names.get(ea, "sub_%X" % ea),
                          arity_of=lambda ea: arities.get(ea, 0))
    code = "\n".join(lines)

    assert "goto mfc_put;" not in code, f"Unexpected 'goto mfc_put;' in:\n{code}"
    assert "outside this function" not in code, f"Unexpected 'outside this function' in:\n{code}"
    # Verify mfc_put is called with 2 arguments (4, ...)
    assert "mfc_put(0x4," in code or "mfc_put(4," in code, f"Expected 'mfc_put(0x4, ...)' in:\n{code}"

    print("ok  test_tail_call_argument_preservation")


def test_tier2_3_improvements():
    """Verify Tier 2/3 enhancements: stack array indexing, phi store hoisting, and constant slots."""
    from spudec import cgen, structure, ssa, opt, scalarize
    from spudec.frame import Frame

    def word0(v):
        return (v & 0xFFFFFFFF) << 96

    # 1. Test indexed stack array access:
    # Frame size 0x120; var_100 is at sp - 0x100 (which is sp_cur + 0x20).
    # Computing (r21 + 0x20) + sp_cur should format as (char *)&var_100 + r21.
    f = Function(0x4520, "cellCryptoSpuAesEncKeySet")
    f.frame = Frame()
    f.frame.size = 0x120
    b0 = f.new_block(0x4520, 0x4550)
    # prologue
    b0.add(Insn(Op.ADD, Var(regs.SP), [Var(regs.SP), Const(word0(-0x120))], EW.W, ea=0x4520))
    # r21 = r4 (index)
    b0.add(Insn(Op.MOV, Var(21), [Var(4)], EW.W, ea=0x4524))
    # r53 = r21 + 0x20
    b0.add(Insn(Op.ADD, Var(53), [Var(21), Const(word0(0x20))], EW.W, ea=0x4528))
    # t105 = r53 + sp
    b0.add(Insn(Op.ADD, Var(105), [Var(53), Var(regs.SP)], EW.W, ea=0x452C))
    # t106 = t105 & 0x3FFF0
    b0.add(Insn(Op.AND, Var(106), [Var(105), Const(word0(0x3FFF0))], EW.W, ea=0x4530))
    # storeq mem, t106, r18
    b0.add(Insn(Op.STOREQ, Var(regs.R_MEM), [Var(regs.R_MEM), Var(106), Var(18)], ea=0x4534))
    b0.add(Insn(Op.RET, None, [Var(regs.R_MEM), Var(regs.LR)], ea=0x4538))

    ssa.to_ssa(f)
    opt.optimize(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info)
    code = "\n".join(lines)
    assert "(char *)&var_100" in code or "&var_100" in code, f"Expected '&var_100' in:\n{code}"
    assert "sp_2" not in code, f"Unexpected 'sp_2' in:\n{code}"

    # 2. Test AES inverse mix columns intrinsic
    f2 = Function(0x4920, "cellCryptoSpuAesDecKeySet")
    b2 = f2.new_block(0x4920, 0x4940)
    # r15 = load from CAE0
    b2.add(Insn(Op.CONST, Var(15), [Const(word0(0xCAE0))], EW.W, ea=0x4920))
    # r10 = shufb(r3, 0, r15)
    b2.add(Insn(Op.CONST, Var(0), [Const(0)], EW.W, ea=0x4924))
    b2.add(Insn(Op.SHUFB, Var(10), [Var(3), Var(0), Var(15)], ea=0x4928))
    b2.add(Insn(Op.RET, None, [Var(regs.R_MEM), Var(regs.LR), Var(10)], ea=0x492C))

    ssa.to_ssa(f2)
    opt.optimize(f2)
    stmts2, info2 = structure.structure(f2)
    lines2 = cgen.generate(f2, stmts2, info2)
    code2 = "\n".join(lines2)
    assert "aes_inv_mix_columns" in code2, f"Expected 'aes_inv_mix_columns' in:\n{code2}"

    print("ok  test_tier2_3_improvements")


def test_aes_and_sha_shuffle_elimination_to_native_c():
    """Verify that AES S-box trees, RotWord, MixColumns, SHA-1 extracts, and splats become native C with 0 __builtin_shuffle."""
    from spudec.ir import Function, Insn, Op, Var, Const, EW
    from spudec import ssa, opt, structure, cgen

    word0 = lambda w: (w & 0xFFFFFFFF) << 96

    f = Function(0x7000, "test_crypto_shuf")
    b = f.new_block(0x7000, 0x7200)

    # 1. AES S-box tree on r3:
    # Mask = r3 & 0x1F1F1F1F
    R3, R_MASK = 3, 100
    b.add(Insn(Op.AND, Var(R_MASK), [Var(R3), Const(word0(0x1F1F1F1F))], EW.W, ea=0x7000))

    # 8 SHUFBs (regs 110..117)
    for i in range(8):
        b.add(Insn(Op.SHUFB, Var(110 + i), [Const(0), Const(0), Var(R_MASK)], ea=0x7004 + i*4))

    # Level 1: 4 SELBs (regs 120..123)
    for i in range(4):
        b.add(Insn(Op.SELB, Var(120 + i), [Var(110 + 2*i), Var(110 + 2*i + 1), Const(0)], EW.Q, ea=0x7024 + i*4))

    # Level 2: 2 SELBs (regs 124..125)
    b.add(Insn(Op.SELB, Var(124), [Var(120), Var(121), Const(0)], EW.Q, ea=0x7034))
    b.add(Insn(Op.SELB, Var(125), [Var(122), Var(123), Const(0)], EW.Q, ea=0x7038))

    # Level 3: Root SELB (reg 12)
    b.add(Insn(Op.SELB, Var(12), [Var(124), Var(125), Const(0)], EW.Q, ea=0x703C))

    # 2. RotWord: r4 rotated left by 8 via mask 0x1D1E1F1C
    b.add(Insn(Op.SHUFB, Var(13), [Const(0), Var(4), Const(0x1D1E1F1C1D1E1F1C1D1E1F1C1D1E1F1C)], ea=0x7040))

    # 3. MixColumns: r5 rotated row 2 via mask loaded from 0xCB00
    b.add(Insn(Op.LOADQ, Var(50), [Var(regs.R_MEM), Const(word0(0xCB00))], EW.Q, ea=0x7044))
    b.add(Insn(Op.SHUFB, Var(14), [Var(5), Const(0), Var(50)], ea=0x7048))

    # 4. SHA-1 word extract: r6[1] via mask 0x04050607
    b.add(Insn(Op.SHUFB, Var(15), [Var(6), Const(0), Const(0x04050607040506070405060704050607)], ea=0x704C))

    # 5. Memset byte splat: r7 splatted via mask 0x03030303
    b.add(Insn(Op.SHUFB, Var(16), [Var(7), Var(7), Const(0x03030303030303030303030303030303)], ea=0x7050))

    b.add(Insn(Op.RET, None, [Var(regs.R_MEM), Var(12), Var(13), Var(14), Var(15), Var(16)], ea=0x7054))

    ssa.to_ssa(f)
    opt.optimize(f)
    stmts, info = structure.structure(f)
    lines = cgen.generate(f, stmts, info)
    code = "\n".join(lines)

    assert "__builtin_shuffle" not in code, f"Unexpected __builtin_shuffle in:\n{code}"
    assert "aes_sub_bytes" in code, f"Expected aes_sub_bytes in:\n{code}"
    assert "_rotl(" in code, f"Expected _rotl in:\n{code}"
    assert "aes_mix_cols_rot" in code, f"Expected aes_mix_cols_rot in:\n{code}"
    assert "[1]" in code, f"Expected word extract [1] in:\n{code}"
    assert "spu_splats(" in code, f"Expected spu_splats in:\n{code}"

    print("ok  test_aes_and_sha_shuffle_elimination_to_native_c")


if __name__ == "__main__":
    test_db_persistence()
    test_proto_parser()
    test_cgen_user_overrides()
    test_codelines_mapping()
    test_stack_local_variables()
    test_condition_idioms()
    test_symbol_resolution()
    test_active_call_args()
    test_extended_arithmetic_simplification()
    test_instruction_translation_to_c()
    test_chained_store_and_fsm_recovery()
    test_primitive_elimination_to_native_c()
    test_interactive_ux_renaming_and_types()
    test_tail_call_argument_preservation()
    test_tier2_3_improvements()
    test_aes_and_sha_shuffle_elimination_to_native_c()
    print("\nall recover tests passed")
