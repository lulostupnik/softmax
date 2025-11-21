#!/usr/bin/env python3
import re
import sys
from collections import Counter

LOG_FILE = "./m5out/softmax_int8_no_lut/exec.log"
if len(sys.argv) > 1:
    LOG_FILE = sys.argv[1]

# Regex para mnemonic y OpClass
mnemonic_re = re.compile(r"@\S+\s+:\s+([a-z0-9_.]+)\s")
opclass_re = re.compile(r":\s+([A-Za-z]+)\s*:")

mnemonic_hist = Counter()
opclass_hist = Counter()

# Subcategorías dentro de IntAlu
total_int_alu = 0
add_like = 0
shift_like = 0
logic_like = 0
branch_like = 0
move_like = 0
compare_like = 0
other_int_alu = 0

# Listas de mnemonics por categoría
branch_mnems = {
    "beq",
    "bne",
    "blt",
    "bge",
    "bltu",
    "bgeu",
    "jal",
    "jalr",
    "c_beqz",
    "c_bnez",
    "c_j",
    "c_jal",
}

move_mnems = {
    "mv",
    "c_mv",
}


def is_add_like(mn):
    # suma/resta, incluyendo comprimidas
    return mn.startswith(("add", "sub", "c_add"))


def is_shift_like(mn):
    # shifts normales y comprimidos
    if mn.startswith(("sll", "srl", "sra")):
        return True
    if mn in ("c_slli", "c_srli"):
        return True
    return False


def is_logic_like(mn):
    # AND/OR/XOR y variantes inmediatas (incl. comprimidas)
    if mn.startswith(("and", "or", "xor")):
        return True
    if mn in ("c_andi",):
        return True
    return False


def is_branch_like(mn):
    return mn in branch_mnems


def is_move_like(mn):
    return mn in move_mnems


def is_compare_like(mn):
    # slt, sltu, slti, sltiu, etc.
    return mn.startswith("slt")


with open(LOG_FILE) as f:
    for line in f:
        m_class = opclass_re.search(line)
        if not m_class:
            continue
        opclass = m_class.group(1)
        opclass_hist[opclass] += 1

        m_mn = mnemonic_re.search(line)
        if not m_mn:
            continue
        mn = m_mn.group(1)
        mnemonic_hist[mn] += 1

        if opclass != "IntAlu":
            continue

        total_int_alu += 1

        if is_add_like(mn):
            add_like += 1
        elif is_shift_like(mn):
            shift_like += 1
        elif is_logic_like(mn):
            logic_like += 1
        elif is_branch_like(mn):
            branch_like += 1
        elif is_move_like(mn):
            move_like += 1
        elif is_compare_like(mn):
            compare_like += 1
        else:
            other_int_alu += 1


# -------- OUTPUT --------

print("==== OpClass counts ====")
total_ops = sum(opclass_hist.values())
for cls, cnt in opclass_hist.most_common():
    pct = 100.0 * cnt / total_ops if total_ops > 0 else 0.0
    print(f"{cls:12s} {cnt:10d}   ({pct:5.2f}%)")

print("\n==== IntAlu breakdown (detailed) ====")
print(f"TOTAL IntAlu     : {total_int_alu}")
print(f"  ADD-like       : {add_like}")
print(f"  SHIFT-like     : {shift_like}")
print(f"  LOGIC          : {logic_like}")
print(f"  BRANCH         : {branch_like}")
print(f"  MOVE           : {move_like}")
print(f"  COMPARE        : {compare_like}")
print(f"  OTHER          : {other_int_alu}")

if total_int_alu > 0:
    print()
    print(f"  %ADD           : {100.0*add_like/total_int_alu:5.2f}%")
    print(f"  %SHIFT         : {100.0*shift_like/total_int_alu:5.2f}%")
    print(f"  %LOGIC         : {100.0*logic_like/total_int_alu:5.2f}%")
    print(f"  %BRANCH        : {100.0*branch_like/total_int_alu:5.2f}%")
    print(f"  %MOVE          : {100.0*move_like/total_int_alu:5.2f}%")
    print(f"  %COMPARE       : {100.0*compare_like/total_int_alu:5.2f}%")
    print(f"  %OTHER         : {100.0*other_int_alu/total_int_alu:5.2f}%")

print("\n==== Top 20 mnemonics ====")
for name, cnt in mnemonic_hist.most_common(20):
    print(f"{name:12s} {cnt}")
