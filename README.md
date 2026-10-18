# Custom Instruction Set Extension for Int8 Softmax Acceleration on Scalar RISC-V

Multidisciplinary Project, MSc in Computer Science and Engineering, Politecnico di Milano (A.Y. 2025-26).
Author: Luciano Stupnik. Advisor: Tommaso Spagnolo. Co-advisor: Prof. Cristina Silvano.

Full report: [`src/softmax/report.pdf`](src/softmax/report.pdf) (6 pages).

This repository is a fork of [gem5](https://github.com/gem5/gem5) v25.0. The work described here lives in
`src/softmax/` and in a small number of simulator files listed under [Repository layout](#repository-layout).
Everything else is upstream gem5, distributed under its original license; the upstream README is kept as
[`README.gem5.md`](README.gem5.md).

## Summary

Softmax is evaluated in every self-attention block of a Transformer. As matrix multiplications are increasingly
served by dedicated MAC arrays, non-linear operations such as softmax become a larger share of the execution
time, especially on scalar cores running 8-bit quantized models. This project proposes a custom RISC-V
instruction, `OnlineDM`, together with a pipelined functional unit in the gem5 O3 core, that accelerates the
online max/denominator accumulation of an integer-only softmax. In simulation, with a three-cycle unit, the instruction reduces
execution time by 21.6% to 35.7% with respect to an integer-only software baseline, for vectors of 1,024 to
32,768 elements.

## Method

**Integer-only softmax.** Following ITA [3], inputs are quantized to B = 8 bits and the exponential is
rewritten in base 2, so that each term becomes 2<sup>−k</sup>, where k is obtained by rounding the scaled
exponent difference to the nearest integer (bucketization, with K<sub>A</sub> = B − log<sub>2</sub>B = 5).
In Q0.7 fixed point, 2<sup>−k</sup> is a right shift. The maximum and the denominator are computed in a single
pass, renormalizing the partial sum whenever a new maximum appears, as in Softermax [4]. A single fixed-point
reciprocal of the denominator is then used to normalize all outputs. The resulting kernel uses only
subtractions, shifts and fixed-point multiplications.

**Bottleneck analysis.** The kernel was cross-compiled for RISC-V and simulated in gem5. The integer ALU is the
only functional unit with significant occupancy (68.28% of cycles for 1,024 elements), and ADD-like and
SHIFT-like operations account for most of its work (Table 1 of the report). These operations are concentrated
in the online accumulation loop, which is therefore the target of the custom instruction.

**The `OnlineDM` instruction.** `OnlineDM` is an R-type instruction in the *custom-0* opcode space
(`0x0B`, funct3 = 0, funct7 = 0). One invocation consumes up to four int8 elements and updates the running
state of the accumulation:

- `rs1` holds the state: the number of valid elements (1–4) in bits 31–24, the running maximum (int8) in
  bits 23–16 and the partial denominator (Q0.7, uint16) in bits 15–0.
- `rs2` holds the data: four int8 elements in bits 31–24, 23–16, 15–8 and 7–0.

Because the encoding uses a reserved opcode, the instruction can be emitted with the standard GNU assembler
directive, without modifying the toolchain:

```c
.insn r 0x0b, 0, 0, rd, rs1, rs2
```

**Functional unit.** A new pipelined unit, `DMUpdateALU` (OpClass `DMUpdateOp`), was added to the execute stage
of the O3 core. It is modeled with three stages: (1) unpacking, masking of invalid lanes and the two pairwise
maxima; (2) local maximum, comparison with the running maximum and rescaling of the partial sum; (3) the four
2<sup>−k</sup> contributions and an adder tree. The latency is a parameter of the model.

## Experimental setup

gem5 v25.0 in syscall-emulation (SE) mode; single-core out-of-order (O3) RISC-V CPU at 1 GHz; private 32 kB L1
instruction and data caches; 256 kB L2; single-channel DDR3-1600 with 2 GB (`src/softmax/run_config.py`).
Both kernels were cross-compiled with `riscv64-linux-gnu-gcc` at the default optimization level (`-O0`).
Execution time is reported in simulated ticks.

## Results

Simulated execution time (SimTicks × 10<sup>6</sup>) and improvement over the integer-only baseline:

| Implementation | 1,024 elements | 8,192 elements | 32,768 elements |
|---|---|---|---|
| Baseline (integer-only, no custom instruction) | 134.77 | 732.52 | 2,778.18 |
| `OnlineDM`, latency 2 | 105.39 (21.80%) | 482.45 (34.14%) | 1,778.49 (35.98%) |
| `OnlineDM`, latency 3 | 105.63 (21.62%) | 484.74 (33.83%) | 1,786.73 (35.69%) |
| `OnlineDM`, latency 4 | 105.89 (21.43%) | 486.69 (33.56%) | 1,794.93 (35.39%) |
| `OnlineDM`, latency 8 | 106.88 (20.69%) | 494.69 (32.47%) | 1,827.43 (34.22%) |

Two observations follow from these measurements:

- The improvement grows with the vector size, but with diminishing returns: between 8,192 and 32,768 elements
  it increases by less than two percentage points for every latency.
- The result is largely insensitive to the latency of the unit. Going from 2 to 8 cycles reduces the
  improvement by at most 1.8 points, because the instruction is executed only ⌈(N − 1)/4⌉ times and represents
  a small fraction of the whole kernel. This leaves room to choose a simpler hardware implementation.

## Limitations

- Results come from a cycle-level simulator; no RTL implementation was built, so area and power were not
  evaluated.
- Both kernels were compiled at `-O0`. The improvement over an optimized baseline (`-O2`/`-O3`) is likely to be
  smaller and was not measured.
- The evaluation measures execution time only. The numerical accuracy of the integer-only softmax with respect
  to a floating-point reference was not part of this study.

## Reproducing the experiments

```sh
# 1. Build gem5 for RISC-V (standard gem5 build).
scons build/RISCV/gem5.opt -j$(nproc)

# 2. Cross-compile the kernels (requires gcc-riscv64-linux-gnu).
#    The vector length is set with N in each source file.
cd src/softmax
./build_lm.sh C/softmax_int8.c          # baseline
./build_lm.sh C/softmax_int8_custom.c   # with OnlineDM
cd ../..

# 3. Run each binary in SE mode. Execution time and FU occupancy are in m5out/<name>/stats.txt.
build/RISCV/gem5.opt -d m5out/softmax_int8_custom \
    --debug-flags=Exec --debug-file=exec.log \
    src/softmax/run_config.py --binary src/softmax/bin/softmax_int8_custom.riscv

# 4. Instruction mix (OpClass distribution and IntAlu breakdown) from the execution trace.
python3 src/softmax/stats.py m5out/softmax_int8_custom/exec.log
```

The latency of the unit is set by `opLat` in `src/cpu/o3/FuncUnitConfig.py` (3 by default) and requires
rebuilding gem5.

## Repository layout

| Path | Contents |
|---|---|
| `src/arch/riscv/isa/decoder.isa` | Decoding and semantics of `OnlineDM` (`dm_update`) |
| `src/cpu/op_class.hh`, `src/cpu/FuncUnit.py` | New OpClass `DMUpdateOp` |
| `src/cpu/o3/FuncUnitConfig.py`, `src/cpu/o3/FUPool.py` | `DMUpdateALU` and its inclusion in the O3 FU pool |
| `src/softmax/C/softmax_int8.c` | Integer-only softmax, baseline |
| `src/softmax/C/softmax_int8_custom.c` | Same kernel using `OnlineDM` |
| `src/softmax/build_lm.sh`, `run_config.py`, `stats.py` | Cross-compilation, simulation and statistics |
| `src/softmax/stats/` | Raw statistics of the reported runs |
| `src/softmax/report.pdf` | Project report |

## References

1. N. Binkert et al. The gem5 simulator. *ACM SIGARCH Computer Architecture News*, 39:1–7, 2011.
2. W. Chang and I. Vasyltsov. Efficient softmax approximation for deep neural networks with attention
   mechanism. arXiv:2111.10770, 2021.
3. G. Paulin et al. ITA: An energy-efficient attention and softmax accelerator for quantized transformers.
   arXiv:2307.03493, 2023.
4. J. R. Stevens et al. Softermax: Hardware/software co-design of an efficient softmax for transformers.
   arXiv:2103.09301, 2021.
