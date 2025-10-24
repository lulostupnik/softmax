#!/usr/bin/env python3
import argparse
import os

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

parser = argparse.ArgumentParser(description="Run a RISC-V binary in gem5 SE mode.")
parser.add_argument("--binary", required=True, help="Path to local RISC-V ELF")
args = parser.parse_args()

binary_path = os.path.abspath(args.binary)

# CPU, caches, memory
processor = SimpleProcessor(cpu_type=CPUTypes.TIMING, num_cores=1, isa=ISA.RISCV)
cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="32kB", l1i_size="32kB", l2_size="256kB"
)
memory = SingleChannelDDR3_1600(size="2GB")

# Board (clk_freq required in v25)
board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

# Set SE binary workload (v25+ API)
board.set_se_binary_workload(BinaryResource(binary_path))

sim = Simulator(board=board)
sim.run()
