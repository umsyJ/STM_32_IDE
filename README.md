# STM32 Block IDE

A Simulink-style visual programming environment for STM32 microcontrollers. Drag blocks onto a canvas, wire them together, set parameter values from a Python workspace, and click ▶ to compile and flash the resulting firmware to the board.

This is an early-stage prototype: the block library is intentionally small (Square Wave, GPIO In, GPIO Out, Scope) and only the NUCLEO-F446RE is supported. Both are easy to extend — see the bottom of `BLOCK_REFERENCE.md` for the extension points.

## What's in this folder

`stm32_block_ide.py` is the main GUI application. Run it with `python stm32_block_ide.py` to open the editor.

`code_templates.py` contains the C code generator, the board pin maps, and the build/flash Makefile template. It's a regular Python module that the GUI imports; you can also call `generate_project()` from your own scripts.

`INSTALLATION.md` lists every prerequisite (Python packages, ARM cross-compiler, STM32CubeF4 HAL drivers, st-flash) with copy-pasteable install commands for Linux, macOS, and Windows.

`GETTING_STARTED.md` is a five-minute walk-through that takes you from a blank canvas to a blinking LED on the Nucleo, demonstrating block placement, wiring, workspace variables, host simulation, and flashing.

`BLOCK_REFERENCE.md` documents every block — what each port and parameter does, what the generated C looks like, and where the rough edges are.

## Architecture at a glance

The block diagram is a directed graph stored in JSON. Each node is a `BlockInstance` with a type, an ID, a position, and a dict of parameter strings. Each edge is a `Connection` from `(src_block, src_port)` to `(dst_block, dst_port)`.

When you click Build, the generator topologically sorts the blocks, evaluates every parameter expression against the Python workspace, and emits a `main.c` plus a Makefile under `build_stm32_ide/current_project/`. The Makefile compiles against the STM32 HAL drivers (CubeF4) using `arm-none-eabi-gcc` and produces a `.bin`. The Run button then invokes `make flash`, which calls `st-flash` to write the binary to address `0x08000000`.

The Scope block, when its `stream` parameter is set, appends a comma-separated line of float samples to USART2 every model step. The IDE's Scope tab opens the corresponding USB virtual COM port and plots the values live with pyqtgraph.

The Python Workspace tab is a small REPL backed by a shared `Workspace` object. Anything you bind there (`Ts = 0.001`, `import numpy as np; sweep = np.linspace(0,1,100)`) is visible to every parameter expression on every block. This is the IDE's analogue of the MATLAB base workspace, except it's actually Python.
