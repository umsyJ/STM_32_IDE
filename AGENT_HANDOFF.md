# STM32 Block IDE — Agent Handoff

Brief for the next AI agent picking up this project. Read this before you touch any code. Written by the previous agent at the end of a long session.

## What this project is

A Simulink-like visual programming environment for STM32 microcontrollers, built in Python (PyQt5 + pyqtgraph). The user drags block diagrams onto a canvas, the IDE generates a complete STM32 HAL C project, compiles it with `arm-none-eabi-gcc`, and flashes it to the board via `st-flash`. A live Scope tab reads telemetry from the MCU over USART2 and plots it.

The user is Umer, an engineer new to STM32 embedded programming. He's running this on Windows against a NUCLEO-F446RE board that's physically connected via ST-Link VCP (shows up as COM3 on his machine).

## Current capabilities (working end-to-end as of this session)

- **Visual editor**: drag blocks, wire ports, edit parameters
- **7 block types** in the library (see Block Library section)
- **Codegen → compile → flash pipeline**: click "Run on Board", firmware lands on the MCU
- **Live scope** with configurable time window (0.5–600 s), auto-Y axis scaling, 30 Hz repaint timer, pyqtgraph downsampling
- **Host-side simulator** (`simulate_model`) that runs the model in Python without hardware, for "what-if" testing
- **Model save/load** as JSON
- **MATLAB-like workspace** tab (separate module) for defining variables used in block parameters

## File layout

```
C:\Users\Umer\Downloads\STM_32_IDE\
├── stm32_block_ide.py      # Main IDE app — ~1085 lines, QMainWindow, BLOCK_CATALOG,
│                             ScopeTab, BlockPalette, simulate_model, BuildFlashWorker
├── code_templates.py       # C-code generator — ~755 lines, generate_project(),
│                             _emit_decls/_emit_init/_emit_step/_emit_helpers,
│                             MAIN_C_TEMPLATE, MAKEFILE_TEMPLATE, HAL_CONF_H, _LD_STUB
├── workspace_shared.py     # Singleton WORKSPACE with eval_param() for parameter strings
├── matlab_workspace.py     # MATLAB-style workspace tab widget
├── test_blocks.py          # Unit tests (added this session) — plain Python + pytest compatible
├── BLOCK_REFERENCE.md
├── GETTING_STARTED.md
├── INSTALLATION.md
└── README.md

C:\Users\Umer\build_stm32_ide\
├── current_project\        # Latest generated STM32 project (regenerated on each build)
│   ├── main.c
│   ├── Makefile
│   ├── stm32f4xx_hal_conf.h
│   ├── STM32F446RETx_FLASH.ld
│   └── model.json
└── uart_test\              # Diagnostic binary created earlier in the session;
                              blinks LD2 at 2 Hz + streams "HELLO FROM STM32" every 250ms.
                              Used to confirm the ST-Link VCP path works.
```

External dependencies the user has installed:
- Python 3.x with `PyQt5`, `pyqtgraph`, `pyserial`, `numpy`
- `arm-none-eabi-gcc` (GNU Arm Embedded Toolchain)
- STM32CubeF4 firmware package (`CUBE_F4` env var auto-detected; path varies)
- `st-flash` from stlink-tools at `C:\Tools\stlink\bin` on PATH

## Block Library (as of end of session)

| Type       | Inputs          | Outputs | Purpose |
|------------|-----------------|---------|---------|
| SquareWave | —               | y       | Host-step-rate square wave generator |
| GpioIn     | —               | y       | Digital input pin (PC13 = user button default) |
| GpioOut    | u               | —       | Digital output pin (PA5 = LD2 green LED default) |
| Scope      | u0, u1, u2      | —       | Streams up to 3 signals over USART2 @ 115200 to host |
| Ultrasonic | —               | d       | HC-SR04 range finder; outputs distance in **meters**. TRIG/ECHO pins configurable (PA0/PA1 default) |
| Sum        | u0, u1          | y       | y = u0 + u1. Unconnected inputs contribute 0.0f. Added late in session. |
| Product    | u0, u1          | y       | y = u0 * u1. Unconnected inputs contribute 1.0f (multiplicative identity). Added late in session. |

The user explicitly told me "Nevermind. I will ask for blocks as I need them" when I offered to expand the library Simulink-style. **Do not proactively add more blocks** unless asked. He prefers incremental additions on-demand. (Sum and Product were added by the linter/user shortly after that request — treat them as the current canonical set.)

Sum/Product are fully wired: BLOCK_CATALOG entries, `simulate_model` cases (executed in `_ct_topo_order` so Sum→Product chains resolve correctly), `_emit_decls` declares `sig_<id>_y`, and `_emit_step` emits `sig_<id>_y = u0 + u1;` / `u0 * u1;` with the identity-element fallback for unconnected inputs.

## What got built/fixed this session (reverse chronological)

### Sum and Product math blocks
Added after the handoff doc was first written. Two-input passthrough-arithmetic blocks. Simulator runs them in topological order (via `_ct_topo_order`) so cascaded Sum → Product chains resolve in a single pass. Codegen uses the identity element of each op for unconnected inputs (0 for Sum, 1 for Product). No hardware wiring, no `_emit_init`, no `_emit_helpers` — pure signal math.

### Unit test suite (`test_blocks.py`)
Covers catalog integrity, simulator numerics, `_topo_order`, `_wires`, `_emit_decls`, `_emit_init`, `_emit_step`, `_emit_helpers`, `generate_project` end-to-end, JSON round-trip. Uses plain-Python `assert` + a discovery runner at the bottom; also runs under pytest. The user/linter has since expanded it — module-level docstring lists the added sections (connections, workspace eval, `_describe`, simulator edge cases, codegen completeness). Treat the expanded version as canon.

Run it: `python test_blocks.py`

### Scope improvements
- **Time window control**: QDoubleSpinBox "Window (s)" in the toolbar, 0.5–600 s, default 5 s. Trims data older than `t_latest - window` and pins the X-range accordingly. `_on_window_changed` also retrims existing buffers and redraws immediately.
- **Auto-Y-scale checkbox**: "Auto-Y" checkbox, default on. Rescales Y to `[min, max]` of visible samples with 8% margin on each repaint. Handles flat-trace case with arbitrary `±1` padding so you still see the line.
- **Decoupled sample and repaint rates**: `_on_sample` now just appends and sets a `_dirty` flag. A `QTimer` at 30 Hz (`_tick_repaint`) does the window trimming and pyqtgraph redraw. This was the fix for "PuTTY updates faster than the scope" — per-sample repaints were starving the Qt event loop.
- **pyqtgraph optimizations**: `setClipToView(True)` + `setDownsampling(auto=True, mode="peak")` on the PlotWidget.

### Ultrasonic (HC-SR04) block
Full new block type with:
- BlockSpec entry in `BLOCK_CATALOG` (params: `trig_pin`, `echo_pin`, `period_ms`, `timeout_us`)
- Simulator case in `simulate_model` that returns zeros by default or reads a `ultrasonic_<id>` workspace override (scalar → broadcast, array → sliced)
- Codegen: `_emit_decls` emits `sig_<id>_d` float. `_emit_init` configures TRIG as push-pull output, ECHO as input with no pull, calls `ultrasonic_init()` once. `_emit_step` emits a per-block static counter (`us_cnt_<id>`) that gates the `ultrasonic_measure_m()` call at the configured `period_ms`.
- New `_emit_helpers` function emits the driver code only when at least one Ultrasonic block is present. Uses DWT cycle counter for microsecond timing (no timer peripheral required). Speed of sound constant: `171.5 f` (343 m/s halved for one-way).
- New `{helpers}` placeholder in `MAIN_C_TEMPLATE`.

Wiring on the board (the user asked; confirmed safe):
- HC-SR04 VCC → NUCLEO 5V
- HC-SR04 GND → NUCLEO GND
- HC-SR04 TRIG → NUCLEO PA0 (direct)
- HC-SR04 ECHO → NUCLEO PA1 (direct — **PA1 is 5V-tolerant in digital input mode**; no divider needed. I initially told him to use a divider; that was wrong. PA0/PA1 are both "FT" pins on F446RE.)

### Critical build fixes (earlier in session)
- Scope was dead silent with the `-specs=nano.specs` linker option. Nano libc's `printf`/`snprintf` doesn't include float formatting by default. Fix: `-u _printf_float` added to `LDFLAGS` in `MAKEFILE_TEMPLATE`. This is a **regression-guarded invariant** — `test_generate_project_makefile_has_printf_float_flag` enforces it.
- Missing `stm32f4xx_hal_conf.h` caused the first builds to fail. CubeF4 ships a template but not an actual conf header. Added `HAL_CONF_H` constant + writes it to project dir in `generate_project`. Enables only the modules we compile (CORTEX, RCC, GPIO, PWR, DMA, FLASH, UART). Another regression-guarded invariant.
- Missing FLASH macros → added `HAL_FLASH_MODULE_ENABLED`, include, and flash/flash_ex/flash_ramfunc sources to Makefile.
- Missing DMA source files → `stm32f4xx_hal_dma.c` and `stm32f4xx_hal_dma_ex.c` added to Makefile sources.
- Diagnostic binary `uart_test/main.c` missing `SysTick_Handler` → CPU trapped in `Default_Handler` after first SysTick. Fix: added `void SysTick_Handler(void) { HAL_IncTick(); }`.
- `_LD_STUB` triple-quoted string was truncated mid-word from an earlier Edit tool/mount sync mishap. Appended `RM.attributes) }\n}\n"""` to close it. File tail should end with a closing `}` and `"""`.

## Known infrastructure quirks (NOT code bugs)

### Windows ↔ bash mount divergence
**This bit us repeatedly.** The session has a Linux bash sandbox that mounts the user's Windows folders at `/sessions/.../mnt/STM_32_IDE/`. The Windows `Edit`/`Write`/`Read` tools see the true current file. The bash-side view **lags** and sometimes shows a stale, truncated version — e.g. Windows sees 755 lines, bash `wc -l` reports 626.

Symptoms:
- `python3 -m py_compile` via bash fails with `SyntaxError: unterminated triple-quoted string literal` at a line that's perfectly well-terminated in the Windows view
- `cat /path/to/file | tail` shows content from hours ago
- `.pyc` caches in `__pycache__/` may let imports succeed against the stale source — confusing

Workarounds when you need bash compilation/test runs:
1. Trust the Windows Read/Edit tools as source of truth.
2. If bash is stuck, make a tiny Windows-side Edit (e.g. a comment change) and wait a few seconds — sometimes forces resync.
3. As a last resort, `Write` the entire file to force a rewrite.
4. The user's actual IDE runs via `python stm32_block_ide.py` on Windows directly, which uses the correct (Windows-side) files. So even when bash verification is impossible, the code works for him.

### Context window warnings
The conversation hit its context limit once and got summarized; you're now picking up after a second long stretch. Key summary from last compaction is preserved in the jsonl transcript at:
`C:\Users\Umer\AppData\Roaming\Claude\local-agent-mode-sessions\...\c1dc2896-74f8-40c7-9cc1-30a4c5fc14f6.jsonl`

## Open threads / natural next steps

1. **User will ask for more blocks one at a time.** When he does, the pattern is:
   - Add entry to `BLOCK_CATALOG` in `stm32_block_ide.py`
   - Add simulator case in `simulate_model`
   - Add codegen cases in `_emit_decls`, `_emit_step` (and `_emit_init`/`_emit_helpers` if hardware-facing)
   - Add tests to `test_blocks.py`
2. **Ultrasonic timing**: Right now measurement blocks the step loop for up to `timeout_us` (default 30 ms). If he wants lower latency or multiple simultaneous sensors, the clean path is TIM2 input-capture on PA1 (non-blocking echo timing in hardware). He knows this is the logical next hardware-feature upgrade.
3. **Feedback loops / algebraic cycles**: Not supported. `_topo_order` doesn't detect cycles; behavior is undefined if present. If he adds blocks like Integrator/UnitDelay with self-loops, need proper cycle handling.
4. **Only one board supported**: NUCLEO-F446RE. `BOARDS` dict in `code_templates.py` is structured for more, but only the F446RE entry is populated.
5. **Scope stream is wasteful**: streams every step (up to 1 kHz) even when values haven't changed. Fine for now, but if UART bandwidth becomes a constraint this is the knob to turn.

## Tips for working with this user

- He runs Windows + NUCLEO-F446RE hardware. COM3 is his VCP. PuTTY is his go-to terminal.
- He's happy to test things live — offering "restart the IDE and try it" is often the right move.
- When he says "it works" he means it; don't second-guess.
- When something looks like PWM or a sensor or similar, think about whether a dedicated block is cleaner than the obvious Simulink-literal translation. (See the Ultrasonic decision — he was going to ask for a PWM block for HC-SR04 trigger, which would have been wrong.)
- Don't over-apologize when corrections come up. I was wrong about PA1 voltage tolerance earlier — a one-sentence correction plus the actual wiring was the right response, not a long self-flagellation.
- He prefers compact, correct explanations to long ones. Code fixes + a sentence of "here's why" is his speed.

## Quick sanity commands for the next agent

```bash
# Verify all tests pass
cd C:\Users\Umer\Downloads\STM_32_IDE
python test_blocks.py

# Check what changes the IDE would generate (dry run, no flashing)
# Open the IDE, click Build (not Run on Board), inspect:
#   C:\Users\Umer\build_stm32_ide\current_project\main.c
#   C:\Users\Umer\build_stm32_ide\current_project\Makefile

# Smoke-test serial path if scope seems dead:
#   Flash uart_test\main.c (it's a standalone diagnostic), open PuTTY on COM3 @ 115200,
#   confirm "HELLO FROM STM32" stream. If that works, problem is in block codegen.
```

Good luck.
