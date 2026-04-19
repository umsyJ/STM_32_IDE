# Block Reference

Detailed documentation of every block currently shipped with the IDE: what it does, what its inputs and outputs mean, what each parameter controls, and what the generated C code looks like for it. Use this as a companion to the visual editor when you're not sure what a parameter means or why a wire isn't doing what you expected.

A few conventions apply to all blocks. Every block carries its own automatically assigned ID (something like `SquareWave_3`) which is used internally and shown beneath the block's display name. Every parameter is interpreted as a Python expression evaluated against the workspace, so you can write either a literal number (`2.0`) or any expression that resolves to one (`f_sample / 4`). The model runs at the step rate set in the toolbar — every step, all blocks execute once in topological (data-flow) order.

## Square Wave

Generates a square wave at the model step rate. Useful as a clock for blinking, a test stimulus, or any time you need an alternating logic-level signal without setting up a hardware timer.

Inputs: none.
Outputs: `y` — the current sample of the wave, as a float.

Parameters:

The `frequency_hz` parameter controls how many full cycles the wave completes per second. It is interpreted relative to the model's wall-clock time, so a 2 Hz wave running at a 1 ms step rate produces a high-low transition every 250 ms. The `amplitude` parameter is the value the output takes during the high portion of each cycle, and `offset` is the value during the low portion. Defaults are 1.0 and 0.0 respectively, which gives you a clean logic-level signal directly compatible with the GPIO Out block's default 0.5 threshold. The `duty` parameter is the fraction of each cycle spent in the high state — 0.5 produces a symmetric square, 0.1 produces a short pulse with a long gap.

Each step the block updates an internal phase accumulator by `frequency_hz * dt` and wraps it modulo 1.0. The output is `amplitude` while the phase is below `duty`, otherwise `offset`. There is no antialiasing — at frequencies near or above the Nyquist limit of the step rate, the output will alias.

Generated C is a few lines updating a `phase_<id>` static and writing the comparison result to `sig_<id>_y`.

## GPIO In

Reads a physical input pin once per model step and exposes its logic level as a float that downstream blocks can consume.

Inputs: none.
Outputs: `y` — 0.0 or 1.0 depending on the pin's logic level.

Parameters:

The `pin` parameter selects the chip-level pin name in the form `P<port><number>`, for example `PC13` for the user button on the NUCLEO-F446RE. The IDE will validate the name against the chosen board's pin map and emit a `__HAL_RCC_GPIOx_CLK_ENABLE()` for whichever port you reference. The `pull` parameter chooses the internal pull resistor and accepts `none`, `up`, or `down`. For a free-floating sensor or a switch wired to ground you'll typically want `up` so that the pin idles high. The `active_low` parameter inverts the reading when set to 1, which is the convention for buttons that close to ground when pressed. With `active_low = 1`, a pressed button reports as 1.0 on the output, which matches what most users intuitively want.

Generated C calls `HAL_GPIO_ReadPin` and casts the result to float, applying the inversion if requested. There is no debouncing in the generated code — if your application needs it, simulate with the host first using the workspace override `gpioin_<id>` to inject a noisy waveform and confirm downstream blocks behave correctly.

## GPIO Out

Drives a physical output pin once per model step based on its float input.

Inputs: `u` — any float signal.
Outputs: none (this is a sink block).

Parameters:

The `pin` parameter selects the output pin in the same `P<port><number>` notation as GPIO In. On the NUCLEO-F446RE, `PA5` is wired to the green user LED; this is the default and gives you something visible without external wiring. The `threshold` parameter is compared against the input each step — if `u > threshold` the pin is driven high, otherwise low. The default of 0.5 sits exactly between the Square Wave block's 1.0 high and 0.0 low, so the two compose without configuration.

Generated C is a single `HAL_GPIO_WritePin` per step, with the comparison inlined as a ternary. The pin is configured as push-pull output at low speed during init.

If you need a higher-frequency or PWM output, the GPIO Out block isn't the right fit — it can only switch at the model step rate, and stepping faster than ~10 kHz starts to eat significant CPU on the F446RE. A future PWM block will be a thin wrapper around a hardware timer.

## Scope

A passive observer that displays incoming signals and optionally streams them back to the host IDE for live plotting.

Inputs: `u0`, `u1`, `u2` — up to three independent signals.
Outputs: none.

Parameters:

The `max_points` parameter caps how many samples the host-side plot retains, which keeps the rolling display responsive when the scope has been running for a long time. The `stream` parameter is the interesting one. When set to 1, every connected channel is appended to a comma-separated frame that the firmware writes to USART2 each step (which on every Nucleo board is bridged to the ST-Link's USB virtual COM port at 115200 baud). The IDE's Scope / Serial tab listens on that port, parses the frames, and plots the channels live. When set to 0, the Scope block is a no-op on the MCU but the host simulator still uses it to decide what to display when you click Simulate.

Note that streaming has an overhead: each step the firmware does a `snprintf` and a UART transmit. If you crank the model step rate up to 1 ms (the maximum) and stream three channels with %.4f formatting, you're producing about 50 KB/s of UART traffic, which is well within the 11.5 KB/s the link can carry only if you slow the step rate down. As a rule of thumb, keep `step_ms` at 5 or higher when streaming, or only enable streaming when you actively need to look at the data.

## Workspace expressions in parameters

Every parameter field accepts an expression, not just a literal. The expression is evaluated against the Python workspace at code generation time, with `numpy` available as `np` and the `math` module available as `math`. This is the mechanism behind reusable models: define `f_sample = 1000` and `pwm_period = 1 / 50` once in the workspace, then reference those names from a dozen blocks.

A parameter expression is re-evaluated every time you click Build or Run, so changing a workspace variable and re-flashing is enough to reconfigure the firmware — you don't need to touch the diagram. If a parameter references a name that doesn't exist, you'll get a clear error in the Build Log instead of silently inheriting a stale value.

## Things the current set deliberately doesn't include

Several block categories are obvious next steps but aren't shipped yet. PWM output (hardware timer driven), ADC input, I2C/SPI peripherals, sum/gain/integrator math blocks, and a Bus block for grouping signals are all on the natural extension path. The block model in the IDE is open: a new block type is roughly a `BlockSpec` entry in `BLOCK_CATALOG`, a host-side simulation case in `simulate_model`, and a code-generation case in `_emit_step`. The same three-part structure applies whether you're adding a math operator or a peripheral driver.
