# Block Reference

Complete documentation for every block in the STM32 Block IDE. Each entry covers what the block does, what its ports mean, what every parameter controls, how it behaves in the host simulator, and what the generated C code looks like.

**General conventions.** Every block has an auto-assigned ID (e.g. `SquareWave_3`) shown beneath its name on the canvas. Every parameter field accepts a Python expression evaluated against the workspace at build time — a literal number (`3.14`), a workspace variable (`f_cutoff`), or any expression (`Vcc / 2`). Unconnected input ports default to a block-specific neutral value (0.0 for additive inputs, 1.0 for multiplicative ones). The model runs at the step rate set in the toolbar; all blocks execute once per step in topological (data-flow) order.

---

## Sources

### Constant

Outputs a fixed float value every model step. The value is resolved once at code-generation time and baked into the firmware as a float literal.

**Outputs:** `y` — the configured constant.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `value` | `1.0` | Any workspace expression resolving to a number. |

**Simulator:** returns the evaluated value unchanged for every sample.

**Generated C:** `sig_<id>_y = <value>f;` — a single assignment, no state.

---

### SquareWave

Generates a square wave using an internal phase accumulator updated each step.

**Outputs:** `y` — float, equals `amplitude` when phase < `duty`, else `offset`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frequency_hz` | `1.0` | Cycles per second (workspace expression). |
| `amplitude` | `1.0` | Output value during the high portion. |
| `offset` | `0.0` | Output value during the low portion. |
| `duty` | `0.5` | Fraction of each cycle spent high (0..1). |

**Simulator:** increments `phase += frequency_hz * dt`, wraps modulo 1.0, outputs `amplitude` or `offset`.

**Generated C:** updates `phase_<id>` then assigns `sig_<id>_y` via a ternary comparison.

---

### SineWave

Generates `A * sin(2π * f * t + φ) + offset`.

**Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frequency_hz` | `1.0` | Frequency in Hz. |
| `amplitude` | `1.0` | Peak amplitude A. |
| `phase_deg` | `0.0` | Phase offset φ in degrees. |
| `offset` | `0.0` | DC offset added to the sine. |

**Generated C:** uses `sinf()` from `<math.h>` with pre-computed angular frequency.

---

### Ramp

Outputs a linear ramp starting at `initial_output` after `start_time`, increasing at `slope` units per second.

**Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slope` | `1.0` | Rate of increase (units/s). |
| `start_time` | `0.0` | Time before which output is `initial_output`. |
| `initial_output` | `0.0` | Value before ramp starts. |

**Generated C:** `y = (t >= start_time) ? initial_output + slope*(t - start_time) : initial_output;`

---

### Clock

Outputs the current simulation time in seconds.

**Outputs:** `y` — elapsed time in seconds.

No parameters.

**Simulator:** `y[k] = k * step_s`.

**Generated C:** divides `HAL_GetTick()` by 1000.0f each step.

---

### PulseGenerator

Generates periodic rectangular pulses.

**Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `amplitude` | `1.0` | Height of each pulse. |
| `period` | `1.0` | Total cycle duration in seconds. |
| `pulse_width` | `50` | High time as a percentage (0–100) of `period`. |
| `phase_delay` | `0.0` | Delay before first pulse starts (seconds). |

**Simulator:** computes phase relative to `phase_delay`, outputs `amplitude` when within the pulse fraction.

---

### Chirp

Linearly swept-frequency sine wave (frequency increases from `f_start` to `f_end` over `sweep_time`).

**Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `amplitude` | `1.0` | Peak amplitude. |
| `f_start` | `1.0` | Starting frequency (Hz). |
| `f_end` | `10.0` | Ending frequency (Hz). |
| `sweep_time` | `5.0` | Duration of one sweep (seconds). |
| `phase_deg` | `0.0` | Initial phase offset (degrees). |

**Simulator:** `φ(t) = 2π(f₀t + (f₁−f₀)t²/(2T))`, output held at final frequency once sweep is complete.

**Generated C:** uses `sinf()` with the instantaneous phase integral.

---

### RandomNumber

Gaussian (normal) random noise with configurable mean and variance.

**Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mean` | `0.0` | Distribution mean µ. |
| `variance` | `1.0` | Distribution variance σ². |
| `seed` | `-1` | RNG seed; −1 = random seed each run. |

**Simulator:** pre-generates the full sample array using `numpy.random.normal` before the simulation loop. Use a fixed seed for repeatable tests.

**Generated C:** calls a lightweight LCG or ARM CMSIS random function scaled to the requested distribution.

---

### FromWorkspace

Reads signal samples from a workspace variable (a 1-D NumPy array). When the simulation runs longer than the array, the last value is held.

**Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `variable_name` | `u` | Name of the workspace variable to read. |
| `default` | `0.0` | Value used if the variable does not exist or is empty. |

**Simulator only** — no MCU code is generated. On hardware, replace with a physical sensor block.

---

### Step

Outputs `initial_value` before `step_time`, then switches permanently to `final_value`.

**Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `step_time` | `1.0` | Switch-over time in seconds. |
| `initial_value` | `0.0` | Output before the step. |
| `final_value` | `1.0` | Output after the step. |

---

### Ground

Always outputs 0.0. Use to explicitly tie unused input ports to a known value without wiring to a Constant block.

**Outputs:** `y` — always 0.0.

No parameters.

---

---

## Hardware Input Blocks

### GpioIn

Reads a digital input pin each step, outputting 0.0 or 1.0.

**Outputs:** `y` — 0.0 (low) or 1.0 (high), after optional inversion.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pin` | `PC13` | Pin name in `P<port><n>` format (e.g. `PC13`). |
| `pull` | `none` | Internal pull resistor: `none`, `up`, or `down`. |
| `active_low` | `1` | If `1`, invert the reading (pressed button → 1.0). |

**Simulator:** defaults to 0.0. Set `WORKSPACE.globals["gpioin_<id>"]` to a scalar or array to inject a custom waveform for testing.

**Generated C:** `HAL_GPIO_ReadPin` with optional inversion; GPIO RCC clock enabled in init.

---

### ADC

Reads one channel of the on-chip ADC and converts to voltage.

**Outputs:** `y` — voltage in volts.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `channel` | `1` | ADC channel number (1–18). |
| `resolution` | `12` | ADC resolution in bits (8/10/12). |
| `vref` | `3.3` | Reference voltage. |
| `sim_value` | `0.0` | Constant voltage returned during simulation. |

**Simulator:** always returns `sim_value`.

**Generated C:** performs a polled single-conversion via `HAL_ADC_Start`/`HAL_ADC_GetValue`, scales to voltage.

---

### EncoderRead

Reads a quadrature encoder connected to a hardware timer in encoder mode.

**Outputs:** `y` — encoder count or velocity.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timer` | `TIM4` | Timer peripheral (TIM1–TIM8). |
| `counts_per_rev` | `1000` | Encoder pulses per revolution (used for velocity mode). |
| `mode` | `position` | `position` = raw count; `velocity` = counts/s. |
| `sim_value` | `0.0` | Value returned during host simulation. |

**Simulator:** always returns `sim_value` (set it to a constant or inject a ramp via workspace override if needed).

---

### I2CRead

Reads 1 or 2 bytes from an I2C device register and scales to a float.

**Outputs:** `y` — scaled reading.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `i2c` | `I2C1` | I2C peripheral handle. |
| `device_addr` | `0x48` | 7-bit I2C address (hex string OK). |
| `reg_addr` | `0x00` | Register address to read. |
| `data_bytes` | `2` | 1 or 2 bytes to read (2 = 16-bit MSB-first). |
| `scale` | `1.0` | Multiplier applied to raw integer reading. |
| `sim_value` | `0.0` | Value returned during host simulation. |

**Simulator:** always returns `sim_value`.

---

### TimerTick

Outputs a scaled version of `HAL_GetTick()` (millisecond counter). Useful for computing elapsed time without using the Clock block.

**Outputs:** `y` — `HAL_GetTick() * scale`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale` | `0.001` | Multiplier (default 0.001 converts ms → s). |

**Simulator:** `y[k] = t_k * 1000 * scale`, exactly matching hardware behaviour.

---

### Ultrasonic (HC-SR04)

Triggers and reads an HC-SR04 ultrasonic range sensor via GPIO bit-bang. Uses the DWT cycle counter for microsecond echo timing.

**Outputs:** `d` — measured distance in metres.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trig_pin` | `PA0` | Trigger output pin. |
| `echo_pin` | `PA1` | Echo input pin. |
| `period_ms` | `60` | Minimum time between measurements (ms). |
| `timeout_us` | `30000` | Echo timeout (µs); returns 0.0 if exceeded. |

**Simulator:** defaults to 0.0 m. Set `WORKSPACE.globals["ultrasonic_<id>"]` to a scalar or array to inject a test distance.

---

---

## Math Blocks

### Sum

Adds two inputs: `y = u0 + u1`. An unconnected port defaults to 0.0.

**Inputs:** `u0`, `u1`  **Outputs:** `y`

No parameters.

**Generated C:** `sig_<id>_y = <u0_expr> + <u1_expr>;` — unconnected ports become `0.0f`.

---

### Product

Multiplies two inputs: `y = u0 * u1`. An unconnected port defaults to 1.0.

**Inputs:** `u0`, `u1`  **Outputs:** `y`

No parameters.

**Generated C:** `sig_<id>_y = <u0_expr> * <u1_expr>;` — unconnected ports become `1.0f`.

**Tip:** Wire a logic-level signal (0/1) to `u1` to gate another signal on and off.

---

### Gain

Multiplies input by a scalar: `y = gain * u`.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gain` | `1.0` | Scalar multiplier (workspace expression). |

---

### Divide

Safe division: `y = u0 / u1`. Returns 0 when `|u1| < eps` to prevent division by zero.

**Inputs:** `u0` (numerator), `u1` (denominator)  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps` | `1e-10` | Near-zero guard threshold. |

---

### Bias

Adds a fixed DC offset: `y = u + bias`.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bias` | `0.0` | Offset value (may be negative). |

---

### Abs

Absolute value: `y = |u|`.

**Inputs:** `u`  **Outputs:** `y`

No parameters.

---

### Sign

Sign function: outputs −1.0, 0.0, or 1.0.

**Inputs:** `u`  **Outputs:** `y`

No parameters.

---

### Sqrt

Square root or signed square root.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `sqrt` | `sqrt`: `y = √max(u,0)`. `signed_sqrt`: `y = sign(u)·√|u|`. |

---

### MathFunction

Applies a standard mathematical function to the input.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `function` | `exp` | `exp`, `log`, `log10`, `square`, `reciprocal`, `pow10`, `pow2`. |
| `exponent` | `2.0` | Reserved for future use. |

Guard behaviour: `log`/`log10` return 0 when `u ≤ 0`; `reciprocal` returns 0 when `|u| < 1e-12`.

---

### RoundingFunction

Rounds the input using a chosen strategy.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `function` | `round` | `floor` (toward −∞), `ceil` (toward +∞), `round` (nearest, ties to even), `fix` (toward 0). |

---

### Polynomial

Evaluates a polynomial using Horner's method (`numpy.polyval`). Coefficients are listed **highest degree first**.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coefficients` | `1 0` | Space-separated, highest degree first. `"1 0"` = u, `"2 0 0"` = 2u², `"1 2 3"` = u²+2u+3. |

**Generated C:** unrolls the Horner evaluation as explicit floating-point multiplications and additions.

---

---

## Nonlinear / Logic Blocks

### Saturation

Clips the input to a fixed range.

**Inputs:** `u`  **Outputs:** `y = clip(u, lower_limit, upper_limit)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `upper_limit` | `1.0` | Maximum output value. |
| `lower_limit` | `-1.0` | Minimum output value. |

---

### SaturationDynamic

Saturation with port-driven limits instead of fixed parameters.

**Inputs:** `u`, `upper`, `lower`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_upper` | `1.0` | Used when `upper` port is unconnected. |
| `default_lower` | `-1.0` | Used when `lower` port is unconnected. |

---

### DeadZone

Outputs zero inside the band `[lower_value, upper_value]`; linear continuation outside it.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lower_value` | `-0.5` | Lower dead-band edge. |
| `upper_value` | `0.5` | Upper dead-band edge. |

---

### MinMax

Selects the smaller or larger of two inputs.

**Inputs:** `u0`, `u1`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `function` | `min` | `min` or `max`. |

---

### RelationalOperator

Compares two inputs: outputs 1.0 when true, 0.0 when false.

**Inputs:** `u0`, `u1`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `operator` | `>` | `>`, `<`, `>=`, `<=`, `==`, `!=`. |

---

### LogicalOperator

Boolean logic on two float inputs (nonzero = true).

**Inputs:** `u0`, `u1`  **Outputs:** `y` (1.0 or 0.0)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `operator` | `AND` | `AND`, `OR`, `NAND`, `NOR`, `XOR`, `NOT` (uses only `u0`). |

---

### Switch

Three-input mux: `y = u0` when `(u1 criteria threshold)`, else `y = u2`.

**Inputs:** `u0` (true branch), `u1` (condition signal), `u2` (false branch)  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.5` | Comparison value for `u1`. |
| `criteria` | `>=` | `>=`, `>`, `==`, `<=`, `<`. |

---

### MultiportSwitch

Selects one of up to 4 data inputs based on an integer selector input.

**Inputs:** `sel`, `u0`–`u3`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inputs` | `4` | Number of active data inputs (1–4). |

`sel` is cast to int (0-based index) and clamped to valid range.

---

### CompareToConstant

Compares input to a fixed constant: outputs 1.0 (true) or 0.0 (false).

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `operator` | `==` | Comparison operator. |
| `constant` | `0.0` | Right-hand side of the comparison. |

---

### DetectRisePositive

Outputs 1.0 for exactly one sample when input transitions from ≤0 to >0 (rising zero-crossing detector).

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_condition` | `0.0` | Assumed u[−1] at t=0. |

---

### Relay

Hysteresis on/off switch. Turns on when `u ≥ on_threshold`, turns off when `u ≤ off_threshold`. State holds between transitions.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `on_threshold` | `0.5` | Input level that energises the relay. |
| `off_threshold` | `-0.5` | Input level that de-energises the relay. |
| `on_value` | `1.0` | Output when relay is on. |
| `off_value` | `0.0` | Output when relay is off. |

---

### RateLimiter

Limits the slew rate of the output in both directions independently.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rising_limit` | `1.0` | Maximum positive rate of change (units/s). |
| `falling_limit` | `-1.0` | Maximum negative rate of change (units/s, must be ≤ 0). |
| `initial_condition` | `0.0` | Output at t=0. |

Per-step logic: `Δ = clamp(u − y_prev, falling_limit·dt, rising_limit·dt)`.

---

### Quantizer

Rounds the input to the nearest multiple of `interval`.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interval` | `0.1` | Quantisation step (must be > 0). |

Uses `round()` (banker's rounding) in the simulator; `roundf()` in generated C.

---

---

## Continuous-Time Control Blocks

### Integrator

Forward-Euler integration with optional output limits.

`y[k] = clip(y[k−1] + u[k−1] · dt, lower, upper)`

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_value` | `0.0` | State at t=0. |
| `upper_limit` | `1e10` | Anti-windup ceiling. |
| `lower_limit` | `-1e10` | Anti-windup floor. |

This is a **delay block**: emits stored state at the start of each step, then updates after all other blocks in the step have run. This makes it safe in closed feedback loops without algebraic loops.

---

### TransferFcn

Continuous-time transfer function H(s) = N(s)/D(s), discretised via bilinear (Tustin) transform at the model step rate.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numerator` | `1` | Coefficients of N(s), highest degree first. |
| `denominator` | `1 1` | Coefficients of D(s). Degree must be ≥ numerator. |

**Generated C:** Direct-Form-II-Transposed IIR with pre-computed bilinear coefficients baked in at build time.

---

### PID

Parallel-form PID with derivative filter:

`u(t) = Kp·e + Ki·∫e·dt + Kd·N·(e − e_filtered)`

**Inputs:** `u` (error)  **Outputs:** `y` (control action)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Kp` | `1.0` | Proportional gain. |
| `Ki` | `0.0` | Integral gain. |
| `Kd` | `0.0` | Derivative gain. |
| `N` | `100.0` | Derivative filter pole (rad/s). Higher = less filtering. |
| `upper_limit` | `1e10` | Output saturation ceiling. |
| `lower_limit` | `-1e10` | Output saturation floor. |

The integral and derivative filter states are delay blocks — correct in feedback loops.

---

### StateSpace (continuous)

Continuous state-space model discretised at the model step rate.

`dx/dt = Ax + Bu,  y = Cx + Du`

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `A` | `0` | System matrix (rows semicolon-separated, e.g. `"0 1; -2 -3"`). |
| `B` | `1` | Input matrix. |
| `C` | `1` | Output matrix. |
| `D` | `0` | Feedthrough matrix. |
| `initial_state` | `` | Space-separated initial state vector (zeros if blank). |
| `method` | `euler` | `euler` (forward Euler) or `zoh` (zero-order-hold exact). |

---

### ZeroPoleGain

Continuous transfer function in zero-pole-gain form: `H(s) = K · ∏(s−zᵢ) / ∏(s−pⱼ)`, discretised via bilinear transform. Only real poles and zeros are currently supported.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `zeros` | `` | Space-separated real zeros (empty = no zeros). |
| `poles` | `-1` | Space-separated real poles. |
| `gain` | `1.0` | DC gain K. |

---

---

## Discrete-Time Control Blocks

### UnitDelay

z⁻¹ — outputs the input value from the previous sample: `y[k] = u[k−1]`.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_condition` | `0.0` | Output at t=0 (i.e. u[−1]). |

**Delay block** — emits stored state first, latches current input after all other blocks complete. Safe in feedback loops.

---

### DiscreteIntegrator

Discrete accumulator with gain and optional output limits.

`y[k] = clip(y[k−1] + K · u[k] · Ts, lower, upper)` (Forward Euler)

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gain_value` | `1.0` | Gain K multiplied before integration. |
| `initial_condition` | `0.0` | Integrator state at t=0. |
| `upper_limit` | `1e10` | Anti-windup ceiling. |
| `lower_limit` | `-1e10` | Anti-windup floor. |
| `method` | `Forward Euler` | `Forward Euler`, `Backward Euler`, or `Trapezoidal`. |

**Delay block** — safe in feedback loops.

---

### DiscreteTransferFcn

Discrete transfer function `H(z) = B(z)/A(z)` in Direct-Form-II-Transposed.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numerator` | `1` | B(z) polynomial, highest power first. |
| `denominator` | `1 -1` | A(z) polynomial, highest power first. A[0] must be 1. |

Example: `denominator = "1 -1"` gives H(z) = 1/(1−z⁻¹), a digital accumulator. The output at k=0 equals the initial condition (zero by default) — it is a **delay block**.

---

### DiscreteStateSpace

Discrete state-space model.

`x[k+1] = Ad·x[k] + Bd·u[k],  y[k] = Cd·x[k] + Dd·u[k]`

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Ad` | `1` | Discrete state matrix. |
| `Bd` | `1` | Discrete input matrix. |
| `Cd` | `1` | Output matrix. |
| `Dd` | `0` | Feedthrough matrix. |
| `initial_state` | `` | Space-separated initial state vector. |

---

### ZeroOrderHold

Samples the input every `sample_time` seconds and holds the value between samples, mimicking hardware ZOH behaviour.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_time` | `0.01` | Hold interval in seconds (must be ≥ `step_ms/1000`). |

---

### Derivative

Numerical first difference: `y[k] = (u[k] − u[k−1]) / dt`.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_condition` | `0.0` | Assumed u[−1] at t=0. |

Noisy signals should be pre-filtered. For a bandwidth-limited derivative, use a TransferFcn approximation `s / (τs + 1)` instead.

---

### TransportDelay

Delays the input by a fixed number of samples using a circular buffer.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delay_samples` | `10` | Integer number of samples of delay (≥ 1). |
| `initial_condition` | `0.0` | Value returned during the initial delay period. |

**Delay block** — the buffer is pre-filled with `initial_condition`. Safe in feedback loops.

---

### MovingAverage

N-sample sliding-window average filter.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | `10` | Number of samples in the average (≥ 1). |

Output ramps up during the first `window` samples while the circular buffer fills. Uses an O(1) running sum accumulator — per-step cost is constant regardless of window size.

---

---

## Lookup Table Blocks

### Lookup1D

Piecewise-linear 1-D lookup table with optional extrapolation beyond the breakpoints.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `breakpoints` | `0 1` | Monotonically increasing input values (space-separated). |
| `table_data` | `0 1` | Output values at each breakpoint. |
| `extrapolation` | `clip` | `clip` (hold endpoint value) or `linear` (extrapolate). |

**Generated C:** binary search + linear interpolation, no `<math.h>` required.

---

### Lookup2D

2-D lookup table with bilinear interpolation.

**Inputs:** `u0` (row axis), `u1` (column axis)  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `row_breakpoints` | `0 1` | Row axis breakpoints (space-separated). |
| `col_breakpoints` | `0 1` | Column axis breakpoints. |
| `table` | `0 1; 1 2` | Table data — rows separated by `;`, columns by spaces. |

Inputs outside the breakpoint range are clamped to the nearest edge before interpolation. The table dimensions must match `len(row_breakpoints) × len(col_breakpoints)`.

---

---

## DSP Blocks

### FIRFilter

Finite Impulse Response (convolution) filter: `y[k] = Σ h[i] · u[k−i]`.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coefficients` | `0.25 0.25 0.25 0.25` | Filter tap weights h[0], h[1], … The tap count is inferred from the number of values. |

Design coefficients in the workspace using `numpy.firwin` and reference them by name. The default is a 4-tap rectangular lowpass (moving average).

**Generated C:** circular buffer with per-step MAC loop; cost is O(taps) per step.

---

### BiquadFilter

Second-order IIR (SOS section) in Direct-Form-II-Transposed:

`y = b0·u + b1·u[k−1] + b2·u[k−2] − a1·y[k−1] − a2·y[k−2]`

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `b0` | `1.0` | Numerator coefficient. |
| `b1` | `0.0` | Numerator coefficient (z⁻¹). |
| `b2` | `0.0` | Numerator coefficient (z⁻²). |
| `a1` | `0.0` | Denominator coefficient (z⁻¹, sign as shown above). |
| `a2` | `0.0` | Denominator coefficient (z⁻²). |

Design using `scipy.signal.iirfilter` or `scipy.signal.butter`, then extract the SOS coefficients.

---

### RunningRMS

Root Mean Square over a sliding window: `y = √(Σ u²[k−i] / N)`.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | `100` | Window size in samples. |

Uses an O(1) running sum-of-squares accumulator — no per-step loop over the window.

---

### MedianFilter

Sliding-window median filter. Excellent at removing impulse (spike) noise while preserving sharp edges in the signal.

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | `5` | Window size, 1–15. Odd values give a clean centre sample. |

**Generated C:** insertion-sort on a small fixed-size buffer; O(window²) per step, practical up to window ≈ 15.

---

### NCO

Numerically Controlled Oscillator — generates sine and cosine at a frequency set by an input port. Suitable for lock-in amplifiers, synchronous demodulation, and software-defined frequency sweeps.

**Inputs:** `freq` — instantaneous frequency (Hz)  **Outputs:** `sin_out`, `cos_out`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `amplitude` | `1.0` | Output amplitude. |
| `initial_phase` | `0.0` | Phase at t=0 (degrees). |

**Generated C:** accumulates phase each step, computes `sinf`/`cosf`.

---

### PeakDetector

Tracks and holds the running peak value with optional exponential decay.

`peak[k] = max(value(u[k]), peak[k−1] − decay_rate · dt)`

**Inputs:** `u`  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `max` | `max` tracks the signed maximum; `abs_max` tracks the absolute maximum. |
| `decay_rate` | `0.0` | Peak decay in units/second. 0 = hold forever. |
| `initial` | `0.0` | Initial peak value. |

---

---

## Custom Code Blocks

### PythonFcn

Evaluates an arbitrary Python expression (or short script) at every simulation step. Simple math expressions are automatically transpiled to C for the generated firmware; complex multi-line code emits a TODO stub that you fill in by hand.

**Inputs:** `u0` … `u3` (only the first `num_inputs` are active)  **Outputs:** `y`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inputs` | `1` | Number of active input ports (1–4). Inputs appear as `u0`…`u3` on the block face. |
| `code` | `u[0]` | Python expression (or multi-line script) evaluated each step. Result = output `y`. |

**Available names inside `code`**

| Name | Type | Description |
|------|------|-------------|
| `u` | `list[float]` | Input values. `u[0]` is `u0`, `u[1]` is `u1`, etc. |
| `t` | `float` | Current simulation time in seconds. |
| `math` | module | Python `math` module — `math.sin`, `math.pi`, etc. |
| `np` / `numpy` | module | NumPy — available during host simulation only. |
| `abs`, `min`, `max`, `round` | builtins | Standard Python builtins. |

**Expression mode (single line)** — the expression value becomes `y`:
```python
math.sin(2 * math.pi * t) * u[0]
u[0]**2 + 2*u[0] + 1
math.sqrt(u[0]**2 + u[1]**2)
```

**Script mode (multi-line)** — assign your result to `y`:
```python
a = u[0] * 2.0
b = u[1] + a
y = b if b > 0.0 else 0.0
```

**Auto-transpilation to C**

Single-line expressions containing only arithmetic operators and `math.*` functions are automatically converted to C when generating firmware:

| Python | C |
|--------|---|
| `u[0]` | upstream signal variable |
| `math.sin(x)` | `sinf(x)` |
| `math.cos(x)` | `cosf(x)` |
| `math.sqrt(x)` | `sqrtf(x)` |
| `math.pi` | `3.14159265f` |
| `x**2` | `powf(x, 2)` |
| `abs(x)` | `fabsf(x)` |

Multi-line code, `return`, `if`/`for`/`while`, and `import` statements cannot be auto-transpiled. The generator emits the Python as a block comment with a `0.0f` stub and a `/* TODO */` marker — replace the stub with your hand-written C before compiling.

**Validation errors**

| Code | Condition |
|------|-----------|
| E001 | `code` is empty |
| E002 | `num_inputs` outside 1–4 |
| E003 | Python syntax error in `code` |

---

---

## Output / Sink Blocks

### GpioOut

Drives a digital output pin based on a threshold comparison each step.

**Inputs:** `u`  **Outputs:** none.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pin` | `PA5` | Output pin (`PA5` = user LED on NUCLEO-F446RE). |
| `threshold` | `0.5` | Sets pin HIGH when `u > threshold`, LOW otherwise. |

**Generated C:** `HAL_GPIO_WritePin(GPIOx, PIN, (u > thr) ? SET : RESET);` — push-pull, low speed.

---

### DAC

Converts a voltage value to an analog output via the on-chip 12-bit DAC.

**Inputs:** `u` — desired output voltage (volts)  **Outputs:** none.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `channel` | `1` | DAC channel (1 or 2). |
| `vref` | `3.3` | Reference voltage used for scaling. |

Input is clamped to `[0, vref]`. Output code = `roundf(u / vref * 4095)`.

---

### PWMOut

Sets the duty cycle of a hardware PWM channel each step.

**Inputs:** `u` — duty value (0 to `max_duty`)  **Outputs:** none.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timer` | `TIM2` | Timer peripheral (TIM1–TIM8). |
| `channel` | `1` | Timer channel (1–4). |
| `frequency_hz` | `1000` | PWM frequency in Hz — configured at init, not per step. |
| `max_duty` | `100.0` | Input value that corresponds to 100% duty cycle. |

**Generated C:** `TIMx->CCRy = (uint32_t)(u / max_duty * (TIMx->ARR + 1));`

---

### UARTSend

Transmits the input value over a chosen UART using printf-style formatting.

**Inputs:** `u`  **Outputs:** none.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `usart` | `USART1` | UART peripheral (USART1–USART6). |
| `format` | `%.4f\r\n` | `printf`-style format string. |
| `timeout` | `10` | HAL transmit timeout (ms). |

USART2 is reserved for Scope streaming on Nucleo boards — use USART1 or USART3–6 here.

---

### I2CWrite

Writes a scaled integer to an I2C device register each step.

**Inputs:** `u`  **Outputs:** none.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `i2c` | `I2C1` | I2C peripheral handle. |
| `device_addr` | `0x48` | 7-bit device address. |
| `reg_addr` | `0x00` | Register to write. |
| `data_bytes` | `2` | 1 or 2 bytes written MSB-first. |
| `scale` | `1.0` | `u` is multiplied by `scale` and rounded to integer before encoding. |

---

### Scope

Passive observer that displays up to 3 signals and optionally streams them to the host IDE over USART2.

**Inputs:** `u0`, `u1`, `u2`  **Outputs:** none.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_points` | `500` | Maximum samples retained in the rolling scope plot. |
| `stream` | `1` | `1` = transmit comma-separated frames over USART2; `0` = display-only. |

Streaming overhead: one `snprintf` + `HAL_UART_Transmit` per step. At 115200 baud keep `step_ms ≥ 5` or limit to ≤ 2 channels when streaming.

---

### ToWorkspace

Captures a signal during host simulation and stores it as a NumPy array in the Python workspace. **Generates no MCU code.**

**Inputs:** `u`  **Outputs:** none.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `variable_name` | `yout` | Name of the workspace array. A `<name>_t` time array is also saved. |
| `max_points` | `10000` | Maximum samples to keep (most recent). |
| `decimation` | `1` | Store every Nth sample (1 = all). |
| `save_time` | `1` | If `1`, also save a companion `<name>_t` time array. |

After simulation, access the data in the Python console: `yout`, `np.fft.fft(yout)`, `plt.plot(yout_t, yout)`, etc.

---

---

## Multi-Rate Scheduling (FreeRTOS only)

Every block has a **Rate (ms)** spinner in the parameter panel (visible when a block is selected). Setting it to `0` (default) assigns the block to the base-rate task. Any non-zero value creates a separate FreeRTOS task running at that period; it must be a positive integer multiple of `step_ms`.

When two or more distinct rates are present and `use_rtos = true`, code generation produces one `model_task_<N>ms` task per rate group:

- Fastest rate gets the highest FreeRTOS priority (base = 5).
- Each additional slower rate gets priority − 1.
- Signal variables shared between tasks are declared `volatile float`.
- Private state variables (`phase_`, `_state_`, etc.) remain non-volatile — each belongs to only one task.
- The base-rate (fastest) task also calls `scope_emit()`.

The host simulator also respects the rate settings: slow-rate blocks execute only on their own period; between executions their last output is held (zero-order hold) so downstream blocks always see a valid value.

Validation (E008): a `sample_time_ms` that is not a positive integer multiple of `step_ms` is flagged as an error before code generation.

Example — SquareWave at 1 ms + DiscreteIntegrator at 10 ms:

```c
/* Signals shared between tasks → volatile */
static volatile float sig_SW_1_y  = 0.0f;
static          float phase_SW_1  = 0.0f;   /* private to 1ms task */
static volatile float sig_DI_1_y  = 0.0f;
static          float _state_DI_1 = 0.0f;   /* private to 10ms task */

xTaskCreate(model_task_1ms,  "Task1ms",  512, NULL, 5, NULL);
xTaskCreate(model_task_10ms, "Task10ms", 512, NULL, 4, NULL);
```

---

## Workspace Expressions

Every parameter accepts a Python expression evaluated against the workspace at build time. `numpy` is available as `np` and `math` as `math`. Define constants once in the workspace (`f_sample = 1000`; `tau = 0.01`) and reference them from any number of blocks. Re-evaluate on demand by clicking Build — no diagram changes needed.

---

## Adding Custom Blocks

A new block type requires three additions:

1. **`BlockSpec` entry** in `BLOCK_CATALOG` (`stm32_block_ide.py`) — defines display name, ports, parameters, colour, and description.
2. **Simulator case** in `simulate_model()` — an `elif btype == "MyBlock":` branch that writes `cur[(bid, "y")]`.
3. **Code-generation case** in `_emit_step()` and `_emit_decls()` (`code_templates.py`) — emits the C variable declaration and the per-step arithmetic.

For stateful blocks (integrators, filters, delays) use the two-phase `pending_updates` pattern: emit the stored state immediately at the start of the step, then append a closure to `pending_updates` that reads the now-resolved input and updates the state. This ensures closed feedback loops work correctly without requiring an iterative solver.

---

## Control / Cart-Pendulum Blocks

These blocks were added specifically for inverted-pendulum / cart-pendulum LQR control on STM32.

---

### WeightedSum

**Category:** Math (also in Control palette)
**Colour:** `#e67e22`

Computes a weighted sum of up to 8 inputs: `y = g₀·u0 + g₁·u1 + … + gₙ₋₁·u(n-1)`.  
Primary use: implement full-state LQR feedback `u = −K·x` in a single block.

| Port | Direction | Description |
|------|-----------|-------------|
| `u0`…`u7` | input | State inputs (first `num_inputs` are active) |
| `y` | output | Weighted sum |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inputs` | `4` | Number of active inputs (1–8) |
| `gains` | `1 1 1 1` | Space-separated gain per input; must have exactly `num_inputs` values |

**Simulation:** `y = Σ gains[i] · u[i]`

**Codegen:** Fully unrolled addition, e.g. `sig_WS_y = 1.0f * sig_A_y + -2.5f * sig_B_y + ...;`

**Validation errors:**
- **E001** — A gain value is not a valid number
- **E002** — Number of gain values ≠ `num_inputs`

**Example — LQR with K = [12, 3.5, 8, 1.2]:**
```
Constant(x1) → u0
Constant(x2) → u1
Constant(x3) → u2
Constant(x4) → u3
WeightedSum(num_inputs=4, gains="-12 -3.5 -8 -1.2") → control output
```

---

### PlantODE

**Category:** Control
**Colour:** `#16a085` (teal)

Simulates a nonlinear ODE plant using 4th-order Runge-Kutta. Designed for in-IDE closed-loop simulation (pendulum, DC motor, two-tank, etc.). Because the real plant is hardware, codegen emits a comment stub — use it for design-time simulation only.

| Port | Direction | Description |
|------|-----------|-------------|
| `u` | input | Control input (shared across all ODE equations as variable `u`) |
| `y0`…`y3` | output | First `num_outputs` states |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `order` | `2` | Number of ODE states (1–8) |
| `equations` | `x[1]`<br>`-9.81*math.sin(x[0]) - 0.1*x[1] + u` | Python expressions, one per line; line `i` = `dx[i]/dt`.<br>Available names: `x` (state vector), `u` (control input), `t` (time), `math.*` |
| `initial_state` | `0 0` | Space-separated initial conditions; must have `order` values |
| `num_outputs` | `2` | How many states to expose as output ports (≤ `order`) |

**Simulation:** RK4 with `dt = step_s`. State vector stored persistently between steps.

**Codegen:** `/* PlantODE: Python simulation only */` comment + `0.0f` stub for each output.

**Validation errors:**
- **E001** — `initial_state` count ≠ `order`
- **E002** — `num_outputs` > `order`
- **E003** — Syntax error in one of the `equations`

**Example — simple pendulum:**
```
equations:
  x[1]
  -9.81*math.sin(x[0]) - 0.5*x[1] + u
initial_state: 0.1 0     # 0.1 rad initial angle, zero velocity
num_outputs:  2          # expose angle (y0) and angular velocity (y1)
```

---

### AngleUnwrap

**Category:** Math (also in Control palette)
**Colour:** `#8e44ad`

Unwraps a wrapped angle signal to produce a continuous output. Prevents ±π (or ±180°) discontinuities that break LQR feedback for pendulum angle measurement.

| Port | Direction | Description |
|------|-----------|-------------|
| `u` | input | Wrapped angle (radians or degrees) |
| `y` | output | Continuous unwrapped angle |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `range` | `auto` | `auto` = ±π radians; `360` = ±180 degrees |

**Simulation state:** `prev` (previous input sample), `offset` (accumulated integer multiple of full range)

**Algorithm:**
```
diff = u − prev
if diff > half: offset -= full
if diff < −half: offset += full
prev = u
y = u + offset
```

**Codegen:** Static variables `_aw_prev_BID` and `_aw_off_BID`; same unwrap logic in C using `fabsf`.

**Validation errors:**
- **E003** — `range` not in `("auto", "360")`

**Example — pendulum encoder output:**
```
EncoderRead → AngleUnwrap(range=auto) → WeightedSum (LQR state 1)
                                      → Scope (continuous angle display)
```

---

### HBridgeOut

**Category:** STM32 HAL
**Colour:** `#c0392b`

Drives an H-bridge motor driver from a signed duty-cycle input.  
Handles direction GPIO, PWM CCR register, and dead-band zeroing. Replaces the manual `PWMOut + GpioOut` combination for bidirectional motor control.

| Port | Direction | Description |
|------|-----------|-------------|
| `u` | input | Signed duty value (e.g. −100 to +100) |
| `pin` | output | Passthrough of `u` for scope monitoring |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timer` | `TIM2` | PWM timer (TIM1–TIM17) |
| `channel` | `1` | Timer channel (1–4); controls `TIMx->CCRy` |
| `dir_pin` | `PB0` | GPIO pin for direction (HIGH = forward) |
| `dead_band_pct` | `5.0` | Dead-band as % of `max_duty`; inputs below this are zeroed |
| `max_duty` | `100.0` | Input magnitude that maps to 100% PWM duty cycle |

**Simulation:** `pin = u` passthrough (no dead-band in simulation for easy scope viewing).

**Codegen:**
```c
float _hb_u = sig_input_y;
float _hb_abs = fabsf(_hb_u);
uint32_t _hb_duty = 0;
if (_hb_abs > 5.000000f) {
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0,
        (_hb_u >= 0.0f) ? GPIO_PIN_SET : GPIO_PIN_RESET);
    _hb_duty = (uint32_t)((_hb_abs / 100.000000f) * (TIM2->ARR + 1));
} else {
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
}
TIM2->CCR1 = _hb_duty;
```

**Validation errors:**
- **E001** — Non-numeric parameter value
- **E002** — `dead_band_pct` < 0 or > 50; `max_duty` ≤ 0; `channel` not 1–4
- **E004** — Invalid `dir_pin` (must match STM32 `Pxnn` format, e.g. `PB0`)

---

### DiscreteIntegratorAW

**Category:** Discrete (also in Control palette)
**Colour:** `#2980b9`

Discrete integrator with back-calculation anti-windup. When the output saturates, the excess state is fed back through a back-calculation gain to prevent the integrator from winding up further. Setting `back_calc_coeff = 0` gives identical behaviour to `DiscreteIntegrator`.

| Port | Direction | Description |
|------|-----------|-------------|
| `u` | input | Input signal to integrate |
| `y` | output | Saturated, anti-windup-corrected integral |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gain_value` | `1.0` | Integrator gain K |
| `initial_condition` | `0.0` | Initial state |
| `upper_limit` | `1e10` | Upper saturation limit |
| `lower_limit` | `-1e10` | Lower saturation limit |
| `method` | `Forward Euler` | Integration method: `Forward Euler`, `Backward Euler`, `Trapezoidal` |
| `back_calc_coeff` | `0.0` | Anti-windup back-calculation coefficient Kaw (0 = disabled, typical = 1/Ki) |

**Simulation (two-phase):**
```
Phase 1: sat_out = clamp(state, lower, upper)
         y = sat_out
Phase 2: back_calc = Kaw * (sat_out − state)
         state += K * u * dt + back_calc * dt
```

**Codegen:**
```c
float _out_BID = _state_BID;
if (_out_BID > upper) _out_BID = upper;
if (_out_BID < lower) _out_BID = lower;
sig_BID_y = _out_BID;
/* When Kaw != 0: */
float _bc_BID = Kaw * (_out_BID - _state_BID);
_state_BID += K * u * dt + _bc_BID * dt;
/* When Kaw == 0: identical to DiscreteIntegrator */
_state_BID += K * u * dt;
```

**Validation errors:**
- **E001** — Non-numeric parameter value (including `back_calc_coeff`)
- **E003** — `method` not one of the three allowed strings
- **E007** — `upper_limit` ≤ `lower_limit`

**Anti-windup tuning:** A typical starting value for `back_calc_coeff` is `1/Ki` (reciprocal of the integral gain). Larger values correct wind-up faster but may cause oscillation.
