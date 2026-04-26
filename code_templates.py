"""
C-code generator for the STM32 Block IDE (v2).

Walks the saved block model and emits a compileable STM32 HAL project
(``main.c`` + Makefile + linker script reference) for the chosen board.

Currently supported boards:
    NUCLEO-F446RE   (STM32F446RE, 180 MHz, USART2 → ST-Link VCP)

Code generation strategy
------------------------
The model executes at a fixed step rate (``step_ms`` from the toolbar) on
SysTick. Every step the generated ``model_step()`` runs all blocks in
data-flow order, propagating signals through internal float variables.

Source blocks (SquareWave, GpioIn) write to an internal signal variable.
Sink blocks (GpioOut, Scope) consume signal variables and produce side
effects (pin writes, serial frames).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Board configuration
# ---------------------------------------------------------------------------

@dataclass
class BoardSpec:
    name: str
    cpu: str
    mcu: str
    flash_kb: int
    sram_kb: int
    flash_origin: str
    flash_size: str
    sram_origin: str
    sram_size: str
    cflags: str
    # mapping pin name -> (port, pin_number)
    pins: Dict[str, Tuple[str, int]]
    # USART used to stream scope data and the pins it uses
    serial_usart: str
    serial_tx: Tuple[str, int]
    serial_rx: Tuple[str, int]


def _f446re() -> BoardSpec:
    pins = {}
    # PA0..PA15, PB0..PB15, PC0..PC15
    for port in "ABCDEFGH":
        for n in range(16):
            pins[f"P{port}{n}"] = (port, n)
    return BoardSpec(
        name="NUCLEO-F446RE",
        cpu="cortex-m4",
        mcu="STM32F446xx",
        flash_kb=512, sram_kb=128,
        flash_origin="0x08000000", flash_size="512K",
        sram_origin="0x20000000", sram_size="128K",
        cflags=("-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard "
                "-DSTM32F446xx -DUSE_HAL_DRIVER"),
        pins=pins,
        serial_usart="USART2",
        serial_tx=("A", 2),
        serial_rx=("A", 3),
    )


BOARDS: Dict[str, BoardSpec] = {
    "NUCLEO-F446RE": _f446re(),
}


# ---------------------------------------------------------------------------
# Topological sort of blocks for execution order
# ---------------------------------------------------------------------------

def _topo_order(model: dict) -> List[dict]:
    blocks = {b["id"]: b for b in model["blocks"]}
    deps: Dict[str, set] = {bid: set() for bid in blocks}
    for c in model["connections"]:
        deps[c["dst_block"]].add(c["src_block"])
    order: List[str] = []
    seen: set = set()

    def visit(bid: str) -> None:
        if bid in seen:
            return
        seen.add(bid)
        for d in deps[bid]:
            visit(d)
        order.append(bid)

    for bid in blocks:
        visit(bid)
    return [blocks[b] for b in order]


def _wires(model: dict) -> Dict[Tuple[str, str], Tuple[str, str]]:
    return {(c["dst_block"], c["dst_port"]): (c["src_block"], c["src_port"])
            for c in model["connections"]}


def _ss_order(b: dict) -> int:
    """Return the number of states for a StateSpace or DiscreteStateSpace block."""
    A_key = "A" if b["type"] == "StateSpace" else "Ad"
    A_str = b["params"].get(A_key, "0")
    rows = [r.strip() for r in A_str.split(";") if r.strip()]
    return max(1, len(rows))


def _bilinear_tf(num_s_str: str, den_s_str: str, fs: float):
    """Convert a continuous-time transfer function to discrete using the
    bilinear (Tustin) transform.

    Parameters
    ----------
    num_s_str, den_s_str : str
        Space-separated polynomial coefficients in the s-domain, highest
        power first.  ``"1 2"`` means s + 2;  ``"1 0 4"`` means s² + 4.
    fs : float
        Sample frequency in Hz (1 / step_s).

    Returns
    -------
    (b_z, a_z) : tuple[ndarray, ndarray]
        Discrete z-domain coefficients, normalized so a_z[0] == 1.
        Length is (order + 1) for both arrays.

    Raises
    ------
    ValueError
        If the transfer function is improper or the strings are malformed.
    """
    import numpy as np

    def _parse(s: str) -> "np.ndarray":
        parts = str(s).split()
        if not parts:
            raise ValueError(f"Empty polynomial string: '{s}'")
        return np.array([float(x) for x in parts], dtype=float)

    num = _parse(num_s_str)
    den = _parse(den_s_str)

    if len(num) > len(den):
        raise ValueError(
            f"Improper transfer function: numerator degree ({len(num)-1}) "
            f"exceeds denominator degree ({len(den)-1}). Must be proper."
        )

    n = len(den) - 1  # system order (= denominator degree)

    # Pad numerator to length n+1 (prepend zeros for missing higher powers)
    num = np.concatenate([np.zeros(n + 1 - len(num)), num])

    # Static gain (order 0): just a constant multiplier, no dynamics.
    if n == 0:
        return np.array([num[0] / den[0]]), np.array([1.0])

    # Bilinear substitution: s → 2*fs*(z−1)/(z+1)
    # Multiplying through by (z+1)^n clears denominators:
    #   coeff[k] * s^(n-k)  →  coeff[k] * (2fs)^(n-k) * (z-1)^(n-k) * (z+1)^k
    twofs = 2.0 * fs
    b_z = np.zeros(n + 1)
    a_z = np.zeros(n + 1)

    for k in range(n + 1):
        power = n - k          # power of s for index k
        poly = np.array([1.0])
        for _ in range(power):                  # (z − 1)^power
            poly = np.convolve(poly, [1.0, -1.0])
        for _ in range(k):                      # (z + 1)^k
            poly = np.convolve(poly, [1.0,  1.0])
        scale = twofs ** power
        b_z += num[k] * scale * poly
        a_z += den[k] * scale * poly

    # Normalize so a_z[0] == 1
    norm = a_z[0]
    return b_z / norm, a_z / norm


# ---------------------------------------------------------------------------
# Per-block C emission
# ---------------------------------------------------------------------------

def _sig_var(bid: str, port: str) -> str:
    return f"sig_{bid}_{port}".replace("-", "_")


def _emit_decls(blocks: List[dict]) -> str:
    decls = []
    for b in blocks:
        if b["type"] == "SquareWave":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
            decls.append(f"static float phase_{b['id']} = 0.0f;")
        elif b["type"] == "Constant":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        elif b["type"] == "GpioIn":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        elif b["type"] == "Ultrasonic":
            decls.append(f"static float {_sig_var(b['id'],'d')} = 0.0f;")
        elif b["type"] in ("Sum", "Product"):
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        elif b["type"] in ("Step", "Integrator", "TransferFcn", "PID"):
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        # Group A sources
        elif b["type"] in ("SineWave", "Ramp", "Clock", "PulseGenerator", "ADC", "TimerTick"):
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        # Group B Math
        elif b["type"] in ("Gain", "Abs", "Sign", "Sqrt", "Saturation", "DeadZone", "MinMax"):
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        # Group C Logic
        elif b["type"] in ("RelationalOperator", "LogicalOperator", "Switch"):
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        # Group D Discrete — output + state
        elif b["type"] == "UnitDelay":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
            decls.append(f"static float _state_{b['id']} = 0.0f;")
        elif b["type"] == "DiscreteIntegrator":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
            decls.append(f"static float _state_{b['id']} = 0.0f;")
        elif b["type"] == "ZeroOrderHold":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
            decls.append(f"static float _state_{b['id']} = 0.0f;")
        elif b["type"] == "Derivative":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
            decls.append(f"static float _state_{b['id']} = 0.0f;")
        # Group E Lookup
        elif b["type"] == "Lookup1D":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        elif b["type"] in ("StateSpace", "DiscreteStateSpace"):
            ns = _ss_order(b)  # helper — see below
            ic_str = b["params"].get("initial_state", "").strip()
            ic_vals = ([float(v) for v in ic_str.split()] if ic_str else [0.0]*ns)
            # Pad or truncate to ns
            ic_vals = (ic_vals + [0.0]*ns)[:ns]
            ic_init = ", ".join(f"{v:.10f}f" for v in ic_vals)
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
            decls.append(f"static float _ss_x_{b['id']}[{ns}] = {{{ic_init}}};")
        elif b["type"] == "ZeroPoleGain":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        # GpioOut, Scope, ToWorkspace, DAC, PWMOut: no signal output
    return "\n".join(decls)


def _emit_helpers(blocks: List[dict]) -> str:
    """Emit helper functions only for block types that need them.

    Currently just the HC-SR04 driver, which uses the DWT cycle counter for
    microsecond-resolution timing of the echo pulse.
    """
    parts: List[str] = []
    if any(b["type"] == "Ultrasonic" for b in blocks):
        parts.append(
            "/* HC-SR04 ultrasonic driver. Uses DWT->CYCCNT for us timing\n"
            "   so the measurement does not depend on SysTick resolution. */\n"
            "static void ultrasonic_init(void) {\n"
            "    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;\n"
            "    DWT->CYCCNT = 0;\n"
            "    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;\n"
            "}\n"
            "\n"
            "static float ultrasonic_measure_m(GPIO_TypeDef *trig_port, uint16_t trig_pin,\n"
            "                                  GPIO_TypeDef *echo_port, uint16_t echo_pin,\n"
            "                                  uint32_t timeout_us) {\n"
            "    uint32_t tpu = SystemCoreClock / 1000000u;       /* cycles per us  */\n"
            "    uint32_t timeout_ticks = timeout_us * tpu;\n"
            "\n"
            "    /* 10 us trigger pulse (HC-SR04 requires >= 10 us). */\n"
            "    HAL_GPIO_WritePin(trig_port, trig_pin, GPIO_PIN_SET);\n"
            "    uint32_t ts = DWT->CYCCNT;\n"
            "    while ((DWT->CYCCNT - ts) < 10u * tpu) { }\n"
            "    HAL_GPIO_WritePin(trig_port, trig_pin, GPIO_PIN_RESET);\n"
            "\n"
            "    /* Wait for echo rising edge (timeout -> 0 m). */\n"
            "    uint32_t t0 = DWT->CYCCNT;\n"
            "    while (HAL_GPIO_ReadPin(echo_port, echo_pin) == GPIO_PIN_RESET) {\n"
            "        if ((DWT->CYCCNT - t0) > timeout_ticks) return 0.0f;\n"
            "    }\n"
            "    uint32_t t_start = DWT->CYCCNT;\n"
            "    /* Wait for echo falling edge (timeout -> 0 m). */\n"
            "    while (HAL_GPIO_ReadPin(echo_port, echo_pin) == GPIO_PIN_SET) {\n"
            "        if ((DWT->CYCCNT - t_start) > timeout_ticks) return 0.0f;\n"
            "    }\n"
            "    uint32_t echo_ticks = DWT->CYCCNT - t_start;\n"
            "    float echo_s = (float)echo_ticks / (float)SystemCoreClock;\n"
            "    /* Round-trip speed of sound ~343 m/s, halved for one-way. */\n"
            "    return echo_s * 171.5f;\n"
            "}\n"
        )
    return "\n".join(parts) if parts else "/* no helpers needed */"


def _emit_init(blocks: List[dict], board: BoardSpec) -> str:
    """GPIO init code based on the blocks present."""
    lines = ["    GPIO_InitTypeDef gi = {0};"]
    used_ports = set()
    for b in blocks:
        if b["type"] in ("GpioIn", "GpioOut"):
            pin = b["params"]["pin"]
            if pin not in board.pins:
                continue
            port, num = board.pins[pin]
            used_ports.add(port)
        elif b["type"] == "Ultrasonic":
            for key in ("trig_pin", "echo_pin"):
                pin = b["params"].get(key)
                if pin in board.pins:
                    used_ports.add(board.pins[pin][0])
    for p in sorted(used_ports):
        lines.append(f"    __HAL_RCC_GPIO{p}_CLK_ENABLE();")

    for b in blocks:
        if b["type"] == "GpioOut":
            pin = b["params"]["pin"]
            if pin not in board.pins:
                continue
            port, num = board.pins[pin]
            lines.append(f"    gi.Pin = GPIO_PIN_{num};")
            lines.append( "    gi.Mode = GPIO_MODE_OUTPUT_PP;")
            lines.append( "    gi.Pull = GPIO_NOPULL;")
            lines.append( "    gi.Speed = GPIO_SPEED_FREQ_LOW;")
            lines.append(f"    HAL_GPIO_Init(GPIO{port}, &gi);")
        elif b["type"] == "GpioIn":
            pin = b["params"]["pin"]
            if pin not in board.pins:
                continue
            port, num = board.pins[pin]
            pull = b["params"].get("pull", "none").lower()
            pull_c = {"none": "GPIO_NOPULL", "up": "GPIO_PULLUP",
                      "down": "GPIO_PULLDOWN"}.get(pull, "GPIO_NOPULL")
            lines.append(f"    gi.Pin = GPIO_PIN_{num};")
            lines.append( "    gi.Mode = GPIO_MODE_INPUT;")
            lines.append(f"    gi.Pull = {pull_c};")
            lines.append( "    gi.Speed = GPIO_SPEED_FREQ_LOW;")
            lines.append(f"    HAL_GPIO_Init(GPIO{port}, &gi);")
        elif b["type"] == "Ultrasonic":
            # TRIG: push-pull output, starts LOW.
            trig = b["params"].get("trig_pin", "PA0")
            if trig in board.pins:
                port, num = board.pins[trig]
                lines.append(f"    /* Ultrasonic {b['id']} TRIG = {trig} */")
                lines.append(f"    HAL_GPIO_WritePin(GPIO{port}, GPIO_PIN_{num}, GPIO_PIN_RESET);")
                lines.append(f"    gi.Pin = GPIO_PIN_{num};")
                lines.append( "    gi.Mode = GPIO_MODE_OUTPUT_PP;")
                lines.append( "    gi.Pull = GPIO_NOPULL;")
                lines.append( "    gi.Speed = GPIO_SPEED_FREQ_HIGH;")
                lines.append(f"    HAL_GPIO_Init(GPIO{port}, &gi);")
            # ECHO: digital input, no pull (sensor drives it actively).
            echo = b["params"].get("echo_pin", "PA1")
            if echo in board.pins:
                port, num = board.pins[echo]
                lines.append(f"    /* Ultrasonic {b['id']} ECHO = {echo} */")
                lines.append(f"    gi.Pin = GPIO_PIN_{num};")
                lines.append( "    gi.Mode = GPIO_MODE_INPUT;")
                lines.append( "    gi.Pull = GPIO_NOPULL;")
                lines.append( "    gi.Speed = GPIO_SPEED_FREQ_HIGH;")
                lines.append(f"    HAL_GPIO_Init(GPIO{port}, &gi);")

    # Enable DWT cycle counter once if any ultrasonic block is present.
    if any(b["type"] == "Ultrasonic" for b in blocks):
        lines.append("    ultrasonic_init();")
    return "\n".join(lines)


def _emit_step(blocks: List[dict], wires, workspace, step_ms: int,
               board: BoardSpec) -> Tuple[str, List[str]]:
    """Emit the model_step() body and the list of streamed scope channels."""
    lines = []
    streamed: List[str] = []   # signal expressions for the scope serial frame
    dt_s = step_ms / 1000.0

    def get_input(bid: str, port: str) -> str:
        src = wires.get((bid, port))
        if src is None:
            return "0.0f"
        return _sig_var(*src)

    def num(name: str, fallback: float = 0.0) -> str:
        try:
            return f"{float(workspace.eval_param(name)):.6f}f"
        except Exception:
            return f"{fallback:.6f}f"

    for b in blocks:
        bid = b["id"]
        t = b["type"]
        p = b["params"]
        if t == "Constant":
            val = num(p["value"], 0.0)
            lines.append(f"    /* block {bid}: Constant */")
            lines.append(f"    {_sig_var(bid,'y')} = {val};")
        elif t == "SquareWave":
            f = num(p["frequency_hz"], 1.0)
            A = num(p["amplitude"], 1.0)
            off = num(p["offset"], 0.0)
            duty = num(p["duty"], 0.5)
            lines.append(f"    /* block {bid}: SquareWave */")
            lines.append(f"    phase_{bid} += {f} * {dt_s:.6f}f;")
            lines.append(f"    if (phase_{bid} >= 1.0f) phase_{bid} -= 1.0f;")
            lines.append(f"    {_sig_var(bid,'y')} = (phase_{bid} < {duty}) ? {A} : {off};")
        elif t == "GpioIn":
            pin = p["pin"]
            port, n = board.pins.get(pin, ("A", 0))
            active_low = int(float(p.get("active_low", 1)))
            level = f"HAL_GPIO_ReadPin(GPIO{port}, GPIO_PIN_{n})"
            if active_low:
                level = f"(!{level})"
            lines.append(f"    /* block {bid}: GpioIn {pin} */")
            lines.append(f"    {_sig_var(bid,'y')} = (float){level};")
        elif t == "GpioOut":
            pin = p["pin"]
            port, n = board.pins.get(pin, ("A", 5))
            thr = num(p["threshold"], 0.5)
            u = get_input(bid, "u")
            lines.append(f"    /* block {bid}: GpioOut {pin} */")
            lines.append(
                f"    HAL_GPIO_WritePin(GPIO{port}, GPIO_PIN_{n}, "
                f"({u} > {thr}) ? GPIO_PIN_SET : GPIO_PIN_RESET);"
            )
        elif t == "Scope":
            stream = int(float(p.get("stream", "1")))
            for port in ("u0", "u1", "u2"):
                if (bid, port) in wires:
                    if stream:
                        streamed.append(get_input(bid, port))
            lines.append(f"    /* block {bid}: Scope ({'stream' if stream else 'no stream'}) */")
        elif t == "Sum":
            u0 = get_input(bid, "u0")   # 0.0f if unconnected
            u1 = get_input(bid, "u1")
            lines.append(f"    /* block {bid}: Sum */")
            lines.append(f"    {_sig_var(bid,'y')} = {u0} + {u1};")
        elif t == "Product":
            # Unconnected inputs must be 1.0f (multiplicative identity).
            u0 = _sig_var(*wires[(bid, "u0")]) if (bid, "u0") in wires else "1.0f"
            u1 = _sig_var(*wires[(bid, "u1")]) if (bid, "u1") in wires else "1.0f"
            lines.append(f"    /* block {bid}: Product */")
            lines.append(f"    {_sig_var(bid,'y')} = {u0} * {u1};")
        elif t == "ToWorkspace":
            var = p.get("variable_name", "yout")
            lines.append(f"    /* block {bid}: ToWorkspace '{var}' — host simulation only, no MCU code */")

        # ── Step ────────────────────────────────────────────────────────────
        elif t == "Step":
            step_time  = num(p.get("step_time",     "1.0"), 1.0)
            init_val   = num(p.get("initial_value",  "0.0"), 0.0)
            final_val  = num(p.get("final_value",    "1.0"), 1.0)
            lines.append(f"    /* block {bid}: Step */")
            lines.append( "    {")
            lines.append(f"        static uint32_t _cnt_{bid} = 0;")
            lines.append(f"        float _t_{bid} = (float)_cnt_{bid} * {dt_s:.6f}f;")
            lines.append(f"        {_sig_var(bid,'y')} = (_t_{bid} >= {step_time}) ? {final_val} : {init_val};")
            lines.append(f"        _cnt_{bid}++;")
            lines.append( "    }")

        # ── Integrator ──────────────────────────────────────────────────────
        elif t == "Integrator":
            u_expr = get_input(bid, "u")
            try:
                ic = float(workspace.eval_param(p.get("initial_value", "0.0")))
            except Exception:
                ic = 0.0
            upper = num(p.get("upper_limit",  "1e10"), 1e10)
            lower = num(p.get("lower_limit", "-1e10"), -1e10)
            lines.append(f"    /* block {bid}: Integrator (IC={ic:.6g}) */")
            lines.append( "    {")
            lines.append(f"        static float _state_{bid} = {ic:.6f}f;")
            lines.append(f"        _state_{bid} += {u_expr} * {dt_s:.6f}f;")
            lines.append(f"        if (_state_{bid} > {upper}) _state_{bid} = {upper};")
            lines.append(f"        if (_state_{bid} < {lower}) _state_{bid} = {lower};")
            lines.append(f"        {_sig_var(bid,'y')} = _state_{bid};")
            lines.append( "    }")

        # ── Transfer Function ────────────────────────────────────────────────
        elif t == "TransferFcn":
            u_expr  = get_input(bid, "u")
            num_str = p.get("numerator",   "1")
            den_str = p.get("denominator", "1 1")
            try:
                # Evaluate any workspace expressions inside the strings
                num_str = str(workspace.eval_param(num_str)) if " " not in num_str else num_str
                den_str = str(workspace.eval_param(den_str)) if " " not in den_str else den_str
            except Exception:
                pass
            try:
                bz, az = _bilinear_tf(num_str, den_str, 1.0 / dt_s)
            except Exception as exc:
                lines.append(f"    /* block {bid}: TransferFcn — codegen error: {exc} */")
                lines.append(f"    {_sig_var(bid,'y')} = 0.0f;")
                continue
            order = len(az) - 1
            lines.append(f"    /* block {bid}: TransferFcn order={order} */")
            if order == 0:
                # Static gain
                lines.append(f"    {_sig_var(bid,'y')} = {bz[0]:.10f}f * {u_expr};")
            else:
                lines.append( "    {")
                # State declarations
                state_decl = ", ".join(f"_s{i}_{bid} = 0.0f" for i in range(order))
                lines.append(f"        static float {state_decl};")
                lines.append(f"        float _u = {u_expr};")
                # Direct Form II Transposed
                lines.append(f"        float _y = {bz[0]:.10f}f * _u"
                              + (f" + _s0_{bid};" if order >= 1 else ";"))
                for i in range(order - 1):
                    rhs = (f"{bz[i+1]:.10f}f * _u - {az[i+1]:.10f}f * _y"
                           + f" + _s{i+1}_{bid}")
                    lines.append(f"        _s{i}_{bid} = {rhs};")
                last = order - 1
                lines.append(f"        _s{last}_{bid} = {bz[order]:.10f}f * _u"
                              f" - {az[order]:.10f}f * _y;")
                lines.append(f"        {_sig_var(bid,'y')} = _y;")
                lines.append( "    }")

        # ── PID Controller ───────────────────────────────────────────────────
        elif t == "PID":
            u_expr = get_input(bid, "u")
            Kp     = num(p.get("Kp",          "1.0"), 1.0)
            Ki     = num(p.get("Ki",          "0.0"), 0.0)
            Kd     = num(p.get("Kd",          "0.0"), 0.0)
            N_filt = num(p.get("N",          "100.0"), 100.0)
            upper  = num(p.get("upper_limit", "1e10"), 1e10)
            lower  = num(p.get("lower_limit","-1e10"), -1e10)
            # Pre-compute constants baked into the firmware
            try:
                _Kd_val = float(workspace.eval_param(p.get("Kd", "0.0")))
                _N_val  = float(workspace.eval_param(p.get("N", "100.0")))
                KdN     = f"{_Kd_val * _N_val:.10f}f"
                N_dt    = f"{min(_N_val * dt_s, 1.99):.10f}f"  # clamp for stability
            except Exception:
                KdN  = f"0.0f"
                N_dt = f"0.0f"
            lines.append(f"    /* block {bid}: PID Kp={Kp} Ki={Ki} Kd={Kd} N={N_filt} */")
            lines.append( "    {")
            lines.append(f"        static float _integ_{bid}  = 0.0f;")
            lines.append(f"        static float _dstate_{bid} = 0.0f;")
            lines.append(f"        float _e = {u_expr};")
            lines.append(f"        float _p = {Kp} * _e;")
            lines.append(f"        _integ_{bid} += {Ki} * _e * {dt_s:.6f}f;")
            lines.append(f"        float _d = {KdN} * (_e - _dstate_{bid});")
            lines.append(f"        _dstate_{bid} += {N_dt} * (_e - _dstate_{bid});")
            lines.append(f"        float _out = _p + _integ_{bid} + _d;")
            lines.append(f"        if (_out > {upper}) _out = {upper};")
            lines.append(f"        if (_out < {lower}) _out = {lower};")
            lines.append(f"        {_sig_var(bid,'y')} = _out;")
            lines.append( "    }")

        elif t == "Ultrasonic":
            trig = p.get("trig_pin", "PA0")
            echo = p.get("echo_pin", "PA1")
            tport, tn = board.pins.get(trig, ("A", 0))
            eport, en = board.pins.get(echo, ("A", 1))
            try:
                period_ms = max(1, int(float(workspace.eval_param(p.get("period_ms", "60")))))
            except Exception:
                period_ms = 60
            try:
                timeout_us = max(1000, int(float(workspace.eval_param(p.get("timeout_us", "30000")))))
            except Exception:
                timeout_us = 30000
            lines.append(f"    /* block {bid}: Ultrasonic HC-SR04 TRIG={trig} ECHO={echo} */")
            lines.append( "    {")
            lines.append(f"        static uint32_t us_cnt_{bid} = 0;")
            lines.append(f"        us_cnt_{bid} += {step_ms};")
            lines.append(f"        if (us_cnt_{bid} >= {period_ms}) {{")
            lines.append(f"            us_cnt_{bid} = 0;")
            lines.append(
                f"            {_sig_var(bid,'d')} = ultrasonic_measure_m("
                f"GPIO{tport}, GPIO_PIN_{tn}, "
                f"GPIO{eport}, GPIO_PIN_{en}, "
                f"{timeout_us}u);"
            )
            lines.append( "        }")
            lines.append( "    }")

        # ---- Group A: Sources ----------------------------------------------

        elif t == "SineWave":
            try:
                freq     = float(workspace.eval_param(p.get("frequency_hz", "1.0")))
                amp      = float(workspace.eval_param(p.get("amplitude",    "1.0")))
                phase_d  = float(workspace.eval_param(p.get("phase_deg",    "0.0")))
                off      = float(workspace.eval_param(p.get("offset",       "0.0")))
            except Exception:
                freq, amp, phase_d, off = 1.0, 1.0, 0.0, 0.0
            import math as _math
            phase_rad = phase_d * _math.pi / 180.0
            lines.append(f"    /* block {bid}: SineWave */")
            lines.append( "    {")
            lines.append(f"        static uint32_t _cnt_{bid} = 0;")
            lines.append(f"        float _t_{bid} = _cnt_{bid} * {dt_s:.6f}f;")
            lines.append(
                f"        {_sig_var(bid,'y')} = {amp:.6f}f * sinf(2.0f * 3.14159265f"
                f" * {freq:.6f}f * _t_{bid} + {phase_rad:.6f}f) + {off:.6f}f;"
            )
            lines.append(f"        _cnt_{bid}++;")
            lines.append( "    }")

        elif t == "Ramp":
            try:
                slope = float(workspace.eval_param(p.get("slope",          "1.0")))
                start = float(workspace.eval_param(p.get("start_time",     "0.0")))
                init  = float(workspace.eval_param(p.get("initial_output", "0.0")))
            except Exception:
                slope, start, init = 1.0, 0.0, 0.0
            lines.append(f"    /* block {bid}: Ramp */")
            lines.append( "    {")
            lines.append(f"        static uint32_t _cnt_{bid} = 0;")
            lines.append(f"        float _t_{bid} = _cnt_{bid} * {dt_s:.6f}f;")
            lines.append(
                f"        {_sig_var(bid,'y')} = (_t_{bid} >= {start:.6f}f)"
                f" ? ({init:.6f}f + {slope:.6f}f * (_t_{bid} - {start:.6f}f))"
                f" : {init:.6f}f;"
            )
            lines.append(f"        _cnt_{bid}++;")
            lines.append( "    }")

        elif t == "Clock":
            lines.append(f"    /* block {bid}: Clock */")
            lines.append( "    {")
            lines.append(f"        static uint32_t _cnt_{bid} = 0;")
            lines.append(f"        {_sig_var(bid,'y')} = _cnt_{bid} * {dt_s:.6f}f;")
            lines.append(f"        _cnt_{bid}++;")
            lines.append( "    }")

        elif t == "PulseGenerator":
            try:
                amp    = float(workspace.eval_param(p.get("amplitude",   "1.0")))
                per    = max(1e-9, float(workspace.eval_param(p.get("period",      "1.0"))))
                pw_pct = float(workspace.eval_param(p.get("pulse_width", "50")))
                delay  = float(workspace.eval_param(p.get("phase_delay", "0.0")))
            except Exception:
                amp, per, pw_pct, delay = 1.0, 1.0, 50.0, 0.0
            pw_frac = pw_pct / 100.0
            lines.append(f"    /* block {bid}: PulseGenerator */")
            lines.append( "    {")
            lines.append(f"        static uint32_t _cnt_{bid} = 0;")
            lines.append(f"        float _t_{bid} = _cnt_{bid} * {dt_s:.6f}f;")
            lines.append(f"        float _rel_{bid} = fmodf(_t_{bid} - {delay:.6f}f, {per:.6f}f);")
            lines.append(
                f"        {_sig_var(bid,'y')} = ((_t_{bid} >= {delay:.6f}f)"
                f" && (_rel_{bid} < {pw_frac:.6f}f * {per:.6f}f)) ? {amp:.6f}f : 0.0f;"
            )
            lines.append(f"        _cnt_{bid}++;")
            lines.append( "    }")

        elif t == "ADC":
            try:
                channel    = int(float(workspace.eval_param(p.get("channel",    "1"))))
                resolution = int(float(workspace.eval_param(p.get("resolution", "12"))))
                vref       = float(workspace.eval_param(p.get("vref",       "3.3")))
            except Exception:
                channel, resolution, vref = 1, 12, 3.3
            maxraw = (1 << resolution) - 1
            lines.append(f"    /* block {bid}: ADC channel {channel} */")
            lines.append( "    {")
            lines.append( "        HAL_ADC_Start(&hadc1);")
            lines.append( "        if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK) {")
            lines.append(
                f"            {_sig_var(bid,'y')} = (HAL_ADC_GetValue(&hadc1)"
                f" / {maxraw}.0f) * {vref:.6f}f;"
            )
            lines.append( "        }")
            lines.append( "        HAL_ADC_Stop(&hadc1);")
            lines.append( "    }")

        elif t == "TimerTick":
            try:
                scale = float(workspace.eval_param(p.get("scale", "0.001")))
            except Exception:
                scale = 0.001
            lines.append(f"    /* block {bid}: TimerTick */")
            lines.append(f"    {_sig_var(bid,'y')} = HAL_GetTick() * {scale:.6f}f;")

        # ---- Group B: Math -------------------------------------------------

        elif t == "Gain":
            try:
                gain = float(workspace.eval_param(p.get("gain", "1.0")))
            except Exception:
                gain = 1.0
            u_expr = get_input(bid, "u")
            lines.append(f"    /* block {bid}: Gain */")
            lines.append(f"    {_sig_var(bid,'y')} = {gain:.6f}f * {u_expr};")

        elif t == "Abs":
            u_expr = get_input(bid, "u")
            lines.append(f"    /* block {bid}: Abs */")
            lines.append(f"    {_sig_var(bid,'y')} = fabsf({u_expr});")

        elif t == "Sign":
            u_expr = get_input(bid, "u")
            lines.append(f"    /* block {bid}: Sign */")
            lines.append(
                f"    {_sig_var(bid,'y')} = ({u_expr} > 0.0f) ? 1.0f"
                f" : ({u_expr} < 0.0f) ? -1.0f : 0.0f;"
            )

        elif t == "Sqrt":
            mode   = p.get("mode", "sqrt").strip().lower()
            u_expr = get_input(bid, "u")
            lines.append(f"    /* block {bid}: Sqrt mode={mode} */")
            if mode == "signed_sqrt":
                lines.append(
                    f"    {_sig_var(bid,'y')} = ({u_expr} >= 0.0f)"
                    f" ? sqrtf({u_expr}) : -sqrtf(-{u_expr});"
                )
            else:
                lines.append(f"    {_sig_var(bid,'y')} = sqrtf(fabsf({u_expr}));")

        elif t == "Saturation":
            upper  = num(p.get("upper_limit",  "1.0"),  1.0)
            lower  = num(p.get("lower_limit", "-1.0"), -1.0)
            u_expr = get_input(bid, "u")
            lines.append(f"    /* block {bid}: Saturation */")
            lines.append( "    {")
            lines.append(f"        float _u_{bid} = {u_expr};")
            lines.append(
                f"        {_sig_var(bid,'y')} = (_u_{bid} > {upper}) ? {upper}"
                f" : (_u_{bid} < {lower}) ? {lower} : _u_{bid};"
            )
            lines.append( "    }")

        elif t == "DeadZone":
            upper  = num(p.get("upper_value",  "0.5"),  0.5)
            lower  = num(p.get("lower_value", "-0.5"), -0.5)
            u_expr = get_input(bid, "u")
            lines.append(f"    /* block {bid}: DeadZone */")
            lines.append( "    {")
            lines.append(f"        float _u_{bid} = {u_expr};")
            lines.append(f"        if      (_u_{bid} > {upper}) {_sig_var(bid,'y')} = _u_{bid} - {upper};")
            lines.append(f"        else if (_u_{bid} < {lower}) {_sig_var(bid,'y')} = _u_{bid} - {lower};")
            lines.append(f"        else                          {_sig_var(bid,'y')} = 0.0f;")
            lines.append( "    }")

        elif t == "MinMax":
            fn     = p.get("function", "min").strip().lower()
            u0     = get_input(bid, "u0")
            u1     = get_input(bid, "u1")
            cfunc  = "fminf" if fn == "min" else "fmaxf"
            lines.append(f"    /* block {bid}: MinMax ({fn}) */")
            lines.append(f"    {_sig_var(bid,'y')} = {cfunc}({u0}, {u1});")

        # ---- Group C: Logic ------------------------------------------------

        elif t == "RelationalOperator":
            op  = p.get("operator", ">").strip()
            u0  = get_input(bid, "u0")
            u1  = get_input(bid, "u1")
            op_map = {
                ">":  f"({u0} > {u1})",
                "<":  f"({u0} < {u1})",
                ">=": f"({u0} >= {u1})",
                "<=": f"({u0} <= {u1})",
                "==": f"({u0} == {u1})",
                "!=": f"({u0} != {u1})",
            }
            cond = op_map.get(op, f"({u0} > {u1})")
            lines.append(f"    /* block {bid}: RelationalOperator ({op}) */")
            lines.append(f"    {_sig_var(bid,'y')} = {cond} ? 1.0f : 0.0f;")

        elif t == "LogicalOperator":
            op = p.get("operator", "AND").strip().upper()
            u0 = get_input(bid, "u0")
            u1 = get_input(bid, "u1")
            lines.append(f"    /* block {bid}: LogicalOperator {op} */")
            if op == "AND":
                expr = f"(({u0} != 0.0f) && ({u1} != 0.0f))"
            elif op == "OR":
                expr = f"(({u0} != 0.0f) || ({u1} != 0.0f))"
            elif op == "NOT":
                expr = f"({u0} == 0.0f)"
            elif op == "NAND":
                expr = f"!(({u0} != 0.0f) && ({u1} != 0.0f))"
            elif op == "NOR":
                expr = f"!(({u0} != 0.0f) || ({u1} != 0.0f))"
            elif op == "XOR":
                expr = f"(({u0} != 0.0f) != ({u1} != 0.0f))"
            else:
                expr = f"(({u0} != 0.0f) && ({u1} != 0.0f))"
            lines.append(f"    {_sig_var(bid,'y')} = {expr} ? 1.0f : 0.0f;")

        elif t == "Switch":
            thr    = num(p.get("threshold", "0.5"), 0.5)
            crit   = p.get("criteria", ">=").strip()
            u0     = get_input(bid, "u0")
            u1     = get_input(bid, "u1")
            u2     = get_input(bid, "u2")
            if crit == ">":
                cond = f"({u1} > {thr})"
            elif crit == "==":
                cond = f"({u1} == {thr})"
            else:
                cond = f"({u1} >= {thr})"
            lines.append(f"    /* block {bid}: Switch criteria={crit} threshold={thr} */")
            lines.append(f"    {_sig_var(bid,'y')} = {cond} ? {u0} : {u2};")

        # ---- Group D: Discrete ---------------------------------------------

        elif t == "UnitDelay":
            u_expr = get_input(bid, "u")
            lines.append(f"    /* block {bid}: UnitDelay */")
            lines.append(f"    {_sig_var(bid,'y')} = _state_{bid};")
            lines.append(f"    _state_{bid} = {u_expr};")

        elif t == "DiscreteIntegrator":
            K      = num(p.get("gain_value",        "1.0"),  1.0)
            upper  = num(p.get("upper_limit",  "1e10"),  1e10)
            lower  = num(p.get("lower_limit", "-1e10"), -1e10)
            u_expr = get_input(bid, "u")
            lines.append(f"    /* block {bid}: DiscreteIntegrator */")
            lines.append( "    {")
            lines.append(f"        float _out_{bid} = _state_{bid};")
            lines.append(f"        if (_out_{bid} > {upper}) _out_{bid} = {upper};")
            lines.append(f"        if (_out_{bid} < {lower}) _out_{bid} = {lower};")
            lines.append(f"        {_sig_var(bid,'y')} = _out_{bid};")
            lines.append(f"        _state_{bid} += {K} * {u_expr} * {dt_s:.6f}f;")
            lines.append(f"        if (_state_{bid} > {upper}) _state_{bid} = {upper};")
            lines.append(f"        if (_state_{bid} < {lower}) _state_{bid} = {lower};")
            lines.append( "    }")

        elif t == "ZeroOrderHold":
            try:
                sample_t = max(dt_s, float(workspace.eval_param(p.get("sample_time", "0.01"))))
            except Exception:
                sample_t = 0.01
            period_ticks = max(1, round(sample_t / dt_s))
            u_expr = get_input(bid, "u")
            lines.append(f"    /* block {bid}: ZeroOrderHold sample_time={sample_t:.4f} */")
            lines.append( "    {")
            lines.append(f"        static uint32_t _hcnt_{bid} = 0;")
            lines.append(f"        static uint32_t _hper_{bid} = {period_ticks}u;")
            lines.append(
                f"        if ((_hper_{bid} == 0) || ((_hcnt_{bid} % _hper_{bid}) == 0))"
                f" _state_{bid} = {u_expr};"
            )
            lines.append(f"        {_sig_var(bid,'y')} = _state_{bid};")
            lines.append(
                f"        if (++_hcnt_{bid} >= _hper_{bid} && _hper_{bid} > 0)"
                f" _hcnt_{bid} = 0;"
            )
            lines.append( "    }")

        elif t == "Derivative":
            u_expr  = get_input(bid, "u")
            inv_dt  = 1.0 / dt_s
            lines.append(f"    /* block {bid}: Derivative */")
            lines.append(f"    {_sig_var(bid,'y')} = ({u_expr} - _state_{bid}) * {inv_dt:.6f}f;")
            lines.append(f"    _state_{bid} = {u_expr};")

        # ---- Group E: Lookup -----------------------------------------------

        elif t == "Lookup1D":
            u_expr  = get_input(bid, "u")
            bp_str  = p.get("breakpoints", "0 1")
            tbl_str = p.get("table_data",  "0 1")
            extrap  = p.get("extrapolation", "clip").strip().lower()
            try:
                bp_vals  = [float(x) for x in bp_str.split()]
                tbl_vals = [float(x) for x in tbl_str.split()]
                sz = len(bp_vals)
                bp_list  = "{" + ", ".join(f"{v:.6f}f" for v in bp_vals)  + "}"
                td_list  = "{" + ", ".join(f"{v:.6f}f" for v in tbl_vals) + "}"
            except Exception:
                bp_vals, tbl_vals, sz = [0.0, 1.0], [0.0, 1.0], 2
                bp_list = "{0.000000f, 1.000000f}"
                td_list = "{0.000000f, 1.000000f}"
            lines.append(f"    /* block {bid}: Lookup1D (extrapolation={extrap}) */")
            lines.append( "    {")
            lines.append(f"        static const float _bp_{bid}[] = {bp_list};")
            lines.append(f"        static const float _td_{bid}[] = {td_list};")
            lines.append(f"        static const int   _sz_{bid}   = {sz};")
            lines.append(f"        float _x_{bid} = {u_expr};")
            lines.append(f"        float _y_{bid};")
            lines.append(f"        if (_x_{bid} <= _bp_{bid}[0]) {{")
            if extrap == "linear" and sz >= 2:
                lines.append(f"            float _sl_{bid} = (_td_{bid}[1] - _td_{bid}[0])"
                              f" / (_bp_{bid}[1] - _bp_{bid}[0]);")
                lines.append(f"            _y_{bid} = _td_{bid}[0] + _sl_{bid}"
                              f" * (_x_{bid} - _bp_{bid}[0]);")
            else:
                lines.append(f"            _y_{bid} = _td_{bid}[0];")
            lines.append(f"        }} else if (_x_{bid} >= _bp_{bid}[_sz_{bid}-1]) {{")
            if extrap == "linear" and sz >= 2:
                lines.append(f"            float _sr_{bid} = (_td_{bid}[_sz_{bid}-1]"
                              f" - _td_{bid}[_sz_{bid}-2])"
                              f" / (_bp_{bid}[_sz_{bid}-1] - _bp_{bid}[_sz_{bid}-2]);")
                lines.append(f"            _y_{bid} = _td_{bid}[_sz_{bid}-1] + _sr_{bid}"
                              f" * (_x_{bid} - _bp_{bid}[_sz_{bid}-1]);")
            else:
                lines.append(f"            _y_{bid} = _td_{bid}[_sz_{bid}-1];")
            lines.append( "        } else {")
            lines.append(f"            int _i_{bid} = 0;")
            lines.append(
                f"            while (_i_{bid} < _sz_{bid}-2"
                f" && _x_{bid} >= _bp_{bid}[_i_{bid}+1]) _i_{bid}++;"
            )
            lines.append(
                f"            float _t_{bid} = (_x_{bid} - _bp_{bid}[_i_{bid}])"
                f" / (_bp_{bid}[_i_{bid}+1] - _bp_{bid}[_i_{bid}]);"
            )
            lines.append(
                f"            _y_{bid} = _td_{bid}[_i_{bid}] + _t_{bid}"
                f" * (_td_{bid}[_i_{bid}+1] - _td_{bid}[_i_{bid}]);"
            )
            lines.append( "        }")
            lines.append(f"        {_sig_var(bid,'y')} = _y_{bid};")
            lines.append( "    }")

        # ── State Space ──────────────────────────────────────────────────────
        elif t == "StateSpace":
            u_expr  = get_input(bid, "u")
            A_str   = p.get("A", "0")
            B_str   = p.get("B", "1")
            C_str   = p.get("C", "1")
            D_val   = float(p.get("D", "0"))
            method  = p.get("method", "euler").strip().lower()
            try:
                import numpy as _np
                def _pm(s):
                    rows = [r.strip() for r in s.split(";") if r.strip()]
                    return _np.array([[float(v) for v in r.split()] for r in rows], dtype=float)
                A_mat = _pm(A_str)
                B_mat = _pm(B_str)
                C_mat = _pm(C_str)
                ns    = A_mat.shape[0]
                B_vec = B_mat.flatten()[:ns]
                C_vec = C_mat.flatten()[:ns]
                # Discretize
                if method == "zoh":
                    try:
                        from scipy.signal import cont2discrete as _c2d
                        _sys = (A_mat, B_vec.reshape(-1,1),
                                C_vec.reshape(1,-1), _np.array([[D_val]]))
                        Ad, Bd_m, _, _, _ = _c2d(_sys, dt_s, method="zoh")
                        Ad = Ad
                        Bd = Bd_m.flatten()
                    except Exception:
                        Ad = _np.eye(ns) + A_mat * dt_s
                        Bd = B_vec * dt_s
                else:
                    Ad = _np.eye(ns) + A_mat * dt_s
                    Bd = B_vec * dt_s
                lines.append(f"    /* block {bid}: StateSpace n={ns} */")
                lines.append(f"    {{")
                lines.append(f"        float _u = {u_expr};")
                lines.append(f"        float _xnew[{ns}];")
                for i in range(ns):
                    row_terms = " + ".join(
                        f"{Ad[i,j]:.10f}f * _ss_x_{bid}[{j}]" for j in range(ns))
                    lines.append(f"        _xnew[{i}] = {Bd[i]:.10f}f * _u + {row_terms};")
                c_terms = " + ".join(f"{C_vec[j]:.10f}f * _ss_x_{bid}[{j}]" for j in range(ns))
                lines.append(f"        {_sig_var(bid,'y')} = {c_terms} + {D_val:.10f}f * _u;")
                lines.append(f"        for (int _i = 0; _i < {ns}; _i++) _ss_x_{bid}[_i] = _xnew[_i];")
                lines.append(f"    }}")
            except Exception as exc:
                lines.append(f"    /* block {bid}: StateSpace — codegen error: {exc} */")
                lines.append(f"    {_sig_var(bid,'y')} = 0.0f;")

        # ── Discrete State Space ──────────────────────────────────────────────
        elif t == "DiscreteStateSpace":
            u_expr = get_input(bid, "u")
            Ad_str = p.get("Ad", "1")
            Bd_str = p.get("Bd", "1")
            Cd_str = p.get("Cd", "1")
            Dd_val = float(p.get("Dd", "0"))
            try:
                import numpy as _np
                def _pm(s):
                    rows = [r.strip() for r in s.split(";") if r.strip()]
                    return _np.array([[float(v) for v in r.split()] for r in rows], dtype=float)
                Ad = _pm(Ad_str)
                Bd_mat = _pm(Bd_str)
                Cd_mat = _pm(Cd_str)
                ns     = Ad.shape[0]
                Bd     = Bd_mat.flatten()[:ns]
                Cd     = Cd_mat.flatten()[:ns]
                lines.append(f"    /* block {bid}: DiscreteStateSpace n={ns} */")
                lines.append(f"    {{")
                lines.append(f"        float _u = {u_expr};")
                lines.append(f"        float _xnew[{ns}];")
                for i in range(ns):
                    row_terms = " + ".join(
                        f"{Ad[i,j]:.10f}f * _ss_x_{bid}[{j}]" for j in range(ns))
                    lines.append(f"        _xnew[{i}] = {Bd[i]:.10f}f * _u + {row_terms};")
                c_terms = " + ".join(f"{Cd[j]:.10f}f * _ss_x_{bid}[{j}]" for j in range(ns))
                lines.append(f"        {_sig_var(bid,'y')} = {c_terms} + {Dd_val:.10f}f * _u;")
                lines.append(f"        for (int _i = 0; _i < {ns}; _i++) _ss_x_{bid}[_i] = _xnew[_i];")
                lines.append(f"    }}")
            except Exception as exc:
                lines.append(f"    /* block {bid}: DiscreteStateSpace — codegen error: {exc} */")
                lines.append(f"    {_sig_var(bid,'y')} = 0.0f;")

        # ── Zero-Pole-Gain ────────────────────────────────────────────────────
        elif t == "ZeroPoleGain":
            u_expr    = get_input(bid, "u")
            zeros_str = p.get("zeros", "").strip()
            poles_str = p.get("poles", "-1").strip()
            gain      = float(p.get("gain", "1.0"))
            try:
                import numpy as _np
                z     = _np.array([float(v) for v in zeros_str.split()]) if zeros_str else _np.array([])
                p_arr = _np.array([float(v) for v in poles_str.split()])
                num_s = " ".join(str(c) for c in _np.atleast_1d(_np.real(_np.poly(z)) * gain))
                den_s = " ".join(str(c) for c in _np.atleast_1d(_np.real(_np.poly(p_arr))))
                bz, az = _bilinear_tf(num_s, den_s, 1.0 / dt_s)
                order  = len(az) - 1
                lines.append(f"    /* block {bid}: ZeroPoleGain order={order} */")
                if order == 0:
                    lines.append(f"    {_sig_var(bid,'y')} = {bz[0]:.10f}f * {u_expr};")
                else:
                    # Direct Form II Transposed
                    sv = f"_zpk_s_{bid}"
                    lines.append(f"    static float {sv}[{order}] = {{0}};")
                    lines.append(f"    {{")
                    lines.append(f"        float _u = {u_expr};")
                    lines.append(f"        float _y = {bz[0]:.10f}f * _u + {sv}[0];")
                    for j in range(order - 1):
                        lines.append(
                            f"        {sv}[{j}] = {bz[j+1]:.10f}f * _u "
                            f"- {az[j+1]:.10f}f * _y + {sv}[{j+1}];")
                    lines.append(
                        f"        {sv}[{order-1}] = {bz[order]:.10f}f * _u "
                        f"- {az[order]:.10f}f * _y;")
                    lines.append(f"        {_sig_var(bid,'y')} = _y;")
                    lines.append(f"    }}")
            except Exception as exc:
                lines.append(f"    /* block {bid}: ZeroPoleGain — codegen error: {exc} */")
                lines.append(f"    {_sig_var(bid,'y')} = 0.0f;")

        # ---- Group F: STM32 HAL --------------------------------------------

        elif t == "DAC":
            try:
                ch   = int(float(workspace.eval_param(p.get("channel", "1"))))
                vref = float(workspace.eval_param(p.get("vref",    "3.3")))
            except Exception:
                ch, vref = 1, 3.3
            u_expr   = get_input(bid, "u")
            ch_macro = "DAC_CHANNEL_1" if ch == 1 else "DAC_CHANNEL_2"
            lines.append(f"    /* block {bid}: DAC channel {ch} */")
            lines.append( "    {")
            lines.append(f"        float _v_{bid} = {u_expr};")
            lines.append(f"        if (_v_{bid} < 0.0f) _v_{bid} = 0.0f;")
            lines.append(f"        if (_v_{bid} > {vref:.6f}f) _v_{bid} = {vref:.6f}f;")
            lines.append(
                f"        uint32_t _raw_{bid} = (uint32_t)((_v_{bid}"
                f" / {vref:.6f}f) * 4095.0f);"
            )
            lines.append(
                f"        HAL_DAC_SetValue(&hdac, {ch_macro},"
                f" DAC_ALIGN_12B_R, _raw_{bid});"
            )
            lines.append(f"        HAL_DAC_Start(&hdac, {ch_macro});")
            lines.append( "    }")

        elif t == "PWMOut":
            try:
                timer    = p.get("timer", "TIM2").strip()
                ch       = int(float(workspace.eval_param(p.get("channel",    "1"))))
                max_duty = float(workspace.eval_param(p.get("max_duty",  "100.0")))
            except Exception:
                timer, ch, max_duty = "TIM2", 1, 100.0
            u_expr      = get_input(bid, "u")
            timer_lower = timer.lower()
            ch_macro    = f"TIM_CHANNEL_{ch}"
            lines.append(f"    /* block {bid}: PWMOut {timer} CH{ch} */")
            lines.append( "    {")
            lines.append(f"        float _duty_{bid} = {u_expr};")
            lines.append(f"        if (_duty_{bid} < 0.0f)          _duty_{bid} = 0.0f;")
            lines.append(f"        if (_duty_{bid} > {max_duty:.6f}f) _duty_{bid} = {max_duty:.6f}f;")
            lines.append(f"        uint32_t _arr_{bid} = __HAL_TIM_GET_AUTORELOAD(&h{timer_lower});")
            lines.append(
                f"        __HAL_TIM_SET_COMPARE(&h{timer_lower}, {ch_macro},"
            )
            lines.append(
                f"            (uint32_t)((_duty_{bid} / {max_duty:.6f}f) * _arr_{bid}));"
            )
            lines.append( "    }")

    return "\n".join(lines), streamed


# ---------------------------------------------------------------------------
# main.c template
# ---------------------------------------------------------------------------

MAIN_C_TEMPLATE = r"""/*
 * Auto-generated by STM32 Block IDE.
 * DO NOT EDIT BY HAND - your changes will be overwritten on the next build.
 *
 * Board:    {board_name}
 * Step:     {step_ms} ms
 * Blocks:   {n_blocks}
 */

#include "stm32f4xx_hal.h"
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------- */
/* Block signals                                                       */
/* ------------------------------------------------------------------- */
{decls}

/* ------------------------------------------------------------------- */
/* Helper functions (emitted only for blocks that need them)           */
/* ------------------------------------------------------------------- */
{helpers}

/* ------------------------------------------------------------------- */
/* Forward decls                                                       */
/* ------------------------------------------------------------------- */
static void SystemClock_Config(void);
static void Error_Handler(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void model_step(void);
static void scope_emit(void);

UART_HandleTypeDef huart2;

volatile uint32_t step_tick_flag = 0;

void SysTick_Handler(void) {{
    HAL_IncTick();
    /* Generated step rate: every {step_ms} ms */
    static uint32_t cnt = 0;
    cnt++;
    if (cnt >= {step_ms}) {{
        cnt = 0;
        step_tick_flag = 1;
    }}
}}

int main(void) {{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART2_UART_Init();

    while (1) {{
        if (step_tick_flag) {{
            step_tick_flag = 0;
            model_step();
            scope_emit();
        }}
    }}
}}

/* ------------------------------------------------------------------- */
static void model_step(void) {{
{step_body}
}}

/* ------------------------------------------------------------------- */
/* Stream scope channels over USART2 as comma-separated floats.        */
/* ------------------------------------------------------------------- */
static void scope_emit(void) {{
{scope_body}
}}

/* ------------------------------------------------------------------- */
/* Board init                                                          */
/* ------------------------------------------------------------------- */
static void MX_GPIO_Init(void) {{
{gpio_init}
}}

static void MX_USART2_UART_Init(void) {{
    __HAL_RCC_USART2_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    GPIO_InitTypeDef gi = {{0}};
    gi.Pin = GPIO_PIN_2 | GPIO_PIN_3;
    gi.Mode = GPIO_MODE_AF_PP;
    gi.Pull = GPIO_NOPULL;
    gi.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    gi.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOA, &gi);

    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart2) != HAL_OK) Error_Handler();
}}

static void SystemClock_Config(void) {{
    RCC_OscInitTypeDef osc = {{0}};
    RCC_ClkInitTypeDef clk = {{0}};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    osc.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    osc.HSIState = RCC_HSI_ON;
    osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    osc.PLL.PLLState = RCC_PLL_ON;
    osc.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    osc.PLL.PLLM = 16; osc.PLL.PLLN = 360;
    osc.PLL.PLLP = RCC_PLLP_DIV2; osc.PLL.PLLQ = 7;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) Error_Handler();
    if (HAL_PWREx_EnableOverDrive() != HAL_OK) Error_Handler();

    clk.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                  | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider = RCC_SYSCLK_DIV1;
    clk.APB1CLKDivider = RCC_HCLK_DIV4;
    clk.APB2CLKDivider = RCC_HCLK_DIV2;
    if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_5) != HAL_OK) Error_Handler();
}}

static void Error_Handler(void) {{
    while (1) {{ /* spin */ }}
}}
"""


MAKEFILE_TEMPLATE = r"""# Auto-generated Makefile for STM32 Block IDE.
# Requires:
#   - arm-none-eabi-gcc (GNU Arm Embedded Toolchain)
#   - STM32CubeF4 firmware package (HAL drivers + CMSIS)
#   - st-flash (from stlink-tools), only for `make flash`
#
# Set CUBE_F4 to the root of your STM32CubeF4 install, e.g.:
#   Windows cmd:  set CUBE_F4=C:/STM32Cube/STM32Cube_FW_F4_V1.28.0
#   bash:         export CUBE_F4=$HOME/STM32Cube/Repository/STM32Cube_FW_F4_V1.28.0

PROJECT  = blockide_app
TARGET   = $(PROJECT).elf
BIN      = $(PROJECT).bin

PREFIX   = arm-none-eabi-
CC       = $(PREFIX)gcc
AS       = $(PREFIX)gcc -x assembler-with-cpp
OBJCOPY  = $(PREFIX)objcopy
SIZE     = $(PREFIX)size

ifeq ($(strip $(CUBE_F4)),)
  $(error CUBE_F4 is not set. Point it at your STM32CubeF4 install, e.g. $$HOME/STM32Cube/Repository/STM32Cube_FW_F4_V1.28.0)
endif

HAL_DIR = $(CUBE_F4)/Drivers/STM32F4xx_HAL_Driver
CMSIS   = $(CUBE_F4)/Drivers/CMSIS

ARCH    = {cflags}
CFLAGS  = $(ARCH) -O2 -Wall -ffunction-sections -fdata-sections \
          -I. \
          -I$(CMSIS)/Include \
          -I$(CMSIS)/Device/ST/STM32F4xx/Include \
          -I$(HAL_DIR)/Inc
ASFLAGS = $(ARCH)
LDSCRIPT = STM32F446RETx_FLASH.ld
LDFLAGS  = $(ARCH) -T$(LDSCRIPT) -Wl,--gc-sections -specs=nano.specs -specs=nosys.specs \
           -u _printf_float

# CubeF4 ships startup_stm32f446xx.s for gcc and system_stm32f4xx.c as a
# template. We reference them directly from the package so the generated
# project needs no extra file copies.
STARTUP = $(CMSIS)/Device/ST/STM32F4xx/Source/Templates/gcc/startup_stm32f446xx.s
SYSTEM  = $(CMSIS)/Device/ST/STM32F4xx/Source/Templates/system_stm32f4xx.c

SRCS_C = main.c \
         $(SYSTEM) \
         $(HAL_DIR)/Src/stm32f4xx_hal.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_cortex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_rcc.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_rcc_ex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_pwr.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_pwr_ex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_gpio.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_dma.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_dma_ex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_flash.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_flash_ex.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_flash_ramfunc.c \
         $(HAL_DIR)/Src/stm32f4xx_hal_uart.c

all: $(BIN)

$(TARGET): $(SRCS_C) $(STARTUP)
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRCS_C) $(STARTUP) -o $@
	$(SIZE) $@

$(BIN): $(TARGET)
	$(OBJCOPY) -O binary $< $@

flash: $(BIN)
	st-flash write $(BIN) 0x8000000

clean:
	rm -f *.elf *.bin *.hex *.o

.PHONY: all flash clean
"""


# Minimal stm32f4xx_hal_conf.h. CubeF4 ships this as a *template* under
# Drivers/STM32F4xx_HAL_Driver/Inc/stm32f4xx_hal_conf_template.h that the
# integrator is supposed to copy into their project. We generate a
# focused version that only enables the modules we actually compile
# (CORTEX, RCC, GPIO, PWR, DMA, UART) — this keeps include order correct
# and avoids unresolved symbols at link time.
HAL_CONF_H = r"""/* Auto-generated stm32f4xx_hal_conf.h for STM32 Block IDE.
   Derived from the CubeF4 template; only modules used by the generated
   code are enabled. */
#ifndef __STM32F4xx_HAL_CONF_H
#define __STM32F4xx_HAL_CONF_H

#ifdef __cplusplus
extern "C" {
#endif

/* ######################## Module Selection ################################ */
#define HAL_MODULE_ENABLED
#define HAL_CORTEX_MODULE_ENABLED
#define HAL_DMA_MODULE_ENABLED
#define HAL_FLASH_MODULE_ENABLED
#define HAL_GPIO_MODULE_ENABLED
#define HAL_PWR_MODULE_ENABLED
#define HAL_RCC_MODULE_ENABLED
#define HAL_UART_MODULE_ENABLED

/* ######################## Oscillator Values adaptation #################### */
#if !defined  (HSE_VALUE)
#define HSE_VALUE              8000000U
#endif
#if !defined  (HSE_STARTUP_TIMEOUT)
#define HSE_STARTUP_TIMEOUT    100U
#endif
#if !defined  (HSI_VALUE)
#define HSI_VALUE              16000000U
#endif
#if !defined  (LSI_VALUE)
#define LSI_VALUE              32000U
#endif
#if !defined  (LSE_VALUE)
#define LSE_VALUE              32768U
#endif
#if !defined  (LSE_STARTUP_TIMEOUT)
#define LSE_STARTUP_TIMEOUT    5000U
#endif
#if !defined  (EXTERNAL_CLOCK_VALUE)
#define EXTERNAL_CLOCK_VALUE   12288000U
#endif

/* ######################## System Configuration ############################ */
#define  VDD_VALUE                    3300U
#define  TICK_INT_PRIORITY            0x0FU
#define  USE_RTOS                     0U
#define  PREFETCH_ENABLE              1U
#define  INSTRUCTION_CACHE_ENABLE     1U
#define  DATA_CACHE_ENABLE            1U

#define  USE_HAL_UART_REGISTER_CALLBACKS   0U
#define  USE_HAL_GPIO_REGISTER_CALLBACKS   0U
#define  USE_HAL_RCC_REGISTER_CALLBACKS    0U
#define  USE_HAL_DMA_REGISTER_CALLBACKS    0U
#define  USE_HAL_PWR_REGISTER_CALLBACKS    0U
#define  USE_HAL_CORTEX_REGISTER_CALLBACKS 0U

#define  USE_SPI_CRC                  0U

/* ######################## Assert Selection ################################ */
/* #define USE_FULL_ASSERT    1U */

/* Includes ---------------------------------------------------------------- */
#ifdef HAL_RCC_MODULE_ENABLED
  #include "stm32f4xx_hal_rcc.h"
#endif
#ifdef HAL_GPIO_MODULE_ENABLED
  #include "stm32f4xx_hal_gpio.h"
#endif
#ifdef HAL_DMA_MODULE_ENABLED
  #include "stm32f4xx_hal_dma.h"
#endif
#ifdef HAL_CORTEX_MODULE_ENABLED
  #include "stm32f4xx_hal_cortex.h"
#endif
#ifdef HAL_FLASH_MODULE_ENABLED
  #include "stm32f4xx_hal_flash.h"
#endif
#ifdef HAL_PWR_MODULE_ENABLED
  #include "stm32f4xx_hal_pwr.h"
#endif
#ifdef HAL_UART_MODULE_ENABLED
  #include "stm32f4xx_hal_uart.h"
#endif

#ifdef USE_FULL_ASSERT
  #define assert_param(expr) ((expr) ? (void)0U : assert_failed((uint8_t *)__FILE__, __LINE__))
  void assert_failed(uint8_t *file, uint32_t line);
#else
  #define assert_param(expr) ((void)0U)
#endif

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_CONF_H */
"""


# ---------------------------------------------------------------------------
# Project generation
# ---------------------------------------------------------------------------

def generate_project(out_dir: Path, model: dict, workspace) -> Path:
    """Write a self-contained project under ``out_dir/<board>_<timestamp>``.

    Returns the project directory path.
    """
    board_name = model.get("board", "NUCLEO-F446RE")
    if board_name not in BOARDS:
        raise ValueError(f"Unsupported board: {board_name}")
    board = BOARDS[board_name]
    step_ms = int(model.get("step_ms", 1))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proj = out_dir / "current_project"
    proj.mkdir(exist_ok=True)

    blocks = _topo_order(model)
    wires = _wires(model)
    decls = _emit_decls(blocks)
    helpers = _emit_helpers(blocks)
    gpio_init = _emit_init(blocks, board)
    step_body, streamed = _emit_step(blocks, wires, workspace, step_ms, board)

    if streamed:
        # printf one line per step. %.4f keeps it readable; semihosting
        # is not used so we go through HAL_UART_Transmit directly.
        fmt = ",".join(["%.4f"] * len(streamed)) + "\\r\\n"
        args = ", ".join(streamed)
        scope_body = (
            "    char buf[96];\n"
            f"    int n = snprintf(buf, sizeof(buf), \"{fmt}\", {args});\n"
            "    if (n > 0) HAL_UART_Transmit(&huart2, (uint8_t*)buf, n, 10);\n"
        )
    else:
        scope_body = "    /* no scope channels connected */\n"

    main_c = MAIN_C_TEMPLATE.format(
        board_name=board_name,
        step_ms=step_ms,
        n_blocks=len(blocks),
        decls=decls,
        helpers=helpers,
        step_body=step_body,
        scope_body=scope_body,
        gpio_init=gpio_init,
    )
    (proj / "main.c").write_text(main_c)
    (proj / "Makefile").write_text(MAKEFILE_TEMPLATE.format(cflags=board.cflags))
    # HAL requires a project-local stm32f4xx_hal_conf.h. Our Makefile puts
    # the project dir first on the include path so this one wins over any
    # stale copy in the CubeF4 tree.
    (proj / "stm32f4xx_hal_conf.h").write_text(HAL_CONF_H)

    # Save model snapshot alongside generated source for traceability.
    import json
    (proj / "model.json").write_text(json.dumps(model, indent=2))

    # Linker script compatible with CubeF4 startup_stm32f446xx.s.
    (proj / "STM32F446RETx_FLASH.ld").write_text(_LD_STUB)

    # README pointing at toolchain prerequisites.
    (proj / "README.txt").write_text(
        "Generated STM32 project.\n\n"
        "Build:  make            (requires arm-none-eabi-gcc + STM32CubeF4)\n"
        "Flash:  make flash      (requires st-flash from stlink-tools)\n\n"
        "Set CUBE_F4 to your STM32Cube_FW_F4 install if it isn't at the\n"
        "default $HOME/STM32Cube/... path.\n"
    )
    return proj


_LD_STUB = """\
/* Linker script for STM32F446RE, compatible with CubeF4's
   startup_stm32f446xx.s (v2). Provides the symbols the startup code
   references: _estack, _sidata, _sdata/_edata, _sbss/_ebss,
   plus the init/fini arrays libc needs. */

ENTRY(Reset_Handler)

_Min_Heap_Size  = 0x200;
_Min_Stack_Size = 0x400;

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   (xrw) : ORIGIN = 0x20000000, LENGTH = 128K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } > FLASH

  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    KEEP (*(.init))
    KEEP (*(.fini))
    . = ALIGN(4);
    _etext = .;
  } > FLASH

  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } > FLASH

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
  .ARM :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } > FLASH

  .preinit_array :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } > FLASH

  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } > FLASH

  .fini_array :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } > FLASH

  _sidata = LOADADDR(.data);

  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM AT> FLASH

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = _ebss;
  } > RAM

  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } > RAM

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
"""
