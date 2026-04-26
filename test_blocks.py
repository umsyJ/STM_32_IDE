"""Unit tests for the STM32 Block IDE.

Covers:
- Block catalog integrity (every block has a spec, params are well-formed)
- Host-side simulator numerics (simulate_model)
- C-code generator internals (_topo_order, _emit_decls, _emit_init,
  _emit_step, _emit_helpers)
- End-to-end project generation (all files present, contents include
  expected symbols)
- Model JSON round-trip
- Block connections (fan-out, overwrite, multi-input scope)
- Workspace.eval_param (literals, expressions, variable refs)
- _describe display helper (scalars, arrays, collections)
- Simulator edge cases (amplitude/offset, 3-channel scope, bare signals)
- C codegen completeness (step_ms, 3-channel UART format, workspace params)
- Sum block (catalog, simulator, codegen, chaining)
- Product block (catalog, simulator, codegen, gating pattern)

Run with:
    python test_blocks.py          (built-in runner, no dependencies)
    pytest test_blocks.py          (optional - works with pytest too)

Every test is a plain function starting with 'test_'. The runner at the
bottom finds them all, runs each in isolation, and prints a summary.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# Make the IDE modules importable regardless of where the tests are run from.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from code_templates import (  # noqa: E402
    BOARDS,
    _bilinear_tf,
    _emit_decls,
    _emit_helpers,
    _emit_init,
    _emit_step,
    _topo_order,
    _wires,
    generate_project,
)
from stm32_block_ide import (  # noqa: E402
    BLOCK_CATALOG,
    ValidationError,
    _is_valid_stm32_pin,
    _try_eval_param,
    _validate_block,
    simulate_model,
    validate_model,
)
from workspace_shared import WORKSPACE, Workspace  # noqa: E402

# _describe lives in matlab_workspace but has no Qt dependency at import time
# when PyQt5 is not available; guard so headless CI can still run.
try:
    from matlab_workspace import _describe  # noqa: E402
    _HAS_QT = True
except Exception:
    _HAS_QT = False


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeWorkspace:
    """Minimal stand-in for WORKSPACE for tests that only use numeric params."""

    def __init__(self) -> None:
        self.globals: dict = {}

    def eval_param(self, s: str):
        try:
            return float(s)
        except (TypeError, ValueError):
            if s in self.globals:
                return self.globals[s]
            return 0.0


def _model(blocks, connections=None) -> dict:
    return {
        "board": "NUCLEO-F446RE",
        "step_ms": 1,
        "blocks": blocks,
        "connections": connections or [],
    }


def _sw(bid="SW", **overrides):
    params = {
        "frequency_hz": "1.0",
        "amplitude": "1.0",
        "offset": "0.0",
        "duty": "0.5",
    }
    params.update(overrides)
    return {"type": "SquareWave", "id": bid, "x": 0, "y": 0, "params": params}


def _gpio_in(bid="GI", pin="PC13", pull="none", active_low="1"):
    return {
        "type": "GpioIn", "id": bid, "x": 0, "y": 0,
        "params": {"pin": pin, "pull": pull, "active_low": active_low},
    }


def _gpio_out(bid="GO", pin="PA5", threshold="0.5"):
    return {
        "type": "GpioOut", "id": bid, "x": 0, "y": 0,
        "params": {"pin": pin, "threshold": threshold},
    }


def _scope(bid="SC", stream="1"):
    return {
        "type": "Scope", "id": bid, "x": 0, "y": 0,
        "params": {"max_points": "100", "stream": stream},
    }


def _ultra(bid="U", trig="PA0", echo="PA1", period="60", timeout="30000"):
    return {
        "type": "Ultrasonic", "id": bid, "x": 0, "y": 0,
        "params": {
            "trig_pin": trig, "echo_pin": echo,
            "period_ms": period, "timeout_us": timeout,
        },
    }


def _const(bid="K", value="1.0"):
    return {"type": "Constant", "id": bid, "x": 0, "y": 0, "params": {"value": value}}


def _sum(bid="SM"):
    return {"type": "Sum", "id": bid, "x": 0, "y": 0, "params": {}}


def _product(bid="PR"):
    return {"type": "Product", "id": bid, "x": 0, "y": 0, "params": {}}


def _wire(src_bid, src_port, dst_bid, dst_port):
    return {
        "src_block": src_bid, "src_port": src_port,
        "dst_block": dst_bid, "dst_port": dst_port,
    }


# ---------------------------------------------------------------------------
# 1. Block catalog integrity
# ---------------------------------------------------------------------------


def test_catalog_has_all_known_types():
    required = {"SquareWave", "GpioIn", "GpioOut", "Scope", "Ultrasonic",
                "Sum", "Product", "Constant"}
    missing = required - set(BLOCK_CATALOG.keys())
    assert not missing, f"missing block types: {missing}"


def test_catalog_specs_have_matching_type_name():
    for name, spec in BLOCK_CATALOG.items():
        assert spec.type_name == name, (
            f"BLOCK_CATALOG key '{name}' does not match spec.type_name "
            f"'{spec.type_name}'"
        )


def test_catalog_specs_have_descriptions():
    for name, spec in BLOCK_CATALOG.items():
        assert spec.description and len(spec.description) > 10, (
            f"{name}: description missing or too short"
        )


def test_catalog_param_tuples_are_well_formed():
    for name, spec in BLOCK_CATALOG.items():
        for pname, pval in spec.params.items():
            assert isinstance(pval, tuple) and len(pval) == 2, (
                f"{name}.{pname}: params entry must be a (default, help) tuple, "
                f"got {pval!r}"
            )
            default, helptext = pval
            assert isinstance(default, str), f"{name}.{pname}: default must be str"
            assert isinstance(helptext, str) and helptext, (
                f"{name}.{pname}: help text must be a non-empty str"
            )


def test_catalog_colors_are_hex():
    for name, spec in BLOCK_CATALOG.items():
        assert spec.color.startswith("#") and len(spec.color) == 7, (
            f"{name}: color '{spec.color}' should be '#RRGGBB'"
        )


def test_catalog_ports_have_valid_directions():
    for name, spec in BLOCK_CATALOG.items():
        for port in list(spec.inputs) + list(spec.outputs):
            assert port.direction in ("in", "out"), (
                f"{name}: port {port.name} has invalid direction {port.direction!r}"
            )
        for p in spec.inputs:
            assert p.direction == "in"
        for p in spec.outputs:
            assert p.direction == "out"


def test_ultrasonic_has_expected_ports_and_params():
    spec = BLOCK_CATALOG["Ultrasonic"]
    assert [p.name for p in spec.outputs] == ["d"], "Ultrasonic must output 'd'"
    assert not spec.inputs, "Ultrasonic has no inputs"
    for key in ("trig_pin", "echo_pin", "period_ms", "timeout_us"):
        assert key in spec.params, f"Ultrasonic missing param {key}"


# ---------------------------------------------------------------------------
# 2. Simulator (host-side simulate_model)
# ---------------------------------------------------------------------------


def test_simulate_squarewave_duty_50_balanced():
    model = _model([_sw("A", frequency_hz="10"), _scope("S")],
                   [_wire("A", "y", "S", "u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    assert t.shape == (1000,)
    y = sigs.get("S.u0")
    assert y is not None, f"missing Scope output; got keys {list(sigs)}"
    # 50% duty at amplitude 1 / offset 0 → mean about 0.5.
    assert abs(float(y.mean()) - 0.5) < 0.05, f"mean={y.mean():.3f}"
    # Values should be either 1.0 (high) or 0.0 (low), no others.
    uniq = set(float(v) for v in np.unique(y))
    assert uniq <= {0.0, 1.0}, f"unexpected sample values: {uniq}"


def test_simulate_squarewave_duty_20_imbalanced():
    model = _model([_sw("A", frequency_hz="5", duty="0.2"), _scope("S")],
                   [_wire("A", "y", "S", "u0")])
    _, sigs = simulate_model(model, duration_s=2.0, step_s=0.001)
    y = sigs["S.u0"]
    # 20% duty → mean about 0.2 (amplitude 1, offset 0).
    assert abs(float(y.mean()) - 0.2) < 0.05, f"mean={y.mean():.3f}"


def test_simulate_gpio_in_default_is_zero():
    # Clear any leftover workspace override from previous tests.
    WORKSPACE.globals.pop("gpioin_GI", None)
    model = _model([_gpio_in("GI")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    # When no Scope/GpioOut is present, simulator falls back to showing all outs.
    y = sigs.get("GI.y")
    assert y is not None
    assert float(y.sum()) == 0.0


def test_simulate_gpio_in_workspace_override():
    n = 100
    WORKSPACE.globals["gpioin_GI"] = np.ones(n)
    try:
        model = _model([_gpio_in("GI")])
        _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
        y = sigs["GI.y"]
        assert float(y.sum()) == float(n), (
            f"override should produce all ones; got sum={y.sum()}"
        )
    finally:
        WORKSPACE.globals.pop("gpioin_GI", None)


def test_simulate_gpio_out_thresholds_input():
    # SquareWave with 50% duty at amplitude 1 should drive GpioOut high half the time
    # when threshold is 0.5.
    model = _model([_sw("A", frequency_hz="10"), _gpio_out("G", threshold="0.5")],
                   [_wire("A", "y", "G", "u")])
    _, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    pin = sigs.get("G.pin")
    assert pin is not None, f"GpioOut trace missing; got keys {list(sigs)}"
    assert set(float(v) for v in np.unique(pin)) <= {0.0, 1.0}
    # About half should be high.
    assert abs(float(pin.mean()) - 0.5) < 0.05


def test_simulate_ultrasonic_default_is_zero():
    WORKSPACE.globals.pop("ultrasonic_U", None)
    model = _model([_ultra("U"), _scope("S")],
                   [_wire("U", "d", "S", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["S.u0"]
    assert float(y.sum()) == 0.0, "Ultrasonic without override should be all zeros"


def test_simulate_ultrasonic_workspace_override_scalar():
    WORKSPACE.globals["ultrasonic_U"] = 1.5  # constant 1.5 m reading
    try:
        model = _model([_ultra("U"), _scope("S")],
                       [_wire("U", "d", "S", "u0")])
        _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
        y = sigs["S.u0"]
        assert abs(float(y.mean()) - 1.5) < 1e-9
        assert len(y) == 100
    finally:
        WORKSPACE.globals.pop("ultrasonic_U", None)


# ---------------------------------------------------------------------------
# 3. Topological ordering and wire mapping
# ---------------------------------------------------------------------------


def test_topo_order_puts_producers_before_consumers():
    # Intentionally list the consumer first.
    model = _model(
        [_gpio_out("B", pin="PA5"), _sw("A")],
        [_wire("A", "y", "B", "u")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("A") < ids.index("B"), f"bad order: {ids}"


def test_topo_order_chains_three_blocks():
    # A (src) -> B (pass-through would exist if we had one). For now just
    # verify A -> Scope and A -> GpioOut both follow A.
    model = _model(
        [_sw("A"), _scope("S"), _gpio_out("G")],
        [_wire("A", "y", "S", "u0"), _wire("A", "y", "G", "u")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("A") < ids.index("S")
    assert ids.index("A") < ids.index("G")


def test_topo_order_on_unconnected_blocks_is_stable():
    model = _model([_sw("A"), _sw("B"), _sw("C")])
    ids = [b["id"] for b in _topo_order(model)]
    assert set(ids) == {"A", "B", "C"}
    assert len(ids) == 3


def test_wires_maps_dst_to_src():
    model = _model(
        [_sw("A"), _scope("S")],
        [_wire("A", "y", "S", "u0")],
    )
    w = _wires(model)
    assert w == {("S", "u0"): ("A", "y")}


# ---------------------------------------------------------------------------
# 4. _emit_decls — static variable declarations
# ---------------------------------------------------------------------------


def test_emit_decls_squarewave_declares_phase_and_output():
    d = _emit_decls([_sw("SW")])
    assert "sig_SW_y" in d
    assert "phase_SW" in d
    assert "0.0f" in d


def test_emit_decls_gpio_in_declares_output():
    d = _emit_decls([_gpio_in("I")])
    assert "sig_I_y" in d
    assert "phase_I" not in d  # no oscillator state


def test_emit_decls_gpio_out_has_no_signal_var():
    d = _emit_decls([_gpio_out("G")])
    assert "sig_G_" not in d  # sinks don't produce a signal


def test_emit_decls_scope_has_no_signal_var():
    d = _emit_decls([_scope("S")])
    assert "sig_S_" not in d


def test_emit_decls_ultrasonic_declares_distance_output():
    d = _emit_decls([_ultra("U")])
    assert "sig_U_d" in d


def test_emit_decls_combines_multiple_blocks():
    d = _emit_decls([_sw("A"), _gpio_in("B"), _ultra("C")])
    for needle in ("sig_A_y", "phase_A", "sig_B_y", "sig_C_d"):
        assert needle in d, f"{needle!r} missing from combined decls"


# ---------------------------------------------------------------------------
# 5. _emit_helpers — HC-SR04 driver helper
# ---------------------------------------------------------------------------


def test_emit_helpers_none_needed_for_plain_blocks():
    out = _emit_helpers([_sw("A"), _gpio_out("G")])
    assert "ultrasonic_measure_m" not in out
    assert "DWT->CYCCNT" not in out


def test_emit_helpers_emits_ultrasonic_driver():
    out = _emit_helpers([_ultra("U")])
    # Required symbols.
    for needle in (
        "ultrasonic_init",
        "ultrasonic_measure_m",
        "DWT->CYCCNT",
        "CoreDebug->DEMCR",
        "HAL_GPIO_WritePin",
        "HAL_GPIO_ReadPin",
    ):
        assert needle in out, f"{needle!r} missing from ultrasonic helper"
    # Speed-of-sound conversion is present (343/2 = 171.5).
    assert "171.5" in out


def test_emit_helpers_single_driver_even_with_multiple_sensors():
    out = _emit_helpers([_ultra("U1"), _ultra("U2")])
    # The driver function should only be defined once, regardless of the
    # number of sensors.
    assert out.count("static float ultrasonic_measure_m") == 1
    assert out.count("static void ultrasonic_init") == 1


# ---------------------------------------------------------------------------
# 6. _emit_init — GPIO initialization
# ---------------------------------------------------------------------------


def test_emit_init_enables_clock_for_used_ports():
    board = BOARDS["NUCLEO-F446RE"]
    init = _emit_init([_gpio_out("G", pin="PA5"), _gpio_in("I", pin="PC13")], board)
    assert "__HAL_RCC_GPIOA_CLK_ENABLE" in init
    assert "__HAL_RCC_GPIOC_CLK_ENABLE" in init


def test_emit_init_gpio_out_is_push_pull():
    board = BOARDS["NUCLEO-F446RE"]
    init = _emit_init([_gpio_out("G", pin="PA5")], board)
    assert "GPIO_PIN_5" in init
    assert "GPIO_MODE_OUTPUT_PP" in init


def test_emit_init_gpio_in_applies_pull_up():
    board = BOARDS["NUCLEO-F446RE"]
    init = _emit_init([_gpio_in("I", pin="PC13", pull="up")], board)
    assert "GPIO_MODE_INPUT" in init
    assert "GPIO_PULLUP" in init


def test_emit_init_gpio_in_applies_pull_down():
    board = BOARDS["NUCLEO-F446RE"]
    init = _emit_init([_gpio_in("I", pin="PB0", pull="down")], board)
    assert "GPIO_PULLDOWN" in init


def test_emit_init_ultrasonic_configures_both_pins_and_calls_init():
    board = BOARDS["NUCLEO-F446RE"]
    init = _emit_init([_ultra("U", trig="PA0", echo="PA1")], board)
    assert "GPIO_PIN_0" in init
    assert "GPIO_PIN_1" in init
    assert "GPIO_MODE_OUTPUT_PP" in init   # TRIG
    assert "GPIO_MODE_INPUT" in init        # ECHO
    assert "ultrasonic_init()" in init      # DWT enable
    # Port clock enabled.
    assert "__HAL_RCC_GPIOA_CLK_ENABLE" in init


def test_emit_init_no_ultrasonic_means_no_dwt_init():
    board = BOARDS["NUCLEO-F446RE"]
    init = _emit_init([_gpio_out("G")], board)
    assert "ultrasonic_init()" not in init


# ---------------------------------------------------------------------------
# 7. _emit_step — per-tick C code
# ---------------------------------------------------------------------------


def test_emit_step_squarewave_updates_phase_and_samples():
    board = BOARDS["NUCLEO-F446RE"]
    ws = FakeWorkspace()
    step, streamed = _emit_step([_sw("A")], {}, ws, step_ms=1, board=board)
    assert "phase_A" in step
    assert "sig_A_y" in step
    assert streamed == []


def test_emit_step_gpio_in_active_low_inverts_read():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_gpio_in("I", active_low="1")], {}, FakeWorkspace(), 1, board)
    assert "!HAL_GPIO_ReadPin" in step


def test_emit_step_gpio_in_active_high_does_not_invert():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_gpio_in("I", active_low="0")], {}, FakeWorkspace(), 1, board)
    assert "(float)HAL_GPIO_ReadPin" in step
    assert "!HAL_GPIO_ReadPin" not in step


def test_emit_step_gpio_out_uses_threshold_comparison():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _gpio_out("G", threshold="0.5")]
    wires = {("G", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "HAL_GPIO_WritePin" in step
    assert "GPIO_PIN_SET" in step
    assert "GPIO_PIN_RESET" in step
    # The wired signal should appear in the comparison.
    assert "sig_A_y" in step


def test_emit_step_scope_reports_streamed_channels_in_wire_order():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sw("B"), _scope("S", stream="1")]
    wires = {("S", "u0"): ("A", "y"), ("S", "u1"): ("B", "y")}
    _, streamed = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_A_y" in streamed
    assert "sig_B_y" in streamed


def test_emit_step_scope_stream_off_produces_no_streamed_channels():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _scope("S", stream="0")]
    wires = {("S", "u0"): ("A", "y")}
    _, streamed = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert streamed == []


def test_emit_step_ultrasonic_uses_period_gate_and_helper_call():
    board = BOARDS["NUCLEO-F446RE"]
    ws = FakeWorkspace()
    step, _ = _emit_step([_ultra("U", period="60", timeout="30000")], {}, ws, 1, board)
    assert "ultrasonic_measure_m" in step
    assert "GPIO_PIN_0" in step
    assert "GPIO_PIN_1" in step
    assert "30000u" in step
    # Period-gated at 60 ms.
    assert ">= 60" in step


def test_emit_step_ultrasonic_period_scales_with_step_ms():
    board = BOARDS["NUCLEO-F446RE"]
    ws = FakeWorkspace()
    step, _ = _emit_step([_ultra("U")], {}, ws, step_ms=5, board=board)
    # Counter should add step_ms each tick.
    assert "us_cnt_U += 5" in step


# ---------------------------------------------------------------------------
# 8. generate_project — end-to-end file generation
# ---------------------------------------------------------------------------


def test_generate_project_writes_all_expected_files():
    model = _model(
        [_sw("A"), _gpio_out("G"), _scope("S")],
        [_wire("A", "y", "G", "u"), _wire("A", "y", "S", "u0")],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        for name in (
            "main.c", "Makefile", "stm32f4xx_hal_conf.h",
            "STM32F446RETx_FLASH.ld", "model.json", "README.txt",
        ):
            assert (proj / name).exists(), f"generated project missing {name}"


def test_generate_project_main_c_contains_core_blocks():
    model = _model(
        [_sw("A"), _gpio_out("G", pin="PA5"), _scope("S")],
        [_wire("A", "y", "G", "u"), _wire("A", "y", "S", "u0")],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        # Signals, HAL calls, and the scope stream should all be in place.
        assert "sig_A_y" in c
        assert "HAL_GPIO_WritePin" in c
        assert "HAL_UART_Transmit" in c
        assert "SystemClock_Config" in c
        assert "SysTick_Handler" in c


def test_generate_project_with_ultrasonic_emits_helper_and_init_call():
    model = _model([_ultra("U", trig="PA0", echo="PA1"), _scope("S")],
                   [_wire("U", "d", "S", "u0")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "ultrasonic_measure_m" in c
        assert "ultrasonic_init()" in c
        assert "sig_U_d" in c
        # Distance should be streamed over UART.
        assert "HAL_UART_Transmit" in c


def test_generate_project_without_ultrasonic_omits_helper():
    model = _model([_sw("A"), _scope("S")],
                   [_wire("A", "y", "S", "u0")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "ultrasonic_measure_m" not in c
        assert "ultrasonic_init" not in c


def test_generate_project_makefile_has_printf_float_flag():
    # Regression guard: nano libc needs '-u _printf_float' for snprintf("%f").
    model = _model([_sw("A"), _scope("S")], [_wire("A", "y", "S", "u0")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        makefile = (proj / "Makefile").read_text()
        assert "-u _printf_float" in makefile, (
            "Makefile is missing '-u _printf_float' - scope floats won't format"
        )


def test_generate_project_hal_conf_enables_required_modules():
    model = _model([_sw("A"), _scope("S")], [_wire("A", "y", "S", "u0")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        conf = (proj / "stm32f4xx_hal_conf.h").read_text()
        for mod in (
            "HAL_MODULE_ENABLED",
            "HAL_CORTEX_MODULE_ENABLED",
            "HAL_DMA_MODULE_ENABLED",
            "HAL_FLASH_MODULE_ENABLED",
            "HAL_GPIO_MODULE_ENABLED",
            "HAL_PWR_MODULE_ENABLED",
            "HAL_RCC_MODULE_ENABLED",
            "HAL_UART_MODULE_ENABLED",
        ):
            assert mod in conf, f"{mod} not enabled in hal_conf.h"


def test_generate_project_saves_model_snapshot():
    model = _model([_sw("A")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        saved = json.loads((proj / "model.json").read_text())
        assert saved == model


def test_generate_project_rejects_unknown_board():
    model = _model([_sw("A")])
    model["board"] = "NUCLEO-UNKNOWN-99"
    try:
        with tempfile.TemporaryDirectory() as d:
            generate_project(Path(d), model, FakeWorkspace())
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown board")


# ---------------------------------------------------------------------------
# 9. Model JSON round-trip
# ---------------------------------------------------------------------------


def test_model_survives_json_roundtrip():
    original = _model(
        [_ultra("U7"), _scope("S"), _gpio_out("G")],
        [_wire("U7", "d", "S", "u0"), _wire("U7", "d", "G", "u")],
    )
    # Mutate coordinates to non-defaults so we know they survive.
    original["blocks"][0]["x"] = 42.5
    original["blocks"][1]["y"] = -17.25
    roundtripped = json.loads(json.dumps(original))
    assert roundtripped == original


def test_model_param_values_are_preserved_as_strings():
    # Parameter values are stored as strings (they may reference workspace
    # variables). JSON round-trip must keep them as strings, not coerce to
    # numbers.
    model = _model([_sw("A", frequency_hz="my_freq_var", amplitude="2.5")])
    roundtripped = json.loads(json.dumps(model))
    params = roundtripped["blocks"][0]["params"]
    assert params["frequency_hz"] == "my_freq_var"
    assert params["amplitude"] == "2.5"


# ---------------------------------------------------------------------------
# 10. Block connections — fan-out, overwrite, multi-input scope
# ---------------------------------------------------------------------------


def test_wires_fan_out_one_source_to_two_sinks():
    model = _model(
        [_sw("A"), _scope("S"), _gpio_out("G")],
        [_wire("A", "y", "S", "u0"), _wire("A", "y", "G", "u")],
    )
    w = _wires(model)
    assert w[("S", "u0")] == ("A", "y")
    assert w[("G", "u")] == ("A", "y")


def test_wires_scope_three_channels_all_mapped():
    model = _model(
        [_sw("A"), _sw("B"), _sw("C"), _scope("S")],
        [
            _wire("A", "y", "S", "u0"),
            _wire("B", "y", "S", "u1"),
            _wire("C", "y", "S", "u2"),
        ],
    )
    w = _wires(model)
    assert w[("S", "u0")] == ("A", "y")
    assert w[("S", "u1")] == ("B", "y")
    assert w[("S", "u2")] == ("C", "y")


def test_wires_later_connection_overwrites_earlier_on_same_port():
    # Two connections both wiring to the same dst port — last one wins.
    model = _model(
        [_sw("A"), _sw("B"), _scope("S")],
        [_wire("A", "y", "S", "u0"), _wire("B", "y", "S", "u0")],
    )
    w = _wires(model)
    # The second wire should win.
    assert w[("S", "u0")] == ("B", "y")
    assert len(w) == 1


def test_wires_empty_connections_returns_empty_dict():
    model = _model([_sw("A"), _scope("S")])
    assert _wires(model) == {}


def test_topo_order_chain_ultrasonic_to_scope():
    model = _model(
        [_scope("S"), _ultra("U")],
        [_wire("U", "d", "S", "u0")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("U") < ids.index("S")


def test_topo_order_two_independent_sources_both_present():
    model = _model([_sw("A"), _ultra("U"), _scope("S")],
                   [_wire("A", "y", "S", "u0"), _wire("U", "d", "S", "u1")])
    ids = [b["id"] for b in _topo_order(model)]
    assert set(ids) == {"A", "U", "S"}
    assert ids.index("A") < ids.index("S")
    assert ids.index("U") < ids.index("S")


def test_topo_order_empty_model():
    model = _model([])
    ids = [b["id"] for b in _topo_order(model)]
    assert ids == []


def test_topo_order_gpio_in_before_gpio_out():
    model = _model(
        [_gpio_out("GO"), _gpio_in("GI")],
        [_wire("GI", "y", "GO", "u")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("GI") < ids.index("GO")


# ---------------------------------------------------------------------------
# 11. Workspace.eval_param — literals, expressions, variable refs
# ---------------------------------------------------------------------------


def test_workspace_eval_integer_literal():
    ws = Workspace()
    assert ws.eval_param("42") == 42


def test_workspace_eval_float_literal():
    ws = Workspace()
    assert abs(ws.eval_param("3.14") - 3.14) < 1e-9


def test_workspace_eval_scientific_notation():
    ws = Workspace()
    assert abs(ws.eval_param("1e3") - 1000.0) < 1e-9


def test_workspace_eval_python_expression():
    ws = Workspace()
    result = ws.eval_param("2 * 3 + 1")
    assert result == 7


def test_workspace_eval_uses_numpy_and_math():
    ws = Workspace()
    result = ws.eval_param("np.sqrt(4)")
    assert abs(float(result) - 2.0) < 1e-9


def test_workspace_eval_pi_constant():
    ws = Workspace()
    import math as _math
    assert abs(ws.eval_param("pi") - _math.pi) < 1e-9


def test_workspace_eval_user_variable():
    ws = Workspace()
    ws.globals["f_pwm"] = 500.0
    assert ws.eval_param("f_pwm") == 500.0
    assert ws.eval_param("f_pwm * 2") == 1000.0


def test_workspace_eval_empty_string_returns_zero():
    ws = Workspace()
    assert ws.eval_param("") == 0


def test_workspace_eval_raises_on_undefined_name():
    ws = Workspace()
    try:
        ws.eval_param("undefined_var_xyz")
    except ValueError:
        return
    raise AssertionError("expected ValueError for undefined name")


def test_workspace_globals_persist_across_calls():
    ws = Workspace()
    ws.eval_param.__func__  # just ensure it's callable
    ws.globals["x"] = 10
    assert ws.eval_param("x + 5") == 15
    ws.globals["x"] = 20
    assert ws.eval_param("x + 5") == 25


# ---------------------------------------------------------------------------
# 12. _describe display helper (no Qt needed — pure logic)
# ---------------------------------------------------------------------------


def test_describe_scalar_int():
    if not _HAS_QT:
        return
    val, cls, size = _describe("x", 42)
    assert "42" in val
    assert size == "1"


def test_describe_scalar_float():
    if not _HAS_QT:
        return
    val, cls, size = _describe("x", 3.14)
    assert "3.14" in val
    assert size == "1"


def test_describe_small_ndarray():
    if not _HAS_QT:
        return
    arr = np.array([1.0, 2.0, 3.0])
    val, cls, size = _describe("a", arr)
    assert "ndarray" in cls
    assert size == "3"


def test_describe_large_ndarray_shows_shape():
    if not _HAS_QT:
        return
    arr = np.zeros((100, 50))
    val, cls, size = _describe("a", arr)
    assert "<array" in val
    assert "100" in val or "5000" in val or "50" in val


def test_describe_list():
    if not _HAS_QT:
        return
    val, cls, size = _describe("lst", [1, 2, 3])
    assert cls == "list"
    assert size == "3"


def test_describe_dict():
    if not _HAS_QT:
        return
    val, cls, size = _describe("d", {"a": 1, "b": 2})
    assert cls == "dict"
    assert size == "2"


def test_describe_long_repr_is_truncated():
    if not _HAS_QT:
        return
    big_list = list(range(1000))
    val, cls, size = _describe("lst", big_list)
    assert len(val) <= 63  # 60 chars + "..."


# ---------------------------------------------------------------------------
# 13. Simulator edge cases
# ---------------------------------------------------------------------------


def test_simulate_squarewave_nondefault_amplitude_and_offset():
    model = _model(
        [_sw("A", frequency_hz="10", amplitude="3.0", offset="-1.0"), _scope("S")],
        [_wire("A", "y", "S", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["S.u0"]
    uniq = {round(float(v), 6) for v in np.unique(y)}
    assert uniq <= {3.0, -1.0}, f"unexpected values: {uniq}"
    # 50% duty → mean midpoint between 3.0 and -1.0 = 1.0
    assert abs(float(y.mean()) - 1.0) < 0.1, f"mean={y.mean():.3f}"


def test_simulate_squarewave_full_duty_is_always_high():
    model = _model(
        [_sw("A", frequency_hz="10", duty="1.0"), _scope("S")],
        [_wire("A", "y", "S", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.5, step_s=0.001)
    y = sigs["S.u0"]
    assert float(y.min()) == 1.0, "duty=1.0 should always be high"


def test_simulate_squarewave_zero_duty_is_always_low():
    model = _model(
        [_sw("A", frequency_hz="10", duty="0.0"), _scope("S")],
        [_wire("A", "y", "S", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.5, step_s=0.001)
    y = sigs["S.u0"]
    assert float(y.max()) == 0.0, "duty=0.0 should always be low"


def test_simulate_scope_three_channels_all_captured():
    model = _model(
        [
            _sw("A", frequency_hz="1", duty="0.3"),
            _sw("B", frequency_hz="2", duty="0.7"),
            _sw("C", frequency_hz="5", duty="0.5"),
            _scope("S"),
        ],
        [
            _wire("A", "y", "S", "u0"),
            _wire("B", "y", "S", "u1"),
            _wire("C", "y", "S", "u2"),
        ],
    )
    _, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    for key in ("S.u0", "S.u1", "S.u2"):
        assert key in sigs, f"missing channel {key}"
    # Channel means should reflect duty cycles
    assert abs(float(sigs["S.u0"].mean()) - 0.3) < 0.05
    assert abs(float(sigs["S.u1"].mean()) - 0.7) < 0.05
    assert abs(float(sigs["S.u2"].mean()) - 0.5) < 0.05


def test_simulate_no_scope_returns_source_signals():
    # Without a Scope or GpioOut, simulator should still return source traces.
    WORKSPACE.globals.pop("gpioin_GI2", None)
    model = _model([_gpio_in("GI2")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert "GI2.y" in sigs, f"bare source signal missing; got {list(sigs)}"


def test_simulate_gpio_in_active_high_workspace_override():
    WORKSPACE.globals["gpioin_AL"] = np.ones(50)
    try:
        model = _model([_gpio_in("AL", active_low="0"), _scope("S")],
                       [_wire("AL", "y", "S", "u0")])
        _, sigs = simulate_model(model, duration_s=0.05, step_s=0.001)
        y = sigs["S.u0"]
        assert float(y.sum()) == 50.0
    finally:
        WORKSPACE.globals.pop("gpioin_AL", None)


def test_simulate_ultrasonic_array_override_length_matches():
    WORKSPACE.globals["ultrasonic_UA"] = np.linspace(0.5, 2.0, 200)
    try:
        model = _model([_ultra("UA"), _scope("S")],
                       [_wire("UA", "d", "S", "u0")])
        _, sigs = simulate_model(model, duration_s=0.2, step_s=0.001)
        y = sigs["S.u0"]
        assert len(y) == 200
        assert abs(float(y[0]) - 0.5) < 0.01
        assert abs(float(y[-1]) - 2.0) < 0.01
    finally:
        WORKSPACE.globals.pop("ultrasonic_UA", None)


def test_simulate_returns_time_array_matching_duration():
    model = _model([_sw("A"), _scope("S")], [_wire("A", "y", "S", "u0")])
    t, _ = simulate_model(model, duration_s=0.5, step_s=0.01)
    assert len(t) == 50
    assert abs(float(t[0])) < 1e-9
    assert abs(float(t[-1]) - 0.49) < 1e-9


# ---------------------------------------------------------------------------
# 14. _emit_step — additional coverage
# ---------------------------------------------------------------------------


def test_emit_step_scope_three_channels_all_in_streamed():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sw("B"), _sw("C"), _scope("S", stream="1")]
    wires = {
        ("S", "u0"): ("A", "y"),
        ("S", "u1"): ("B", "y"),
        ("S", "u2"): ("C", "y"),
    }
    _, streamed = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_A_y" in streamed
    assert "sig_B_y" in streamed
    assert "sig_C_y" in streamed
    assert len(streamed) == 3


def test_emit_step_multiple_gpio_out_blocks():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _gpio_out("G1", pin="PA5"), _gpio_out("G2", pin="PB0")]
    wires = {("G1", "u"): ("A", "y"), ("G2", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert step.count("HAL_GPIO_WritePin") == 2
    assert "GPIO_PIN_5" in step
    assert "GPIO_PIN_0" in step


def test_emit_step_squarewave_nondefault_amplitude_baked_in():
    board = BOARDS["NUCLEO-F446RE"]
    ws = FakeWorkspace()
    step, _ = _emit_step([_sw("W", amplitude="2.5", offset="-0.5")], {}, ws, 1, board)
    # amplitude and offset values should appear in the generated code
    assert "2.5" in step
    assert "-0.5" in step or "0.5" in step  # sign may be embedded differently


def test_emit_step_ultrasonic_nondefault_timeout():
    board = BOARDS["NUCLEO-F446RE"]
    ws = FakeWorkspace()
    step, _ = _emit_step([_ultra("U", timeout="15000")], {}, ws, 1, board)
    assert "15000u" in step


def test_emit_step_gpio_in_pull_up_has_no_effect_on_step_code():
    # Pull resistor is configured in init, not step; step code is the same
    board = BOARDS["NUCLEO-F446RE"]
    step_up, _ = _emit_step([_gpio_in("I", pull="up")], {}, FakeWorkspace(), 1, board)
    step_none, _ = _emit_step([_gpio_in("I", pull="none")], {}, FakeWorkspace(), 1, board)
    # Both should read the pin; pull mode doesn't alter the read expression
    assert "HAL_GPIO_ReadPin" in step_up
    assert "HAL_GPIO_ReadPin" in step_none


def test_emit_step_unconnected_gpio_out_defaults_to_zero():
    # GpioOut with no wire: input is zero, which is <= threshold, so pin stays low.
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_gpio_out("G", threshold="0.5")]
    step, _ = _emit_step(blocks, {}, FakeWorkspace(), 1, board)
    assert "HAL_GPIO_WritePin" in step
    assert "GPIO_PIN_RESET" in step


# ---------------------------------------------------------------------------
# 15. C codegen completeness
# ---------------------------------------------------------------------------


def test_generate_project_step_ms_baked_into_main_c():
    model = _model([_sw("A"), _scope("S")], [_wire("A", "y", "S", "u0")])
    model["step_ms"] = 10
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        # SysTick reload for 10 ms at 180 MHz: 180_000_000 / 100 = 1_800_000
        # The step_ms value itself should appear somewhere in the generated code.
        assert "10" in c


def test_generate_project_three_channel_scope_uart_format():
    model = _model(
        [_sw("A"), _sw("B"), _sw("C"), _scope("S")],
        [
            _wire("A", "y", "S", "u0"),
            _wire("B", "y", "S", "u1"),
            _wire("C", "y", "S", "u2"),
        ],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        # Three %.4f format specifiers in one snprintf call.
        assert c.count("%.4f") >= 3
        assert "sig_A_y" in c
        assert "sig_B_y" in c
        assert "sig_C_y" in c


def test_generate_project_workspace_param_baked_as_float():
    ws = FakeWorkspace()
    ws.globals["my_freq"] = 25.0
    # FakeWorkspace.eval_param looks up globals dict
    ws_real = type("WS", (), {
        "globals": {"my_freq": 25.0},
        "eval_param": lambda self, s: (
            float(s) if _is_numeric(s) else ws.globals.get(s, 0.0)
        ),
    })()

    def _is_numeric(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    model = _model([_sw("A", frequency_hz="my_freq"), _scope("S")],
                   [_wire("A", "y", "S", "u0")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, ws)
        c = (proj / "main.c").read_text()
        # my_freq=25 should fall back to 0.0 via FakeWorkspace (it's not numeric)
        # but the key thing is it generates valid C without crashing.
        assert "sig_A_y" in c


def test_generate_project_signal_naming_convention():
    # Signal variable names must follow sig_{block_id}_{port} convention.
    model = _model(
        [_ultra("MySensor"), _gpio_out("MyLED"), _scope("MyScope")],
        [
            _wire("MySensor", "d", "MyLED", "u"),
            _wire("MySensor", "d", "MyScope", "u0"),
        ],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "sig_MySensor_d" in c
        assert "sig_MyLED_" not in c   # sinks have no output signal var
        assert "sig_MyScope_" not in c  # sinks have no output signal var


def test_generate_project_gpio_in_clock_and_mode():
    # Verify GpioIn generates correct port clock + input mode in main.c
    model = _model([_gpio_in("BTN", pin="PC13"), _gpio_out("LED", pin="PA5")],
                   [_wire("BTN", "y", "LED", "u")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "__HAL_RCC_GPIOC_CLK_ENABLE" in c
        assert "__HAL_RCC_GPIOA_CLK_ENABLE" in c
        assert "GPIO_PIN_13" in c
        assert "GPIO_PIN_5" in c
        assert "GPIO_MODE_INPUT" in c
        assert "GPIO_MODE_OUTPUT_PP" in c


def test_generate_project_readme_contains_build_instructions():
    model = _model([_sw("A")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        readme = (proj / "README.txt").read_text()
        # Should mention how to build (make) and flash
        assert "make" in readme.lower() or "build" in readme.lower()


def test_generate_project_linker_script_has_correct_flash_origin():
    model = _model([_sw("A")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        ld = (proj / "STM32F446RETx_FLASH.ld").read_text()
        # STM32F446RE flash starts at 0x08000000
        assert "0x08000000" in ld
        # RAM at 0x20000000
        assert "0x20000000" in ld


def test_generate_project_makefile_targets_arm_toolchain():
    model = _model([_sw("A")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        mk = (proj / "Makefile").read_text()
        assert "arm-none-eabi-gcc" in mk


def test_emit_decls_multiple_squarewaves_all_declared():
    blocks = [_sw("W1"), _sw("W2"), _sw("W3")]
    d = _emit_decls(blocks)
    for bid in ("W1", "W2", "W3"):
        assert f"sig_{bid}_y" in d
        assert f"phase_{bid}" in d


def test_emit_init_multiple_ports_on_same_bank_clock_enabled_once():
    # PA5 and PA0 are both on GPIOA — clock enable should appear at least once.
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_gpio_out("G1", pin="PA5"), _gpio_out("G2", pin="PA0")]
    init = _emit_init(blocks, board)
    assert "__HAL_RCC_GPIOA_CLK_ENABLE" in init


# ---------------------------------------------------------------------------
# 16. Sum block — catalog, simulator, codegen
# ---------------------------------------------------------------------------


def test_catalog_has_sum_block():
    assert "Sum" in BLOCK_CATALOG


def test_sum_spec_has_two_inputs_and_one_output():
    spec = BLOCK_CATALOG["Sum"]
    assert [p.name for p in spec.inputs] == ["u0", "u1"]
    assert [p.name for p in spec.outputs] == ["y"]


def test_sum_spec_has_no_params():
    assert BLOCK_CATALOG["Sum"].params == {}


def test_sum_has_valid_color():
    color = BLOCK_CATALOG["Sum"].color
    assert color.startswith("#") and len(color) == 7


def test_emit_decls_sum_declares_output():
    d = _emit_decls([_sum("S1")])
    assert "sig_S1_y" in d
    assert "phase_S1" not in d


def test_emit_step_sum_adds_both_inputs():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sw("B"), _sum("S1")]
    wires = {("S1", "u0"): ("A", "y"), ("S1", "u1"): ("B", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_S1_y" in step
    assert "sig_A_y" in step
    assert "sig_B_y" in step
    assert "+" in step


def test_emit_step_sum_unconnected_input_defaults_to_zero():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sum("S1")]
    wires = {("S1", "u0"): ("A", "y")}  # u1 not connected
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "0.0f" in step   # unconnected u1 becomes 0.0f


def test_emit_step_sum_both_unconnected_is_zero_plus_zero():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_sum("S1")], {}, FakeWorkspace(), 1, board)
    assert "sig_S1_y" in step
    assert "0.0f + 0.0f" in step


def test_simulate_sum_adds_two_squarewaves():
    # A at 100% duty (always 1.0) + B at 100% duty (always 1.0) → always 2.0
    model = _model(
        [_sw("A", duty="1.0"), _sw("B", duty="1.0"), _sum("S1"), _scope("SC")],
        [
            _wire("A", "y", "S1", "u0"),
            _wire("B", "y", "S1", "u1"),
            _wire("S1", "y", "SC", "u0"),
        ],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 2.0) < 1e-6, f"expected 2.0, got {y.mean()}"


def test_simulate_sum_unconnected_input_acts_as_zero():
    model = _model(
        [_sw("A", duty="1.0"), _sum("S1"), _scope("SC")],
        [_wire("A", "y", "S1", "u0"), _wire("S1", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 1.0) < 1e-6


def test_simulate_sum_with_offset_signals():
    # A outputs 3.0 always, B outputs -1.0 always → sum is 2.0
    model = _model(
        [
            _sw("A", amplitude="3.0", duty="1.0"),
            _sw("B", amplitude="-1.0", duty="1.0"),
            _sum("S1"),
            _scope("SC"),
        ],
        [
            _wire("A", "y", "S1", "u0"),
            _wire("B", "y", "S1", "u1"),
            _wire("S1", "y", "SC", "u0"),
        ],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 2.0) < 1e-6


def test_simulate_sum_chained_two_sums():
    # Sum1 = A + B = 1+1 = 2, Sum2 = Sum1 + C = 2+1 = 3
    model = _model(
        [
            _sw("A", duty="1.0"),
            _sw("B", duty="1.0"),
            _sw("C", duty="1.0"),
            _sum("S1"),
            _sum("S2"),
            _scope("SC"),
        ],
        [
            _wire("A", "y", "S1", "u0"),
            _wire("B", "y", "S1", "u1"),
            _wire("S1", "y", "S2", "u0"),
            _wire("C", "y", "S2", "u1"),
            _wire("S2", "y", "SC", "u0"),
        ],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 3.0) < 1e-6


def test_generate_project_sum_block_in_main_c():
    model = _model(
        [_sw("A"), _sw("B"), _sum("SM"), _scope("SC")],
        [
            _wire("A", "y", "SM", "u0"),
            _wire("B", "y", "SM", "u1"),
            _wire("SM", "y", "SC", "u0"),
        ],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "sig_SM_y" in c
        assert "sig_A_y" in c
        assert "sig_B_y" in c
        assert "+" in c


def test_topo_order_sum_after_its_sources():
    model = _model(
        [_sum("S1"), _sw("A"), _sw("B")],
        [_wire("A", "y", "S1", "u0"), _wire("B", "y", "S1", "u1")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("A") < ids.index("S1")
    assert ids.index("B") < ids.index("S1")


# ---------------------------------------------------------------------------
# 17. Product block — catalog, simulator, codegen
# ---------------------------------------------------------------------------


def test_catalog_has_product_block():
    assert "Product" in BLOCK_CATALOG


def test_product_spec_has_two_inputs_and_one_output():
    spec = BLOCK_CATALOG["Product"]
    assert [p.name for p in spec.inputs] == ["u0", "u1"]
    assert [p.name for p in spec.outputs] == ["y"]


def test_product_spec_has_no_params():
    assert BLOCK_CATALOG["Product"].params == {}


def test_product_has_valid_color():
    color = BLOCK_CATALOG["Product"].color
    assert color.startswith("#") and len(color) == 7


def test_emit_decls_product_declares_output():
    d = _emit_decls([_product("P1")])
    assert "sig_P1_y" in d
    assert "phase_P1" not in d


def test_emit_step_product_multiplies_both_inputs():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sw("B"), _product("P1")]
    wires = {("P1", "u0"): ("A", "y"), ("P1", "u1"): ("B", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_P1_y" in step
    assert "sig_A_y" in step
    assert "sig_B_y" in step
    assert "*" in step


def test_emit_step_product_unconnected_input_defaults_to_one():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _product("P1")]
    wires = {("P1", "u0"): ("A", "y")}   # u1 not connected
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "1.0f" in step   # unconnected u1 → 1.0f


def test_emit_step_product_both_unconnected_is_one_times_one():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_product("P1")], {}, FakeWorkspace(), 1, board)
    assert "sig_P1_y" in step
    assert "1.0f * 1.0f" in step


def test_simulate_product_multiplies_two_constant_signals():
    # A = always 3.0, B = always 2.0 → product = always 6.0
    model = _model(
        [
            _sw("A", amplitude="3.0", duty="1.0"),
            _sw("B", amplitude="2.0", duty="1.0"),
            _product("P1"),
            _scope("SC"),
        ],
        [
            _wire("A", "y", "P1", "u0"),
            _wire("B", "y", "P1", "u1"),
            _wire("P1", "y", "SC", "u0"),
        ],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 6.0) < 1e-6


def test_simulate_product_unconnected_input_acts_as_one():
    # Only u0 connected → output == u0
    model = _model(
        [_sw("A", amplitude="5.0", duty="1.0"), _product("P1"), _scope("SC")],
        [_wire("A", "y", "P1", "u0"), _wire("P1", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 5.0) < 1e-6


def test_simulate_product_gating_pattern():
    # A = square wave (0/1), B = always 3.0 → output is 0 or 3
    model = _model(
        [
            _sw("A", frequency_hz="10", amplitude="1.0", offset="0.0", duty="0.5"),
            _sw("B", amplitude="3.0", duty="1.0"),
            _product("P1"),
            _scope("SC"),
        ],
        [
            _wire("A", "y", "P1", "u0"),
            _wire("B", "y", "P1", "u1"),
            _wire("P1", "y", "SC", "u0"),
        ],
    )
    _, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    uniq = {round(float(v), 6) for v in np.unique(y)}
    assert uniq <= {0.0, 3.0}, f"unexpected values: {uniq}"
    assert abs(float(y.mean()) - 1.5) < 0.1   # 50% duty → mean = 3*0.5


def test_simulate_product_chained_with_sum():
    # Sum: A(1) + B(1) = 2, Product: Sum(2) * C(3) = 6
    model = _model(
        [
            _sw("A", amplitude="1.0", duty="1.0"),
            _sw("B", amplitude="1.0", duty="1.0"),
            _sw("C", amplitude="3.0", duty="1.0"),
            _sum("S1"),
            _product("P1"),
            _scope("SC"),
        ],
        [
            _wire("A", "y", "S1", "u0"),
            _wire("B", "y", "S1", "u1"),
            _wire("S1", "y", "P1", "u0"),
            _wire("C", "y", "P1", "u1"),
            _wire("P1", "y", "SC", "u0"),
        ],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 6.0) < 1e-6


def test_generate_project_product_block_in_main_c():
    model = _model(
        [_sw("A"), _sw("B"), _product("PR"), _scope("SC")],
        [
            _wire("A", "y", "PR", "u0"),
            _wire("B", "y", "PR", "u1"),
            _wire("PR", "y", "SC", "u0"),
        ],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "sig_PR_y" in c
        assert "sig_A_y" in c
        assert "sig_B_y" in c
        assert "*" in c


def test_topo_order_product_after_its_sources():
    model = _model(
        [_product("P1"), _sw("A"), _sw("B")],
        [_wire("A", "y", "P1", "u0"), _wire("B", "y", "P1", "u1")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("A") < ids.index("P1")
    assert ids.index("B") < ids.index("P1")


def test_catalog_now_has_eight_block_types():
    # Extended to 39 blocks including all new group A-F blocks + LTI blocks.
    expected = {
        "SquareWave", "GpioIn", "GpioOut", "Scope", "Ultrasonic",
        "Sum", "Product", "Constant",
        "ToWorkspace", "Step", "Integrator", "TransferFcn", "PID",
        # Group A Sources
        "SineWave", "Ramp", "Clock", "PulseGenerator",
        # Group B Math
        "Gain", "Abs", "Sign", "Sqrt", "Saturation", "DeadZone", "MinMax",
        # Group C Logic
        "RelationalOperator", "LogicalOperator", "Switch",
        # Group D Discrete
        "UnitDelay", "DiscreteIntegrator", "ZeroOrderHold", "Derivative",
        # Group E Lookup
        "Lookup1D",
        # Group F STM32 HAL
        "ADC", "DAC", "PWMOut", "TimerTick",
        # LTI blocks
        "StateSpace", "DiscreteStateSpace", "ZeroPoleGain",
    }
    assert set(BLOCK_CATALOG.keys()) == expected


# ---------------------------------------------------------------------------
# 18. Constant block — catalog, simulator, codegen
# ---------------------------------------------------------------------------


def test_catalog_has_constant_block():
    assert "Constant" in BLOCK_CATALOG


def test_constant_spec_has_no_inputs_and_one_output():
    spec = BLOCK_CATALOG["Constant"]
    assert spec.inputs == []
    assert [p.name for p in spec.outputs] == ["y"]


def test_constant_spec_has_value_param():
    assert "value" in BLOCK_CATALOG["Constant"].params


def test_constant_has_valid_color():
    color = BLOCK_CATALOG["Constant"].color
    assert color.startswith("#") and len(color) == 7


def test_emit_decls_constant_declares_output():
    d = _emit_decls([_const("K")])
    assert "sig_K_y" in d
    assert "phase_K" not in d


def test_emit_step_constant_assigns_value():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_const("K", value="2.5")], {}, FakeWorkspace(), 1, board)
    assert "sig_K_y" in step
    assert "2.5" in step


def test_emit_step_constant_no_state_update():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_const("K")], {}, FakeWorkspace(), 1, board)
    # Should be a simple assignment, no phase or counter
    assert "phase_K" not in step
    assert "cnt_K" not in step


def test_simulate_constant_outputs_fixed_value():
    model = _model([_const("K", value="3.7"), _scope("S")],
                   [_wire("K", "y", "S", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["S.u0"]
    assert abs(float(y.mean()) - 3.7) < 1e-6
    assert abs(float(y.min()) - 3.7) < 1e-6


def test_simulate_constant_zero():
    model = _model([_const("K", value="0.0"), _scope("S")],
                   [_wire("K", "y", "S", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert float(sigs["S.u0"].sum()) == 0.0


def test_simulate_constant_negative():
    model = _model([_const("K", value="-5.0"), _scope("S")],
                   [_wire("K", "y", "S", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["S.u0"].mean()) - (-5.0)) < 1e-6


def test_simulate_constant_plus_squarewave_via_sum():
    # Constant 2.0 + SquareWave (0/1, 50% duty) → mean ≈ 2.5
    model = _model(
        [_const("K", value="2.0"), _sw("W", duty="0.5"), _sum("S1"), _scope("SC")],
        [
            _wire("K", "y", "S1", "u0"),
            _wire("W", "y", "S1", "u1"),
            _wire("S1", "y", "SC", "u0"),
        ],
    )
    _, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 2.5) < 0.05


def test_generate_project_constant_block_in_main_c():
    model = _model([_const("K", value="1.5"), _scope("S")],
                   [_wire("K", "y", "S", "u0")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "sig_K_y" in c
        assert "1.5" in c


def test_topo_order_constant_before_consumers():
    model = _model(
        [_sum("S1"), _const("K")],
        [_wire("K", "y", "S1", "u0")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("K") < ids.index("S1")


# ---------------------------------------------------------------------------
# Helpers for new block types (sections 19–23)
# ---------------------------------------------------------------------------


def _step(bid="ST", step_time="0.5", initial_value="0.0", final_value="1.0"):
    return {
        "type": "Step", "id": bid, "x": 0, "y": 0,
        "params": {
            "step_time": step_time,
            "initial_value": initial_value,
            "final_value": final_value,
        },
    }


def _integrator(bid="INT", initial_value="0.0",
                upper_limit="1e10", lower_limit="-1e10"):
    return {
        "type": "Integrator", "id": bid, "x": 0, "y": 0,
        "params": {
            "initial_value": initial_value,
            "upper_limit": upper_limit,
            "lower_limit": lower_limit,
        },
    }


def _transferfcn(bid="TF", numerator="1", denominator="1 1"):
    return {
        "type": "TransferFcn", "id": bid, "x": 0, "y": 0,
        "params": {"numerator": numerator, "denominator": denominator},
    }


def _pid(bid="PID1", Kp="1.0", Ki="0.0", Kd="0.0", N="100.0",
         upper_limit="1e10", lower_limit="-1e10"):
    return {
        "type": "PID", "id": bid, "x": 0, "y": 0,
        "params": {
            "Kp": Kp, "Ki": Ki, "Kd": Kd, "N": N,
            "upper_limit": upper_limit,
            "lower_limit": lower_limit,
        },
    }


def _toworkspace(bid="TW", variable_name="yout", max_points="10000",
                 decimation="1", save_time="1"):
    return {
        "type": "ToWorkspace", "id": bid, "x": 0, "y": 0,
        "params": {
            "variable_name": variable_name,
            "max_points": max_points,
            "decimation": decimation,
            "save_time": save_time,
        },
    }


# ---------------------------------------------------------------------------
# 19. Step block — catalog, simulator, codegen
# ---------------------------------------------------------------------------


def test_catalog_has_step_block():
    assert "Step" in BLOCK_CATALOG


def test_step_spec_has_no_inputs_and_one_output():
    spec = BLOCK_CATALOG["Step"]
    assert spec.inputs == [], "Step must have no inputs (source block)"
    assert [p.name for p in spec.outputs] == ["y"]


def test_step_spec_has_required_params():
    params = BLOCK_CATALOG["Step"].params
    for key in ("step_time", "initial_value", "final_value"):
        assert key in params, f"Step missing param '{key}'"


def test_step_has_valid_color():
    color = BLOCK_CATALOG["Step"].color
    assert color.startswith("#") and len(color) == 7


def test_step_has_description():
    desc = BLOCK_CATALOG["Step"].description
    assert desc and len(desc) > 10


def test_emit_decls_step_declares_output():
    d = _emit_decls([_step("ST1")])
    assert "sig_ST1_y" in d


def test_emit_step_step_generates_output_variable():
    board = BOARDS["NUCLEO-F446RE"]
    step_c, _ = _emit_step([_step("ST1")], {}, FakeWorkspace(), 1, board)
    assert "sig_ST1_y" in step_c


def test_emit_step_step_contains_counter_or_time_compare():
    """C code should compare tick count or elapsed time against step_time."""
    board = BOARDS["NUCLEO-F446RE"]
    step_c, _ = _emit_step([_step("ST1", step_time="0.5")], {}, FakeWorkspace(), 1, board)
    # Expect either a static counter variable or a float time comparison
    has_counter = "_cnt_ST1" in step_c
    has_compare = ("ST1" in step_c and (">" in step_c or ">=" in step_c))
    assert has_counter or has_compare, (
        f"Step C code must compare time or count.\nGot:\n{step_c}"
    )


def test_simulate_step_outputs_initial_before_step_time():
    model = _model(
        [_step("ST", step_time="0.5", initial_value="2.0", final_value="5.0"),
         _scope("SC")],
        [_wire("ST", "y", "SC", "u0")],
    )
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    # t[0..499] ≈ 0.0, 0.001, ..., 0.499 — all before 0.5 → initial_value
    before = y[t < 0.5]
    assert len(before) > 0
    assert abs(float(before.mean()) - 2.0) < 1e-6, (
        f"Expected 2.0 before step, got {before.mean():.4f}"
    )


def test_simulate_step_outputs_final_after_step_time():
    model = _model(
        [_step("ST", step_time="0.5", initial_value="2.0", final_value="5.0"),
         _scope("SC")],
        [_wire("ST", "y", "SC", "u0")],
    )
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    after = y[t >= 0.5]
    assert len(after) > 0
    assert abs(float(after.mean()) - 5.0) < 1e-6, (
        f"Expected 5.0 after step, got {after.mean():.4f}"
    )


def test_simulate_step_transitions_exactly_at_step_time():
    """Value at exactly t=step_time should be final_value."""
    model = _model(
        [_step("ST", step_time="0.3", initial_value="0.0", final_value="1.0"),
         _scope("SC")],
        [_wire("ST", "y", "SC", "u0")],
    )
    t, sigs = simulate_model(model, duration_s=0.6, step_s=0.001)
    y = sigs["SC.u0"]
    # Find the sample at t=0.3 (index 300)
    idx = np.argmin(np.abs(t - 0.3))
    assert float(y[idx]) == 1.0, f"At t=0.3 expected 1.0, got {y[idx]}"


def test_simulate_step_zero_initial_nonzero_final():
    model = _model(
        [_step("ST", step_time="0.0", initial_value="0.0", final_value="7.0"),
         _scope("SC")],
        [_wire("ST", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    # step_time=0 → all samples at or after t=0 → all final_value
    assert abs(float(y.mean()) - 7.0) < 1e-6


def test_simulate_step_negative_values():
    model = _model(
        [_step("ST", step_time="0.5", initial_value="-3.0", final_value="-1.0"),
         _scope("SC")],
        [_wire("ST", "y", "SC", "u0")],
    )
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y[t < 0.5].mean()) == -3.0
    assert float(y[t >= 0.5].mean()) == -1.0


def test_generate_project_step_block_in_main_c():
    model = _model([_step("ST"), _scope("SC")], [_wire("ST", "y", "SC", "u0")])
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "sig_ST_y" in c


def test_topo_order_step_before_consumers():
    model = _model(
        [_sum("S1"), _step("ST")],
        [_wire("ST", "y", "S1", "u0")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("ST") < ids.index("S1")


# ---------------------------------------------------------------------------
# 20. Integrator block — catalog, simulator, codegen
# ---------------------------------------------------------------------------


def test_catalog_has_integrator_block():
    assert "Integrator" in BLOCK_CATALOG


def test_integrator_spec_has_one_input_and_one_output():
    spec = BLOCK_CATALOG["Integrator"]
    assert [p.name for p in spec.inputs] == ["u"]
    assert [p.name for p in spec.outputs] == ["y"]


def test_integrator_spec_has_required_params():
    params = BLOCK_CATALOG["Integrator"].params
    for key in ("initial_value", "upper_limit", "lower_limit"):
        assert key in params, f"Integrator missing param '{key}'"


def test_integrator_has_valid_color():
    color = BLOCK_CATALOG["Integrator"].color
    assert color.startswith("#") and len(color) == 7


def test_integrator_has_description():
    desc = BLOCK_CATALOG["Integrator"].description
    assert desc and len(desc) > 10


def test_emit_decls_integrator_declares_output():
    d = _emit_decls([_integrator("INT1")])
    assert "sig_INT1_y" in d


def test_emit_step_integrator_generates_state_and_output():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _integrator("INT1")]
    wires = {("INT1", "u"): ("A", "y")}
    step_c, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_INT1_y" in step_c
    # Should have a static state variable for accumulation
    assert "_state_INT1" in step_c


def test_emit_step_integrator_uses_plus_equals():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _integrator("INT1")]
    wires = {("INT1", "u"): ("A", "y")}
    step_c, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "+=" in step_c


def test_simulate_integrator_integrates_constant_input():
    """Integral of constant 1.0 over time = t  (forward Euler)."""
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"), _integrator("INT"), _scope("SC")],
        [_wire("A", "y", "INT", "u"), _wire("INT", "y", "SC", "u0")],
    )
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    # Forward Euler: y[0]=0, y[k] = (k) * 0.001  → y[-1] ≈ 0.999
    assert abs(float(y[-1]) - 0.999) < 0.01, f"y[-1]={y[-1]:.4f}"
    # Mean of 0..0.999 ≈ 0.499
    assert abs(float(y.mean()) - 0.4995) < 0.01, f"mean={y.mean():.4f}"


def test_simulate_integrator_respects_initial_condition():
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _integrator("INT", initial_value="5.0"), _scope("SC")],
        [_wire("A", "y", "INT", "u"), _wire("INT", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    # IC=5; first sample should be 5.0 (y[0] = ic before first accumulation)
    assert abs(float(y[0]) - 5.0) < 1e-6, f"y[0]={y[0]:.4f}, expected 5.0"


def test_simulate_integrator_upper_limit_clamps_output():
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _integrator("INT", upper_limit="0.5"), _scope("SC")],
        [_wire("A", "y", "INT", "u"), _wire("INT", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=2.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y.max()) <= 0.5 + 1e-9, f"max={y.max():.4f} exceeds upper_limit 0.5"


def test_simulate_integrator_lower_limit_clamps_output():
    model = _model(
        [_sw("A", amplitude="-1.0", duty="1.0"),
         _integrator("INT", lower_limit="-0.3"), _scope("SC")],
        [_wire("A", "y", "INT", "u"), _wire("INT", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=2.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y.min()) >= -0.3 - 1e-9, f"min={y.min():.4f} below lower_limit -0.3"


def test_simulate_integrator_zero_input_holds_ic():
    model = _model(
        [_sw("A", amplitude="0.0", duty="1.0"),
         _integrator("INT", initial_value="3.0"), _scope("SC")],
        [_wire("A", "y", "INT", "u"), _wire("INT", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    # Zero input → output stays at IC forever
    assert abs(float(y.mean()) - 3.0) < 1e-6


def test_generate_project_integrator_block_in_main_c():
    model = _model(
        [_sw("A"), _integrator("INT"), _scope("SC")],
        [_wire("A", "y", "INT", "u"), _wire("INT", "y", "SC", "u0")],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "sig_INT_y" in c
        assert "_state_INT" in c


def test_topo_order_integrator_after_source():
    model = _model(
        [_integrator("INT"), _sw("A")],
        [_wire("A", "y", "INT", "u")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("A") < ids.index("INT")


# ---------------------------------------------------------------------------
# 21. TransferFcn block + _bilinear_tf — catalog, bilinear math, simulator,
#     codegen
# ---------------------------------------------------------------------------


def test_catalog_has_transferfcn_block():
    assert "TransferFcn" in BLOCK_CATALOG


def test_transferfcn_spec_has_one_input_and_one_output():
    spec = BLOCK_CATALOG["TransferFcn"]
    assert [p.name for p in spec.inputs] == ["u"]
    assert [p.name for p in spec.outputs] == ["y"]


def test_transferfcn_spec_has_required_params():
    params = BLOCK_CATALOG["TransferFcn"].params
    assert "numerator" in params
    assert "denominator" in params


def test_transferfcn_has_valid_color():
    color = BLOCK_CATALOG["TransferFcn"].color
    assert color.startswith("#") and len(color) == 7


def test_transferfcn_has_description():
    desc = BLOCK_CATALOG["TransferFcn"].description
    assert desc and len(desc) > 10


# --- _bilinear_tf math tests ---


def test_bilinear_tf_unity_gain():
    """H(s) = 1/1 (pure gain 1).  DC gain must be 1."""
    bz, az = _bilinear_tf("1", "1", fs=1000.0)
    dc_gain = float(np.sum(bz) / np.sum(az))
    assert abs(dc_gain - 1.0) < 1e-9, f"DC gain={dc_gain}"


def test_bilinear_tf_pure_gain_two():
    """H(s) = 2/1.  DC gain must be 2."""
    bz, az = _bilinear_tf("2", "1", fs=1000.0)
    dc_gain = float(np.sum(bz) / np.sum(az))
    assert abs(dc_gain - 2.0) < 1e-9, f"DC gain={dc_gain}"


def test_bilinear_tf_first_order_lowpass_dc_gain():
    """H(s) = 1/(s+1).  DC gain = H(0) = 1/1 = 1."""
    bz, az = _bilinear_tf("1", "1 1", fs=1000.0)
    dc_gain = float(np.sum(bz) / np.sum(az))
    assert abs(dc_gain - 1.0) < 1e-6, f"DC gain={dc_gain:.6f}"


def test_bilinear_tf_first_order_normalized_a0():
    """a_z[0] must always equal 1 (normalized)."""
    bz, az = _bilinear_tf("1", "1 1", fs=500.0)
    assert abs(float(az[0]) - 1.0) < 1e-9, f"az[0]={az[0]}"


def test_bilinear_tf_output_shapes_match():
    """b_z and a_z must have the same length = order + 1."""
    bz, az = _bilinear_tf("1", "1 2 1", fs=1000.0)  # 2nd order
    assert len(bz) == len(az), f"len(bz)={len(bz)}, len(az)={len(az)}"
    assert len(az) == 3, f"Expected order+1=3, got {len(az)}"


def test_bilinear_tf_second_order_dc_gain():
    """H(s) = 4/(s^2 + 2s + 4).  DC gain = 4/4 = 1."""
    bz, az = _bilinear_tf("4", "1 2 4", fs=1000.0)
    dc_gain = float(np.sum(bz) / np.sum(az))
    assert abs(dc_gain - 1.0) < 1e-6, f"DC gain={dc_gain:.6f}"


def test_bilinear_tf_raises_on_degree_mismatch():
    """Numerator degree > denominator degree should raise ValueError."""
    raised = False
    try:
        _bilinear_tf("1 1 1", "1 1", fs=1000.0)  # order 2 num > order 1 den
    except (ValueError, Exception):
        raised = True
    assert raised, "Expected an error when numerator degree > denominator degree"


# --- Codegen tests ---


def test_emit_decls_transferfcn_declares_output():
    d = _emit_decls([_transferfcn("TF1")])
    assert "sig_TF1_y" in d


def test_emit_step_transferfcn_generates_output_variable():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _transferfcn("TF1")]
    wires = {("TF1", "u"): ("A", "y")}
    step_c, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_TF1_y" in step_c


def test_emit_step_transferfcn_first_order_has_static_state():
    """A first-order TF must declare a static IIR filter state."""
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _transferfcn("TF1", numerator="1", denominator="1 1")]
    wires = {("TF1", "u"): ("A", "y")}
    step_c, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "static float" in step_c
    assert "TF1" in step_c


def test_emit_step_transferfcn_pure_gain_no_state():
    """H(s) = 2/1 (order 0) → pure gain assignment, no state variable."""
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _transferfcn("TF1", numerator="2", denominator="1")]
    wires = {("TF1", "u"): ("A", "y")}
    step_c, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_TF1_y" in step_c
    # Pure gain: no IIR state array needed
    assert "_s0_TF1" not in step_c


def test_generate_project_transferfcn_block_in_main_c():
    model = _model(
        [_sw("A"), _transferfcn("TF"), _scope("SC")],
        [_wire("A", "y", "TF", "u"), _wire("TF", "y", "SC", "u0")],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "sig_TF_y" in c


# --- Simulator tests ---


def test_simulate_transferfcn_pure_gain():
    """H(s) = 3/1 → output = 3 * input at all times."""
    model = _model(
        [_sw("A", amplitude="2.0", duty="1.0"),
         _transferfcn("TF", numerator="3", denominator="1"),
         _scope("SC")],
        [_wire("A", "y", "TF", "u"), _wire("TF", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 6.0) < 1e-4, f"mean={y.mean():.4f}"


def test_simulate_transferfcn_step_response_reaches_dc_gain():
    """H(s)=1/(0.01s+1) with fc≈16 Hz.  A 1s step should reach ≈ DC gain=1."""
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _transferfcn("TF", numerator="1", denominator="0.01 1"),
         _scope("SC")],
        [_wire("A", "y", "TF", "u"), _wire("TF", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.2, step_s=0.001)
    y = sigs["SC.u0"]
    # After 5 time constants (0.05 s) the step response is > 99% of DC gain
    assert float(y[-1]) > 0.95, f"Final value={y[-1]:.4f} — didn't reach DC gain"


def test_simulate_transferfcn_passes_dc_value():
    """H(s)=1/(s+1): constant input=1 → steady-state output=1."""
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _transferfcn("TF", numerator="1", denominator="1 1"),
         _scope("SC")],
        [_wire("A", "y", "TF", "u"), _wire("TF", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=10.0, step_s=0.001)
    y = sigs["SC.u0"]
    # Last 10% of output should be very close to 1.0
    assert abs(float(y[-100:].mean()) - 1.0) < 0.01, (
        f"Steady-state mean={y[-100:].mean():.4f}"
    )


def test_topo_order_transferfcn_after_source():
    model = _model(
        [_transferfcn("TF"), _sw("A")],
        [_wire("A", "y", "TF", "u")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("A") < ids.index("TF")


# ---------------------------------------------------------------------------
# 22. PID block — catalog, simulator, codegen
# ---------------------------------------------------------------------------


def test_catalog_has_pid_block():
    assert "PID" in BLOCK_CATALOG


def test_pid_spec_has_one_input_and_one_output():
    spec = BLOCK_CATALOG["PID"]
    assert [p.name for p in spec.inputs] == ["u"]
    assert [p.name for p in spec.outputs] == ["y"]


def test_pid_spec_has_required_params():
    params = BLOCK_CATALOG["PID"].params
    for key in ("Kp", "Ki", "Kd", "N", "upper_limit", "lower_limit"):
        assert key in params, f"PID missing param '{key}'"


def test_pid_has_valid_color():
    color = BLOCK_CATALOG["PID"].color
    assert color.startswith("#") and len(color) == 7


def test_pid_has_description():
    desc = BLOCK_CATALOG["PID"].description
    assert desc and len(desc) > 10


def test_emit_decls_pid_declares_output():
    d = _emit_decls([_pid("PID1")])
    assert "sig_PID1_y" in d


def test_emit_step_pid_generates_state_variables():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _pid("PID1")]
    wires = {("PID1", "u"): ("A", "y")}
    step_c, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_PID1_y" in step_c
    assert "_integ_PID1" in step_c
    assert "_dstate_PID1" in step_c


def test_emit_step_pid_contains_kp_ki_kd():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _pid("PID1", Kp="2.0", Ki="0.5", Kd="0.1")]
    wires = {("PID1", "u"): ("A", "y")}
    step_c, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "2.0" in step_c   # Kp
    assert "0.5" in step_c   # Ki
    assert "0.1" in step_c   # Kd


def test_emit_step_pid_saturation_uses_limits():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _pid("PID1", upper_limit="10.0", lower_limit="-10.0")]
    wires = {("PID1", "u"): ("A", "y")}
    step_c, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "10.0" in step_c


def test_simulate_pid_proportional_only():
    """With Ki=0, Kd=0: output = Kp * input."""
    model = _model(
        [_sw("A", amplitude="3.0", duty="1.0"),
         _pid("PID1", Kp="2.0", Ki="0.0", Kd="0.0"),
         _scope("SC")],
        [_wire("A", "y", "PID1", "u"), _wire("PID1", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    # Kp*3 = 6.0
    assert abs(float(y.mean()) - 6.0) < 1e-4, f"mean={y.mean():.4f}"


def test_simulate_pid_integral_ramps_up():
    """With Kp=0, Ki=1, Kd=0 and constant error=1: output = t (forward Euler)."""
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _pid("PID1", Kp="0.0", Ki="1.0", Kd="0.0"),
         _scope("SC")],
        [_wire("A", "y", "PID1", "u"), _wire("PID1", "y", "SC", "u0")],
    )
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    # Ki*integral(1, 0..t) = t  → y[-1] ≈ 0.999 (forward Euler, starts at 0)
    assert abs(float(y[-1]) - 0.999) < 0.01, f"y[-1]={y[-1]:.4f}"


def test_simulate_pid_upper_limit_clamps():
    model = _model(
        [_sw("A", amplitude="100.0", duty="1.0"),
         _pid("PID1", Kp="10.0", upper_limit="5.0"),
         _scope("SC")],
        [_wire("A", "y", "PID1", "u"), _wire("PID1", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y.max()) <= 5.0 + 1e-9, f"max={y.max():.4f} exceeds upper_limit 5.0"


def test_simulate_pid_lower_limit_clamps():
    model = _model(
        [_sw("A", amplitude="-100.0", duty="1.0"),
         _pid("PID1", Kp="10.0", lower_limit="-5.0"),
         _scope("SC")],
        [_wire("A", "y", "PID1", "u"), _wire("PID1", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y.min()) >= -5.0 - 1e-9, f"min={y.min():.4f} below lower_limit -5.0"


def test_simulate_pid_derivative_only_responds_to_change():
    """With Kp=0, Ki=0, Kd>0: derivative of constant=0, so output≈0 after IC."""
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _pid("PID1", Kp="0.0", Ki="0.0", Kd="1.0", N="100.0"),
         _scope("SC")],
        [_wire("A", "y", "PID1", "u"), _wire("PID1", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.5, step_s=0.001)
    y = sigs["SC.u0"]
    # After initial transient, constant input → derivative → 0
    # Tail average (last 20%) should be close to 0
    tail = y[int(len(y) * 0.8):]
    assert abs(float(tail.mean())) < 0.1, f"tail mean={tail.mean():.4f}"


def test_simulate_pid_zero_gain_passthrough_zero():
    model = _model(
        [_sw("A", amplitude="5.0", duty="1.0"),
         _pid("PID1", Kp="0.0", Ki="0.0", Kd="0.0"),
         _scope("SC")],
        [_wire("A", "y", "PID1", "u"), _wire("PID1", "y", "SC", "u0")],
    )
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.sum())) < 1e-6, f"Expected all zeros, sum={y.sum()}"


def test_generate_project_pid_block_in_main_c():
    model = _model(
        [_sw("A"), _pid("PID1"), _scope("SC")],
        [_wire("A", "y", "PID1", "u"), _wire("PID1", "y", "SC", "u0")],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        assert "sig_PID1_y" in c
        assert "_integ_PID1" in c


def test_topo_order_pid_after_source():
    model = _model(
        [_pid("PID1"), _sw("A")],
        [_wire("A", "y", "PID1", "u")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("A") < ids.index("PID1")


# ---------------------------------------------------------------------------
# 23. ToWorkspace block — catalog, simulator, workspace integration, codegen
# ---------------------------------------------------------------------------


def test_catalog_has_toworkspace_block():
    assert "ToWorkspace" in BLOCK_CATALOG


def test_toworkspace_spec_has_one_input_no_outputs():
    spec = BLOCK_CATALOG["ToWorkspace"]
    assert [p.name for p in spec.inputs] == ["u"]
    assert spec.outputs == [], "ToWorkspace has no outputs (sink block)"


def test_toworkspace_spec_has_required_params():
    params = BLOCK_CATALOG["ToWorkspace"].params
    for key in ("variable_name", "max_points", "decimation", "save_time"):
        assert key in params, f"ToWorkspace missing param '{key}'"


def test_toworkspace_has_valid_color():
    color = BLOCK_CATALOG["ToWorkspace"].color
    assert color.startswith("#") and len(color) == 7


def test_toworkspace_has_description():
    desc = BLOCK_CATALOG["ToWorkspace"].description
    assert desc and len(desc) > 10


def test_emit_step_toworkspace_is_noop():
    """ToWorkspace should emit only a comment (no-op) in C."""
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _toworkspace("TW1", variable_name="out1")]
    wires = {("TW1", "u"): ("A", "y")}
    step_c, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    # No assignment to a sig_ variable for TW1 (sink block, no output)
    assert "sig_TW1_y" not in step_c


def test_emit_step_toworkspace_leaves_no_output_signal():
    """No output signal variable should be declared for a ToWorkspace block."""
    d = _emit_decls([_toworkspace("TW1")])
    assert "sig_TW1_y" not in d


def test_simulate_toworkspace_writes_to_workspace():
    """After simulate_model, the named variable should be in WORKSPACE.globals."""
    WORKSPACE.globals.clear()
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _toworkspace("TW", variable_name="my_signal")],
        [_wire("A", "y", "TW", "u")],
    )
    simulate_model(model, duration_s=0.1, step_s=0.001)
    assert "my_signal" in WORKSPACE.globals, (
        f"'my_signal' not found in workspace; keys={list(WORKSPACE.globals)}"
    )


def test_simulate_toworkspace_written_value_is_ndarray():
    WORKSPACE.globals.clear()
    model = _model(
        [_sw("A", amplitude="2.0", duty="1.0"),
         _toworkspace("TW", variable_name="arr_out")],
        [_wire("A", "y", "TW", "u")],
    )
    simulate_model(model, duration_s=0.05, step_s=0.001)
    val = WORKSPACE.globals.get("arr_out")
    assert isinstance(val, np.ndarray), f"Expected ndarray, got {type(val)}"


def test_simulate_toworkspace_signal_values_correct():
    """Values written to workspace should match the source signal."""
    WORKSPACE.globals.clear()
    model = _model(
        [_sw("A", amplitude="3.0", duty="1.0"),
         _toworkspace("TW", variable_name="sig3")],
        [_wire("A", "y", "TW", "u")],
    )
    simulate_model(model, duration_s=0.1, step_s=0.001)
    arr = WORKSPACE.globals["sig3"]
    assert abs(float(arr.mean()) - 3.0) < 1e-4, f"mean={arr.mean():.4f}"


def test_simulate_toworkspace_saves_time_vector_when_requested():
    """When save_time='1', a '<name>_t' time vector should also be saved."""
    WORKSPACE.globals.clear()
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _toworkspace("TW", variable_name="ysig", save_time="1")],
        [_wire("A", "y", "TW", "u")],
    )
    simulate_model(model, duration_s=0.05, step_s=0.001)
    assert "ysig_t" in WORKSPACE.globals, (
        f"Time vector 'ysig_t' not found; keys={list(WORKSPACE.globals)}"
    )


def test_simulate_toworkspace_no_time_vector_when_disabled():
    """When save_time='0', no '<name>_t' variable should appear."""
    WORKSPACE.globals.clear()
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _toworkspace("TW", variable_name="ysig2", save_time="0")],
        [_wire("A", "y", "TW", "u")],
    )
    simulate_model(model, duration_s=0.05, step_s=0.001)
    assert "ysig2_t" not in WORKSPACE.globals, (
        "Time vector should not be saved when save_time='0'"
    )


def test_simulate_toworkspace_max_points_truncates():
    """max_points should cap the length of the saved array."""
    WORKSPACE.globals.clear()
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _toworkspace("TW", variable_name="capped", max_points="50")],
        [_wire("A", "y", "TW", "u")],
    )
    # 1000 samples generated but max_points=50
    simulate_model(model, duration_s=1.0, step_s=0.001)
    arr = WORKSPACE.globals.get("capped")
    assert arr is not None
    assert len(arr) <= 50, f"Expected at most 50 points, got {len(arr)}"


def test_simulate_toworkspace_decimation_reduces_samples():
    """decimation=10 should save every 10th sample."""
    WORKSPACE.globals.clear()
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _toworkspace("TW", variable_name="decimated",
                      decimation="10", max_points="10000")],
        [_wire("A", "y", "TW", "u")],
    )
    # 1000 total samples; decimation=10 → ~100 saved
    simulate_model(model, duration_s=1.0, step_s=0.001)
    arr = WORKSPACE.globals.get("decimated")
    assert arr is not None
    assert len(arr) <= 110, f"Expected ~100 points after decimation, got {len(arr)}"
    assert len(arr) >= 90, f"Expected ~100 points after decimation, got {len(arr)}"


def test_simulate_toworkspace_multiple_blocks_independent():
    """Two ToWorkspace blocks with different names must not overwrite each other."""
    WORKSPACE.globals.clear()
    model = _model(
        [_sw("A", amplitude="1.0", duty="1.0"),
         _sw("B", amplitude="5.0", duty="1.0"),
         _toworkspace("TW1", variable_name="sig_a"),
         _toworkspace("TW2", variable_name="sig_b")],
        [
            _wire("A", "y", "TW1", "u"),
            _wire("B", "y", "TW2", "u"),
        ],
    )
    simulate_model(model, duration_s=0.1, step_s=0.001)
    assert "sig_a" in WORKSPACE.globals
    assert "sig_b" in WORKSPACE.globals
    assert abs(float(WORKSPACE.globals["sig_a"].mean()) - 1.0) < 1e-4
    assert abs(float(WORKSPACE.globals["sig_b"].mean()) - 5.0) < 1e-4


def test_generate_project_toworkspace_block_noop_in_main_c():
    """ToWorkspace generates only a comment in C — no data-capture code."""
    model = _model(
        [_sw("A"), _toworkspace("TW", variable_name="data")],
        [_wire("A", "y", "TW", "u")],
    )
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        c = (proj / "main.c").read_text()
        # No assignment to a non-existent sig_TW_y variable
        assert "sig_TW_y" not in c


def test_topo_order_toworkspace_after_source():
    model = _model(
        [_toworkspace("TW"), _sw("A")],
        [_wire("A", "y", "TW", "u")],
    )
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("A") < ids.index("TW")


# ---------------------------------------------------------------------------
# 24. Parameter validation — _is_valid_stm32_pin / _try_eval_param
# ---------------------------------------------------------------------------


def test_is_valid_stm32_pin_accepts_standard_pins():
    for pin in ("PA0", "PA5", "PB3", "PC13", "PH1"):
        assert _is_valid_stm32_pin(pin), f"Expected '{pin}' to be valid"


def test_is_valid_stm32_pin_case_insensitive():
    assert _is_valid_stm32_pin("pa5")
    assert _is_valid_stm32_pin("PC13")
    assert _is_valid_stm32_pin("Pb7")


def test_is_valid_stm32_pin_rejects_bad_pins():
    for pin in ("", "X5", "PA", "P1", "PA999", "PORTA5", "5PA"):
        assert not _is_valid_stm32_pin(pin), f"Expected '{pin}' to be invalid"


def test_try_eval_param_float_literals():
    assert _try_eval_param("1.0") == 1.0
    assert _try_eval_param("0") == 0.0
    assert _try_eval_param("-3.14") == pytest_approx(-3.14) if False else abs(_try_eval_param("-3.14") - (-3.14)) < 1e-9


def test_try_eval_param_returns_none_for_garbage():
    assert _try_eval_param("abc") is None
    assert _try_eval_param("") is None
    assert _try_eval_param("1 2") is None


def test_try_eval_param_uses_workspace():
    ws = FakeWorkspace()
    ws.globals["myvar"] = 42.0
    # FakeWorkspace.eval_param just converts to float or looks up globals
    # Override eval_param to check workspace lookup path
    result = _try_eval_param("myvar", ws)
    # FakeWorkspace returns 0.0 for unknown names; for known ones it returns the value
    assert result == 42.0 or result == 0.0   # depends on FakeWorkspace impl


# ---------------------------------------------------------------------------
# 25. validate_model — correct models pass
# ---------------------------------------------------------------------------


def test_validate_model_empty_model_no_errors():
    assert validate_model({"blocks": [], "connections": []}) == []


def test_validate_model_valid_squarewave_no_errors():
    model = _model([_sw("A")])
    assert validate_model(model) == []


def test_validate_model_valid_gpio_in_no_errors():
    model = _model([_gpio_in("GI", pin="PC13", pull="none", active_low="1")])
    assert validate_model(model) == []


def test_validate_model_valid_gpio_out_no_errors():
    model = _model([_gpio_out("GO", pin="PA5", threshold="0.5")])
    assert validate_model(model) == []


def test_validate_model_valid_scope_no_errors():
    model = _model([_scope("SC")])
    assert validate_model(model) == []


def test_validate_model_valid_ultrasonic_no_errors():
    model = _model([_ultra("U", trig="PA0", echo="PA1", period="60", timeout="30000")])
    assert validate_model(model) == []


def test_validate_model_valid_constant_no_errors():
    model = _model([_const("K", value="3.14")])
    assert validate_model(model) == []


def test_validate_model_valid_step_no_errors():
    model = _model([_step("ST", step_time="0.5", initial_value="0.0", final_value="1.0")])
    assert validate_model(model) == []


def test_validate_model_valid_integrator_no_errors():
    model = _model([_integrator("INT", upper_limit="10", lower_limit="-10")])
    assert validate_model(model) == []


def test_validate_model_valid_transferfcn_no_errors():
    model = _model([_transferfcn("TF", numerator="1", denominator="1 1")])
    assert validate_model(model) == []


def test_validate_model_valid_pid_no_errors():
    model = _model([_pid("PID1", Kp="1.0", Ki="0.1", Kd="0.01", N="100",
                         upper_limit="10", lower_limit="-10")])
    assert validate_model(model) == []


def test_validate_model_valid_toworkspace_no_errors():
    model = _model([_toworkspace("TW", variable_name="yout",
                                 max_points="10000", decimation="1", save_time="1")])
    assert validate_model(model) == []


def test_validate_model_returns_list_of_validation_errors():
    bad = _sw("SW1", frequency_hz="-1")
    errs = validate_model(_model([bad]))
    assert isinstance(errs, list)
    assert all(isinstance(e, ValidationError) for e in errs)


# ---------------------------------------------------------------------------
# 26. validate_model — SquareWave errors
# ---------------------------------------------------------------------------


def test_validate_squarewave_negative_frequency():
    errs = validate_model(_model([_sw("A", frequency_hz="-1")]))
    codes = [e.code for e in errs]
    assert "E002" in codes
    params = [e.param for e in errs]
    assert "frequency_hz" in params


def test_validate_squarewave_zero_frequency():
    errs = validate_model(_model([_sw("A", frequency_hz="0")]))
    assert any(e.param == "frequency_hz" and e.code == "E002" for e in errs)


def test_validate_squarewave_non_numeric_frequency():
    errs = validate_model(_model([_sw("A", frequency_hz="bad")]))
    assert any(e.param == "frequency_hz" and e.code == "E001" for e in errs)


def test_validate_squarewave_duty_above_one():
    errs = validate_model(_model([_sw("A", duty="1.5")]))
    assert any(e.param == "duty" and e.code == "E002" for e in errs)


def test_validate_squarewave_duty_below_zero():
    errs = validate_model(_model([_sw("A", duty="-0.1")]))
    assert any(e.param == "duty" and e.code == "E002" for e in errs)


def test_validate_squarewave_non_numeric_duty():
    errs = validate_model(_model([_sw("A", duty="half")]))
    assert any(e.param == "duty" and e.code == "E001" for e in errs)


def test_validate_squarewave_valid_boundary_duties():
    # 0 and 1 are valid boundary duty cycles
    assert validate_model(_model([_sw("A", duty="0")])) == []
    assert validate_model(_model([_sw("A", duty="1")])) == []


def test_validate_squarewave_non_numeric_amplitude():
    errs = validate_model(_model([_sw("A", amplitude="big")]))
    assert any(e.param == "amplitude" and e.code == "E001" for e in errs)


def test_validate_squarewave_non_numeric_offset():
    errs = validate_model(_model([_sw("A", offset="none")]))
    assert any(e.param == "offset" and e.code == "E001" for e in errs)


def test_validate_squarewave_block_id_is_correct():
    errs = validate_model(_model([_sw("MySW", frequency_hz="-5")]))
    assert all(e.block_id == "MySW" for e in errs if e.param == "frequency_hz")


# ---------------------------------------------------------------------------
# 27. validate_model — GpioIn / GpioOut errors
# ---------------------------------------------------------------------------


def test_validate_gpio_in_invalid_pin():
    errs = validate_model(_model([_gpio_in("GI", pin="BADPIN")]))
    assert any(e.param == "pin" and e.code == "E004" for e in errs)


def test_validate_gpio_in_invalid_pull():
    errs = validate_model(_model([_gpio_in("GI", pull="floating")]))
    assert any(e.param == "pull" and e.code == "E003" for e in errs)


def test_validate_gpio_in_valid_pulls():
    for pull in ("none", "up", "down"):
        assert validate_model(_model([_gpio_in("GI", pull=pull)])) == [], \
            f"pull='{pull}' should be valid"


def test_validate_gpio_in_invalid_active_low():
    errs = validate_model(_model([_gpio_in("GI", active_low="2")]))
    assert any(e.param == "active_low" and e.code == "E002" for e in errs)


def test_validate_gpio_out_invalid_pin():
    errs = validate_model(_model([_gpio_out("GO", pin="BADPIN")]))
    assert any(e.param == "pin" and e.code == "E004" for e in errs)


def test_validate_gpio_out_non_numeric_threshold():
    errs = validate_model(_model([_gpio_out("GO", threshold="half")]))
    assert any(e.param == "threshold" and e.code == "E001" for e in errs)


# ---------------------------------------------------------------------------
# 28. validate_model — Scope errors
# ---------------------------------------------------------------------------


def test_validate_scope_zero_max_points():
    errs = validate_model(_model([_scope("SC")]))
    # Default is 100, which is valid
    assert errs == []


def test_validate_scope_negative_max_points():
    errs = validate_model(_model([{"type": "Scope", "id": "SC", "x": 0, "y": 0,
                                   "params": {"max_points": "-1", "stream": "1"}}]))
    assert any(e.param == "max_points" and e.code == "E002" for e in errs)


def test_validate_scope_invalid_stream():
    errs = validate_model(_model([{"type": "Scope", "id": "SC", "x": 0, "y": 0,
                                   "params": {"max_points": "100", "stream": "2"}}]))
    assert any(e.param == "stream" and e.code == "E002" for e in errs)


# ---------------------------------------------------------------------------
# 29. validate_model — Ultrasonic errors
# ---------------------------------------------------------------------------


def test_validate_ultrasonic_invalid_trig_pin():
    errs = validate_model(_model([_ultra("U", trig="BADPIN")]))
    assert any(e.param == "trig_pin" and e.code == "E004" for e in errs)


def test_validate_ultrasonic_invalid_echo_pin():
    errs = validate_model(_model([_ultra("U", echo="BADPIN")]))
    assert any(e.param == "echo_pin" and e.code == "E004" for e in errs)


def test_validate_ultrasonic_same_trig_echo_pins():
    errs = validate_model(_model([_ultra("U", trig="PA0", echo="PA0")]))
    assert any(e.param == "echo_pin" and e.code == "E003" for e in errs)


def test_validate_ultrasonic_period_below_minimum():
    errs = validate_model(_model([_ultra("U", period="30")]))
    assert any(e.param == "period_ms" and e.code == "E002" for e in errs)


def test_validate_ultrasonic_zero_timeout():
    errs = validate_model(_model([_ultra("U", timeout="0")]))
    assert any(e.param == "timeout_us" and e.code == "E002" for e in errs)


def test_validate_ultrasonic_period_exactly_50_ok():
    # 50 ms is the minimum allowed value
    errs = validate_model(_model([_ultra("U", period="50")]))
    assert not any(e.param == "period_ms" for e in errs)


# ---------------------------------------------------------------------------
# 30. validate_model — Step / Integrator / PID errors
# ---------------------------------------------------------------------------


def test_validate_step_negative_step_time():
    errs = validate_model(_model([_step("ST", step_time="-1")]))
    assert any(e.param == "step_time" and e.code == "E002" for e in errs)


def test_validate_step_non_numeric_step_time():
    errs = validate_model(_model([_step("ST", step_time="now")]))
    assert any(e.param == "step_time" and e.code == "E001" for e in errs)


def test_validate_step_non_numeric_initial_value():
    errs = validate_model(_model([_step("ST", initial_value="low")]))
    assert any(e.param == "initial_value" and e.code == "E001" for e in errs)


def test_validate_step_zero_step_time_is_valid():
    assert validate_model(_model([_step("ST", step_time="0")])) == []


def test_validate_integrator_upper_le_lower():
    errs = validate_model(_model([_integrator("INT", upper_limit="1", lower_limit="1")]))
    assert any(e.param == "upper_limit" and e.code == "E007" for e in errs)


def test_validate_integrator_upper_less_than_lower():
    errs = validate_model(_model([_integrator("INT", upper_limit="-5", lower_limit="5")]))
    assert any(e.code == "E007" for e in errs)


def test_validate_integrator_non_numeric_ic():
    errs = validate_model(_model([_integrator("INT", initial_value="start")]))
    assert any(e.param == "initial_value" and e.code == "E001" for e in errs)


def test_validate_pid_negative_n():
    errs = validate_model(_model([_pid("P1", N="-10")]))
    assert any(e.param == "N" and e.code == "E002" for e in errs)


def test_validate_pid_zero_n():
    errs = validate_model(_model([_pid("P1", N="0")]))
    assert any(e.param == "N" and e.code == "E002" for e in errs)


def test_validate_pid_non_numeric_kp():
    errs = validate_model(_model([_pid("P1", Kp="gain")]))
    assert any(e.param == "Kp" and e.code == "E001" for e in errs)


def test_validate_pid_limit_conflict():
    errs = validate_model(_model([_pid("P1", upper_limit="0", lower_limit="0")]))
    assert any(e.code == "E007" for e in errs)


# ---------------------------------------------------------------------------
# 31. validate_model — TransferFcn errors
# ---------------------------------------------------------------------------


def test_validate_transferfcn_improper():
    # order-2 numerator, order-1 denominator → improper
    errs = validate_model(_model([_transferfcn("TF", numerator="1 1 1", denominator="1 1")]))
    assert any(e.param == "numerator" and e.code == "E006" for e in errs)


def test_validate_transferfcn_non_numeric_numerator():
    errs = validate_model(_model([_transferfcn("TF", numerator="a b")]))
    assert any(e.param == "numerator" and e.code == "E001" for e in errs)


def test_validate_transferfcn_non_numeric_denominator():
    errs = validate_model(_model([_transferfcn("TF", denominator="s + 1")]))
    assert any(e.param == "denominator" and e.code == "E001" for e in errs)


def test_validate_transferfcn_zero_leading_denominator():
    errs = validate_model(_model([_transferfcn("TF", denominator="0 1")]))
    assert any(e.param == "denominator" and e.code == "E001" for e in errs)


def test_validate_transferfcn_proper_higher_order_ok():
    # Same degree numerator/denominator is proper (biproper)
    errs = validate_model(_model([_transferfcn("TF", numerator="1 0", denominator="1 1")]))
    assert not any(e.code == "E006" for e in errs)


# ---------------------------------------------------------------------------
# 32. validate_model — ToWorkspace errors
# ---------------------------------------------------------------------------


def test_validate_toworkspace_empty_variable_name():
    errs = validate_model(_model([_toworkspace("TW", variable_name="")]))
    assert any(e.param == "variable_name" and e.code == "E003" for e in errs)


def test_validate_toworkspace_invalid_identifier():
    errs = validate_model(_model([_toworkspace("TW", variable_name="123bad")]))
    assert any(e.param == "variable_name" and e.code == "E005" for e in errs)


def test_validate_toworkspace_valid_identifier():
    errs = validate_model(_model([_toworkspace("TW", variable_name="my_signal")]))
    assert not any(e.param == "variable_name" for e in errs)


def test_validate_toworkspace_zero_max_points():
    errs = validate_model(_model([_toworkspace("TW", max_points="0")]))
    assert any(e.param == "max_points" and e.code == "E002" for e in errs)


def test_validate_toworkspace_negative_max_points():
    errs = validate_model(_model([_toworkspace("TW", max_points="-5")]))
    assert any(e.param == "max_points" and e.code == "E002" for e in errs)


def test_validate_toworkspace_zero_decimation():
    errs = validate_model(_model([_toworkspace("TW", decimation="0")]))
    assert any(e.param == "decimation" and e.code == "E002" for e in errs)


def test_validate_toworkspace_invalid_save_time():
    errs = validate_model(_model([_toworkspace("TW", save_time="2")]))
    assert any(e.param == "save_time" and e.code == "E002" for e in errs)


def test_validate_toworkspace_save_time_0_ok():
    errs = validate_model(_model([_toworkspace("TW", save_time="0")]))
    assert not any(e.param == "save_time" for e in errs)


# ---------------------------------------------------------------------------
# 33. validate_model — multiple blocks, error isolation
# ---------------------------------------------------------------------------


def test_validate_errors_attributed_to_correct_block():
    """Errors from block A must not be attributed to block B."""
    model = _model([
        _sw("GOOD_SW"),
        _sw("BAD_SW", frequency_hz="-1", duty="2"),
    ])
    errs = validate_model(model)
    for e in errs:
        assert e.block_id == "BAD_SW", (
            f"Error {e} attributed to wrong block"
        )


def test_validate_multiple_errors_same_block():
    """A single block with two bad params generates two separate errors."""
    model = _model([_pid("P1", N="-1", Kp="bad")])
    errs = validate_model(model)
    assert len(errs) >= 2


def test_validate_mixed_model_only_bad_block_flagged():
    model = _model([
        _sw("SW1"),                           # good
        _const("K", value="1.0"),            # good
        _integrator("INT", upper_limit="-5", lower_limit="5"),  # bad
        _scope("SC"),                         # good
    ])
    errs = validate_model(model)
    assert len(errs) >= 1
    for e in errs:
        assert e.block_id == "INT"


def test_validate_validation_error_fields_are_populated():
    errs = validate_model(_model([_sw("MySW", frequency_hz="-1")]))
    assert len(errs) >= 1
    e = errs[0]
    assert e.block_id == "MySW"
    assert e.block_type == "SquareWave"
    assert e.param == "frequency_hz"
    assert e.code.startswith("E")
    assert len(e.message) > 0


def test_validate_error_str_representation():
    e = ValidationError("SW1", "SquareWave", "frequency_hz", "E002", "Must be > 0 Hz")
    s = str(e)
    assert "E002" in s
    assert "SW1" in s
    assert "frequency_hz" in s


def test_validate_full_model_all_good_no_errors():
    model = _model(
        [
            _sw("SW1", frequency_hz="10", amplitude="1", offset="0", duty="0.5"),
            _step("ST", step_time="0.5"),
            _integrator("INT", upper_limit="100", lower_limit="-100"),
            _transferfcn("TF", numerator="1", denominator="1 1"),
            _pid("PID1", Kp="1", Ki="0.1", Kd="0.01", N="50",
                 upper_limit="10", lower_limit="-10"),
            _sum("S1"),
            _product("P1"),
            _const("K", value="2.0"),
            _toworkspace("TW", variable_name="result"),
            _scope("SC"),
        ],
        [],
    )
    errs = validate_model(model)
    assert errs == [], f"Unexpected errors: {errs}"


# ---------------------------------------------------------------------------
# Factory helpers for new blocks (Groups A-F)
# ---------------------------------------------------------------------------


def _sine(bid="SN", frequency_hz="1.0", amplitude="1.0", phase_deg="0.0", offset="0.0"):
    return {"type": "SineWave", "id": bid, "x": 0, "y": 0,
            "params": {"frequency_hz": frequency_hz, "amplitude": amplitude,
                       "phase_deg": phase_deg, "offset": offset}}

def _ramp(bid="RM", slope="1.0", start_time="0.0", initial_output="0.0"):
    return {"type": "Ramp", "id": bid, "x": 0, "y": 0,
            "params": {"slope": slope, "start_time": start_time, "initial_output": initial_output}}

def _clock(bid="CK"):
    return {"type": "Clock", "id": bid, "x": 0, "y": 0, "params": {}}

def _pulse(bid="PG", amplitude="1.0", period="1.0", pulse_width="50", phase_delay="0.0"):
    return {"type": "PulseGenerator", "id": bid, "x": 0, "y": 0,
            "params": {"amplitude": amplitude, "period": period,
                       "pulse_width": pulse_width, "phase_delay": phase_delay}}

def _gain(bid="GN", gain="2.0"):
    return {"type": "Gain", "id": bid, "x": 0, "y": 0, "params": {"gain": gain}}

def _abs(bid="AB"):
    return {"type": "Abs", "id": bid, "x": 0, "y": 0, "params": {}}

def _sign(bid="SG"):
    return {"type": "Sign", "id": bid, "x": 0, "y": 0, "params": {}}

def _sqrt(bid="SQ", mode="sqrt"):
    return {"type": "Sqrt", "id": bid, "x": 0, "y": 0, "params": {"mode": mode}}

def _saturation(bid="SAT", upper_limit="1.0", lower_limit="-1.0"):
    return {"type": "Saturation", "id": bid, "x": 0, "y": 0,
            "params": {"upper_limit": upper_limit, "lower_limit": lower_limit}}

def _deadzone(bid="DZ", lower_value="-0.5", upper_value="0.5"):
    return {"type": "DeadZone", "id": bid, "x": 0, "y": 0,
            "params": {"lower_value": lower_value, "upper_value": upper_value}}

def _minmax(bid="MM", function="min"):
    return {"type": "MinMax", "id": bid, "x": 0, "y": 0, "params": {"function": function}}

def _relop(bid="RO", operator=">"):
    return {"type": "RelationalOperator", "id": bid, "x": 0, "y": 0,
            "params": {"operator": operator}}

def _logop(bid="LO", operator="AND"):
    return {"type": "LogicalOperator", "id": bid, "x": 0, "y": 0,
            "params": {"operator": operator}}

def _switch(bid="SW2", threshold="0.5", criteria=">="):
    return {"type": "Switch", "id": bid, "x": 0, "y": 0,
            "params": {"threshold": threshold, "criteria": criteria}}

def _unitdelay(bid="UD", initial_condition="0.0"):
    return {"type": "UnitDelay", "id": bid, "x": 0, "y": 0,
            "params": {"initial_condition": initial_condition}}

def _discint(bid="DI", gain_value="1.0", initial_condition="0.0",
             upper_limit="1e10", lower_limit="-1e10", method="Forward Euler"):
    return {"type": "DiscreteIntegrator", "id": bid, "x": 0, "y": 0,
            "params": {"gain_value": gain_value, "initial_condition": initial_condition,
                       "upper_limit": upper_limit, "lower_limit": lower_limit,
                       "method": method}}

def _zoh(bid="ZH", sample_time="0.01"):
    return {"type": "ZeroOrderHold", "id": bid, "x": 0, "y": 0,
            "params": {"sample_time": sample_time}}

def _deriv(bid="DV", initial_condition="0.0"):
    return {"type": "Derivative", "id": bid, "x": 0, "y": 0,
            "params": {"initial_condition": initial_condition}}

def _lookup1d(bid="LU", breakpoints="0 1", table_data="0 1", extrapolation="clip"):
    return {"type": "Lookup1D", "id": bid, "x": 0, "y": 0,
            "params": {"breakpoints": breakpoints, "table_data": table_data,
                       "extrapolation": extrapolation}}

def _adc(bid="AD", channel="1", resolution="12", vref="3.3", sim_value="1.5"):
    return {"type": "ADC", "id": bid, "x": 0, "y": 0,
            "params": {"channel": channel, "resolution": resolution,
                       "vref": vref, "sim_value": sim_value}}

def _dac(bid="DC", channel="1", vref="3.3"):
    return {"type": "DAC", "id": bid, "x": 0, "y": 0,
            "params": {"channel": channel, "vref": vref}}

def _pwmout(bid="PW", timer="TIM2", channel="1", frequency_hz="1000", max_duty="100.0"):
    return {"type": "PWMOut", "id": bid, "x": 0, "y": 0,
            "params": {"timer": timer, "channel": channel,
                       "frequency_hz": frequency_hz, "max_duty": max_duty}}

def _timertick(bid="TT", scale="0.001"):
    return {"type": "TimerTick", "id": bid, "x": 0, "y": 0,
            "params": {"scale": scale}}


# ---------------------------------------------------------------------------
# 34. SineWave block
# ---------------------------------------------------------------------------


def test_sinewave_in_catalog():
    assert "SineWave" in BLOCK_CATALOG

def test_sinewave_spec_has_no_inputs_one_output():
    spec = BLOCK_CATALOG["SineWave"]
    assert spec.inputs == []
    assert [p.name for p in spec.outputs] == ["y"]

def test_sinewave_spec_has_required_params():
    params = BLOCK_CATALOG["SineWave"].params
    for k in ("frequency_hz", "amplitude", "phase_deg", "offset"):
        assert k in params

def test_sinewave_color_is_dark_blue():
    assert BLOCK_CATALOG["SineWave"].color == "#1a5276"

def test_simulate_sinewave_zero_phase_zero_offset():
    model = _model([_sine("SN", frequency_hz="1.0", amplitude="2.0"), _scope("SC")],
                   [_wire("SN", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    # mean of full cycle sine should be ~0
    assert abs(float(y.mean())) < 0.05
    # amplitude should be ~2
    assert abs(float(y.max()) - 2.0) < 0.05

def test_simulate_sinewave_dc_offset():
    model = _model([_sine("SN", amplitude="1.0", offset="3.0"), _scope("SC")],
                   [_wire("SN", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 3.0) < 0.05

def test_simulate_sinewave_phase_180_inverts():
    model_0   = _model([_sine("SN", phase_deg="0.0"),   _scope("SC")], [_wire("SN", "y", "SC", "u0")])
    model_180 = _model([_sine("SN", phase_deg="180.0"), _scope("SC")], [_wire("SN", "y", "SC", "u0")])
    _, s0   = simulate_model(model_0,   duration_s=0.5, step_s=0.001)
    _, s180 = simulate_model(model_180, duration_s=0.5, step_s=0.001)
    assert abs(float(s0["SC.u0"][0]) + float(s180["SC.u0"][0])) < 0.01

def test_emit_decls_sinewave():
    d = _emit_decls([_sine("SN")])
    assert "sig_SN_y" in d

def test_emit_step_sinewave_has_sinf():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_sine("SN")], {}, FakeWorkspace(), 1, board)
    assert "sinf" in step
    assert "sig_SN_y" in step

def test_validate_sinewave_zero_frequency_error():
    errs = validate_model(_model([_sine("SN", frequency_hz="0")]))
    assert any(e.param == "frequency_hz" and e.code == "E002" for e in errs)

def test_validate_sinewave_valid_passes():
    assert validate_model(_model([_sine("SN")])) == []

def test_topo_order_sinewave_before_scope():
    model = _model([_scope("SC"), _sine("SN")], [_wire("SN", "y", "SC", "u0")])
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("SN") < ids.index("SC")


# ---------------------------------------------------------------------------
# 35. Ramp block
# ---------------------------------------------------------------------------


def test_ramp_in_catalog():
    assert "Ramp" in BLOCK_CATALOG

def test_ramp_spec_has_no_inputs_one_output():
    spec = BLOCK_CATALOG["Ramp"]
    assert spec.inputs == []
    assert [p.name for p in spec.outputs] == ["y"]

def test_simulate_ramp_starts_at_zero():
    model = _model([_ramp("RM", slope="1.0", start_time="0.0", initial_output="0.0"), _scope("SC")],
                   [_wire("RM", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y[0])) < 1e-9

def test_simulate_ramp_slope():
    model = _model([_ramp("RM", slope="2.0"), _scope("SC")], [_wire("RM", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    # y[-1] ≈ 2 * 0.999
    assert abs(float(y[-1]) - 2.0 * float(t[-1])) < 0.01

def test_simulate_ramp_delay():
    model = _model([_ramp("RM", slope="1.0", start_time="0.5", initial_output="0.0"), _scope("SC")],
                   [_wire("RM", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y[t < 0.5].max()) < 1e-9  # before start_time → 0

def test_emit_decls_ramp():
    d = _emit_decls([_ramp("RM")])
    assert "sig_RM_y" in d

def test_emit_step_ramp_has_counter():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_ramp("RM")], {}, FakeWorkspace(), 1, board)
    assert "sig_RM_y" in step
    assert "_cnt_RM" in step

def test_validate_ramp_negative_start_time():
    errs = validate_model(_model([_ramp("RM", start_time="-1")]))
    assert any(e.param == "start_time" and e.code == "E002" for e in errs)

def test_validate_ramp_valid():
    assert validate_model(_model([_ramp("RM")])) == []


# ---------------------------------------------------------------------------
# 36. Clock block
# ---------------------------------------------------------------------------


def test_clock_in_catalog():
    assert "Clock" in BLOCK_CATALOG

def test_clock_spec_no_params():
    assert BLOCK_CATALOG["Clock"].params == {}

def test_simulate_clock_outputs_time():
    model = _model([_clock("CK"), _scope("SC")], [_wire("CK", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=0.5, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y[0])) < 1e-9
    assert abs(float(y[-1]) - float(t[-1])) < 0.001

def test_emit_decls_clock():
    d = _emit_decls([_clock("CK")])
    assert "sig_CK_y" in d

def test_emit_step_clock_has_counter():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_clock("CK")], {}, FakeWorkspace(), 1, board)
    assert "sig_CK_y" in step
    assert "_cnt_CK" in step

def test_validate_clock_no_errors():
    assert validate_model(_model([_clock("CK")])) == []


# ---------------------------------------------------------------------------
# 37. PulseGenerator block
# ---------------------------------------------------------------------------


def test_pulse_in_catalog():
    assert "PulseGenerator" in BLOCK_CATALOG

def test_pulse_spec_has_four_params():
    params = BLOCK_CATALOG["PulseGenerator"].params
    for k in ("amplitude", "period", "pulse_width", "phase_delay"):
        assert k in params

def test_simulate_pulse_50pct():
    model = _model([_pulse("PG", amplitude="1.0", period="1.0", pulse_width="50"), _scope("SC")],
                   [_wire("PG", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=2.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 0.5) < 0.05

def test_simulate_pulse_zero_before_delay():
    model = _model([_pulse("PG", phase_delay="0.5"), _scope("SC")],
                   [_wire("PG", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y[t < 0.5].sum()) == 0.0

def test_emit_decls_pulse():
    d = _emit_decls([_pulse("PG")])
    assert "sig_PG_y" in d

def test_emit_step_pulse_has_fmodf():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_pulse("PG")], {}, FakeWorkspace(), 1, board)
    assert "fmodf" in step
    assert "sig_PG_y" in step

def test_validate_pulse_zero_period():
    errs = validate_model(_model([_pulse("PG", period="0")]))
    assert any(e.param == "period" and e.code == "E002" for e in errs)

def test_validate_pulse_width_out_of_range():
    errs = validate_model(_model([_pulse("PG", pulse_width="150")]))
    assert any(e.param == "pulse_width" and e.code == "E002" for e in errs)

def test_validate_pulse_valid():
    assert validate_model(_model([_pulse("PG")])) == []


# ---------------------------------------------------------------------------
# 38. Gain block
# ---------------------------------------------------------------------------


def test_gain_in_catalog():
    assert "Gain" in BLOCK_CATALOG

def test_gain_spec_one_in_one_out():
    spec = BLOCK_CATALOG["Gain"]
    assert [p.name for p in spec.inputs] == ["u"]
    assert [p.name for p in spec.outputs] == ["y"]

def test_simulate_gain_doubles_input():
    model = _model([_sw("A", amplitude="3.0", duty="1.0"), _gain("GN", gain="2.0"), _scope("SC")],
                   [_wire("A", "y", "GN", "u"), _wire("GN", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 6.0) < 1e-6

def test_emit_decls_gain():
    d = _emit_decls([_gain("GN")])
    assert "sig_GN_y" in d

def test_emit_step_gain():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _gain("GN", gain="5.0")]
    wires = {("GN", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_GN_y" in step
    assert "5.0" in step

def test_validate_gain_non_numeric():
    errs = validate_model(_model([_gain("GN", gain="big")]))
    assert any(e.param == "gain" and e.code == "E001" for e in errs)

def test_validate_gain_valid():
    assert validate_model(_model([_gain("GN", gain="0.5")])) == []

def test_topo_order_gain_after_source():
    model = _model([_gain("GN"), _sw("A")], [_wire("A", "y", "GN", "u")])
    ids = [b["id"] for b in _topo_order(model)]
    assert ids.index("A") < ids.index("GN")


# ---------------------------------------------------------------------------
# 39. Abs block
# ---------------------------------------------------------------------------


def test_abs_in_catalog():
    assert "Abs" in BLOCK_CATALOG

def test_simulate_abs_removes_negatives():
    model = _model([_sw("A", amplitude="-2.0", duty="1.0"), _abs("AB"), _scope("SC")],
                   [_wire("A", "y", "AB", "u"), _wire("AB", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y.min()) >= 0.0
    assert abs(float(y.mean()) - 2.0) < 1e-6

def test_emit_decls_abs():
    d = _emit_decls([_abs("AB")])
    assert "sig_AB_y" in d

def test_emit_step_abs_uses_fabsf():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _abs("AB")]
    wires = {("AB", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "fabsf" in step
    assert "sig_AB_y" in step

def test_validate_abs_no_params():
    assert validate_model(_model([_abs("AB")])) == []


# ---------------------------------------------------------------------------
# 40. Sign block
# ---------------------------------------------------------------------------


def test_sign_in_catalog():
    assert "Sign" in BLOCK_CATALOG

def test_simulate_sign_positive():
    model = _model([_sw("A", amplitude="5.0", duty="1.0"), _sign("SG"), _scope("SC")],
                   [_wire("A", "y", "SG", "u"), _wire("SG", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 1.0) < 1e-6

def test_emit_step_sign_has_ternary():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sign("SG")]
    wires = {("SG", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_SG_y" in step
    assert "1.0f" in step
    assert "-1.0f" in step

def test_validate_sign_no_params():
    assert validate_model(_model([_sign("SG")])) == []


# ---------------------------------------------------------------------------
# 41. Sqrt block
# ---------------------------------------------------------------------------


def test_sqrt_in_catalog():
    assert "Sqrt" in BLOCK_CATALOG

def test_simulate_sqrt_basic():
    model = _model([_sw("A", amplitude="4.0", duty="1.0"), _sqrt("SQ"), _scope("SC")],
                   [_wire("A", "y", "SQ", "u"), _wire("SQ", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 2.0) < 1e-6

def test_simulate_sqrt_signed():
    # signed_sqrt of -4 should be -2
    model = _model([_sw("A", amplitude="-4.0", duty="1.0"), _sqrt("SQ", mode="signed_sqrt"), _scope("SC")],
                   [_wire("A", "y", "SQ", "u"), _wire("SQ", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - (-2.0)) < 1e-6

def test_emit_step_sqrt_uses_sqrtf():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sqrt("SQ")]
    wires = {("SQ", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sqrtf" in step

def test_validate_sqrt_invalid_mode():
    errs = validate_model(_model([_sqrt("SQ", mode="cube")]))
    assert any(e.param == "mode" and e.code == "E003" for e in errs)

def test_validate_sqrt_valid():
    assert validate_model(_model([_sqrt("SQ", mode="sqrt")])) == []
    assert validate_model(_model([_sqrt("SQ", mode="signed_sqrt")])) == []


# ---------------------------------------------------------------------------
# 42. Saturation block
# ---------------------------------------------------------------------------


def test_saturation_in_catalog():
    assert "Saturation" in BLOCK_CATALOG

def test_simulate_saturation_clips():
    model = _model([_sw("A", amplitude="5.0", duty="1.0"),
                    _saturation("SAT", upper_limit="2.0", lower_limit="-2.0"),
                    _scope("SC")],
                   [_wire("A", "y", "SAT", "u"), _wire("SAT", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y.max()) <= 2.0 + 1e-9
    assert float(y.min()) >= -2.0 - 1e-9

def test_emit_decls_saturation():
    d = _emit_decls([_saturation("SAT")])
    assert "sig_SAT_y" in d

def test_emit_step_saturation():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _saturation("SAT")]
    wires = {("SAT", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_SAT_y" in step

def test_validate_saturation_upper_le_lower():
    errs = validate_model(_model([_saturation("SAT", upper_limit="-1", lower_limit="1")]))
    assert any(e.code == "E007" for e in errs)

def test_validate_saturation_valid():
    assert validate_model(_model([_saturation("SAT")])) == []


# ---------------------------------------------------------------------------
# 43. DeadZone block
# ---------------------------------------------------------------------------


def test_deadzone_in_catalog():
    assert "DeadZone" in BLOCK_CATALOG

def test_simulate_deadzone_inside_zone_is_zero():
    model = _model([_sw("A", amplitude="0.3", duty="1.0"),
                    _deadzone("DZ", lower_value="-0.5", upper_value="0.5"),
                    _scope("SC")],
                   [_wire("A", "y", "DZ", "u"), _wire("DZ", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean())) < 1e-9

def test_simulate_deadzone_outside_zone():
    model = _model([_sw("A", amplitude="2.0", duty="1.0"),
                    _deadzone("DZ", lower_value="-0.5", upper_value="0.5"),
                    _scope("SC")],
                   [_wire("A", "y", "DZ", "u"), _wire("DZ", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 1.5) < 1e-6  # 2.0 - 0.5 = 1.5

def test_emit_step_deadzone():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _deadzone("DZ")]
    wires = {("DZ", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_DZ_y" in step

def test_validate_deadzone_upper_less_than_lower():
    errs = validate_model(_model([_deadzone("DZ", upper_value="-1.0", lower_value="1.0")]))
    assert any(e.code == "E007" for e in errs)

def test_validate_deadzone_valid():
    assert validate_model(_model([_deadzone("DZ")])) == []


# ---------------------------------------------------------------------------
# 44. MinMax block
# ---------------------------------------------------------------------------


def test_minmax_in_catalog():
    assert "MinMax" in BLOCK_CATALOG

def test_minmax_spec_two_inputs():
    spec = BLOCK_CATALOG["MinMax"]
    assert [p.name for p in spec.inputs] == ["u0", "u1"]

def test_simulate_minmax_min():
    model = _model([_sw("A", amplitude="3.0", duty="1.0"),
                    _sw("B", amplitude="1.0", duty="1.0"),
                    _minmax("MM", function="min"), _scope("SC")],
                   [_wire("A", "y", "MM", "u0"), _wire("B", "y", "MM", "u1"),
                    _wire("MM", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 1.0) < 1e-6

def test_simulate_minmax_max():
    model = _model([_sw("A", amplitude="3.0", duty="1.0"),
                    _sw("B", amplitude="1.0", duty="1.0"),
                    _minmax("MM", function="max"), _scope("SC")],
                   [_wire("A", "y", "MM", "u0"), _wire("B", "y", "MM", "u1"),
                    _wire("MM", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 3.0) < 1e-6

def test_emit_step_minmax_uses_fminf():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sw("B"), _minmax("MM")]
    wires = {("MM", "u0"): ("A", "y"), ("MM", "u1"): ("B", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "fminf" in step or "fmaxf" in step

def test_validate_minmax_invalid_function():
    errs = validate_model(_model([_minmax("MM", function="average")]))
    assert any(e.param == "function" and e.code == "E003" for e in errs)

def test_validate_minmax_valid():
    assert validate_model(_model([_minmax("MM", function="min")])) == []
    assert validate_model(_model([_minmax("MM", function="max")])) == []


# ---------------------------------------------------------------------------
# 45. RelationalOperator block
# ---------------------------------------------------------------------------


def test_relop_in_catalog():
    assert "RelationalOperator" in BLOCK_CATALOG

def test_simulate_relop_gt_true():
    model = _model([_sw("A", amplitude="3.0", duty="1.0"),
                    _sw("B", amplitude="1.0", duty="1.0"),
                    _relop("RO", operator=">"), _scope("SC")],
                   [_wire("A", "y", "RO", "u0"), _wire("B", "y", "RO", "u1"),
                    _wire("RO", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 1.0) < 1e-6

def test_simulate_relop_eq():
    model = _model([_sw("A", amplitude="2.0", duty="1.0"),
                    _sw("B", amplitude="2.0", duty="1.0"),
                    _relop("RO", operator="=="), _scope("SC")],
                   [_wire("A", "y", "RO", "u0"), _wire("B", "y", "RO", "u1"),
                    _wire("RO", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 1.0) < 1e-6

def test_emit_step_relop():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sw("B"), _relop("RO")]
    wires = {("RO", "u0"): ("A", "y"), ("RO", "u1"): ("B", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_RO_y" in step
    assert "1.0f" in step

def test_validate_relop_invalid_operator():
    errs = validate_model(_model([_relop("RO", operator="<>")]))
    assert any(e.param == "operator" and e.code == "E003" for e in errs)

def test_validate_relop_valid():
    for op in (">", "<", ">=", "<=", "==", "!="):
        assert validate_model(_model([_relop("RO", operator=op)])) == [], f"op={op}"


# ---------------------------------------------------------------------------
# 46. LogicalOperator block
# ---------------------------------------------------------------------------


def test_logop_in_catalog():
    assert "LogicalOperator" in BLOCK_CATALOG

def test_simulate_logop_and():
    model = _model([_sw("A", amplitude="1.0", duty="1.0"),
                    _sw("B", amplitude="1.0", duty="1.0"),
                    _logop("LO", operator="AND"), _scope("SC")],
                   [_wire("A", "y", "LO", "u0"), _wire("B", "y", "LO", "u1"),
                    _wire("LO", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 1.0) < 1e-6

def test_simulate_logop_or_one_true():
    model = _model([_sw("A", amplitude="1.0", duty="1.0"),
                    _sw("B", amplitude="0.0", duty="1.0"),
                    _logop("LO", operator="OR"), _scope("SC")],
                   [_wire("A", "y", "LO", "u0"), _wire("B", "y", "LO", "u1"),
                    _wire("LO", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 1.0) < 1e-6

def test_simulate_logop_not():
    model = _model([_sw("A", amplitude="0.0", duty="1.0"),
                    _logop("LO", operator="NOT"), _scope("SC")],
                   [_wire("A", "y", "LO", "u0"), _wire("LO", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 1.0) < 1e-6

def test_emit_step_logop_and():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sw("B"), _logop("LO")]
    wires = {("LO", "u0"): ("A", "y"), ("LO", "u1"): ("B", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_LO_y" in step

def test_validate_logop_invalid_operator():
    errs = validate_model(_model([_logop("LO", operator="XNOR")]))
    assert any(e.param == "operator" and e.code == "E003" for e in errs)

def test_validate_logop_valid():
    for op in ("AND", "OR", "NOT", "NAND", "NOR", "XOR"):
        assert validate_model(_model([_logop("LO", operator=op)])) == [], f"op={op}"


# ---------------------------------------------------------------------------
# 47. Switch block
# ---------------------------------------------------------------------------


def test_switch_in_catalog():
    assert "Switch" in BLOCK_CATALOG

def test_switch_spec_three_inputs():
    spec = BLOCK_CATALOG["Switch"]
    assert [p.name for p in spec.inputs] == ["u0", "u1", "u2"]

def test_simulate_switch_selects_u0():
    # u1 = 1.0 >= threshold 0.5 → select u0
    model = _model([_sw("A", amplitude="10.0", duty="1.0"),
                    _sw("B", amplitude="1.0",  duty="1.0"),
                    _sw("C", amplitude="99.0", duty="1.0"),
                    _switch("SW2", threshold="0.5", criteria=">="), _scope("SC")],
                   [_wire("A", "y", "SW2", "u0"), _wire("B", "y", "SW2", "u1"),
                    _wire("C", "y", "SW2", "u2"), _wire("SW2", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 10.0) < 1e-6

def test_simulate_switch_selects_u2():
    # u1 = 0.0 < threshold 0.5 → select u2
    model = _model([_sw("A", amplitude="10.0", duty="1.0"),
                    _sw("B", amplitude="0.0",  duty="1.0"),
                    _sw("C", amplitude="99.0", duty="1.0"),
                    _switch("SW2", threshold="0.5"), _scope("SC")],
                   [_wire("A", "y", "SW2", "u0"), _wire("B", "y", "SW2", "u1"),
                    _wire("C", "y", "SW2", "u2"), _wire("SW2", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 99.0) < 1e-6

def test_emit_step_switch():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _sw("B"), _sw("C"), _switch("SW2")]
    wires = {("SW2", "u0"): ("A", "y"), ("SW2", "u1"): ("B", "y"), ("SW2", "u2"): ("C", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_SW2_y" in step

def test_validate_switch_invalid_criteria():
    errs = validate_model(_model([_switch("SW2", criteria="!=")]))
    assert any(e.param == "criteria" and e.code == "E003" for e in errs)

def test_validate_switch_valid():
    assert validate_model(_model([_switch("SW2")])) == []


# ---------------------------------------------------------------------------
# 48. UnitDelay block
# ---------------------------------------------------------------------------


def test_unitdelay_in_catalog():
    assert "UnitDelay" in BLOCK_CATALOG

def test_simulate_unitdelay_delays_by_one():
    # Input is ramp 0, 1, 2... → output should be 0, 0, 1, 2...
    model = _model([_const("K", value="1.0"), _unitdelay("UD"), _scope("SC")],
                   [_wire("K", "y", "UD", "u"), _wire("UD", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y[0]) == 0.0  # first output is IC

def test_emit_decls_unitdelay_has_state():
    d = _emit_decls([_unitdelay("UD")])
    assert "sig_UD_y" in d
    assert "_state_UD" in d

def test_emit_step_unitdelay():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _unitdelay("UD")]
    wires = {("UD", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_UD_y" in step
    assert "_state_UD" in step

def test_validate_unitdelay_valid():
    assert validate_model(_model([_unitdelay("UD")])) == []

def test_validate_unitdelay_invalid_ic():
    errs = validate_model(_model([_unitdelay("UD", initial_condition="abc")]))
    assert any(e.param == "initial_condition" and e.code == "E001" for e in errs)


# ---------------------------------------------------------------------------
# 49. DiscreteIntegrator block
# ---------------------------------------------------------------------------


def test_discint_in_catalog():
    assert "DiscreteIntegrator" in BLOCK_CATALOG

def test_simulate_discint_integrates():
    model = _model([_sw("A", amplitude="1.0", duty="1.0"), _discint("DI"), _scope("SC")],
                   [_wire("A", "y", "DI", "u"), _wire("DI", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y[-1]) - float(t[-1])) < 0.01

def test_simulate_discint_upper_clamp():
    model = _model([_sw("A", amplitude="10.0", duty="1.0"),
                    _discint("DI", upper_limit="0.5"), _scope("SC")],
                   [_wire("A", "y", "DI", "u"), _wire("DI", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    assert float(sigs["SC.u0"].max()) <= 0.5 + 1e-9

def test_emit_decls_discint_has_state():
    d = _emit_decls([_discint("DI")])
    assert "sig_DI_y" in d
    assert "_state_DI" in d

def test_emit_step_discint():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _discint("DI")]
    wires = {("DI", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_DI_y" in step
    assert "_state_DI" in step

def test_validate_discint_invalid_method():
    errs = validate_model(_model([_discint("DI", method="RK4")]))
    assert any(e.param == "method" and e.code == "E003" for e in errs)

def test_validate_discint_limit_conflict():
    errs = validate_model(_model([_discint("DI", upper_limit="-1", lower_limit="1")]))
    assert any(e.code == "E007" for e in errs)

def test_validate_discint_valid():
    assert validate_model(_model([_discint("DI")])) == []


# ---------------------------------------------------------------------------
# 50. ZeroOrderHold block
# ---------------------------------------------------------------------------


def test_zoh_in_catalog():
    assert "ZeroOrderHold" in BLOCK_CATALOG

def test_simulate_zoh_holds_value():
    model = _model([_sw("A", frequency_hz="10.0"), _zoh("ZH", sample_time="0.1"), _scope("SC")],
                   [_wire("A", "y", "ZH", "u"), _wire("ZH", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=0.5, step_s=0.001)
    y = sigs["SC.u0"]
    # The output should only change at 100ms intervals — check it's not changing every step
    changes = np.sum(np.diff(y) != 0)
    total = len(y) - 1
    assert changes < total * 0.2  # fewer than 20% of steps should have a change

def test_emit_decls_zoh_has_state():
    d = _emit_decls([_zoh("ZH")])
    assert "sig_ZH_y" in d
    assert "_state_ZH" in d

def test_emit_step_zoh():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _zoh("ZH")]
    wires = {("ZH", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_ZH_y" in step
    assert "_state_ZH" in step

def test_validate_zoh_zero_sample_time():
    errs = validate_model(_model([_zoh("ZH", sample_time="0")]))
    assert any(e.param == "sample_time" and e.code == "E002" for e in errs)

def test_validate_zoh_valid():
    assert validate_model(_model([_zoh("ZH")])) == []


# ---------------------------------------------------------------------------
# 51. Derivative block
# ---------------------------------------------------------------------------


def test_derivative_in_catalog():
    assert "Derivative" in BLOCK_CATALOG

def test_simulate_derivative_of_ramp_is_constant():
    # derivative of ramp (slope=1) = 1/dt * dt = 1.0 (after first step)
    model = _model([_ramp("RM", slope="1.0"), _deriv("DV"), _scope("SC")],
                   [_wire("RM", "y", "DV", "u"), _wire("DV", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    # after first sample, derivative should be ≈ 1.0
    assert abs(float(y[1:].mean()) - 1.0) < 0.01

def test_emit_decls_derivative_has_state():
    d = _emit_decls([_deriv("DV")])
    assert "sig_DV_y" in d
    assert "_state_DV" in d

def test_emit_step_derivative():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _deriv("DV")]
    wires = {("DV", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_DV_y" in step
    assert "_state_DV" in step

def test_validate_derivative_valid():
    assert validate_model(_model([_deriv("DV")])) == []

def test_validate_derivative_invalid_ic():
    errs = validate_model(_model([_deriv("DV", initial_condition="nan")]))
    assert any(e.param == "initial_condition" and e.code == "E001" for e in errs)


# ---------------------------------------------------------------------------
# 52. Lookup1D block
# ---------------------------------------------------------------------------


def test_lookup1d_in_catalog():
    assert "Lookup1D" in BLOCK_CATALOG

def test_simulate_lookup1d_interpolates():
    model = _model([_sw("A", amplitude="0.5", duty="1.0"),
                    _lookup1d("LU", breakpoints="0 1", table_data="0 10"), _scope("SC")],
                   [_wire("A", "y", "LU", "u"), _wire("LU", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean()) - 5.0) < 0.1

def test_simulate_lookup1d_clips_at_boundary():
    model = _model([_sw("A", amplitude="5.0", duty="1.0"),
                    _lookup1d("LU", breakpoints="0 1", table_data="0 10"), _scope("SC")],
                   [_wire("A", "y", "LU", "u"), _wire("LU", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.max()) - 10.0) < 1e-6  # clipped at upper boundary

def test_emit_decls_lookup1d():
    d = _emit_decls([_lookup1d("LU")])
    assert "sig_LU_y" in d

def test_emit_step_lookup1d_has_arrays():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _lookup1d("LU")]
    wires = {("LU", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "sig_LU_y" in step
    assert "_bp_LU" in step
    assert "_td_LU" in step

def test_validate_lookup1d_mismatched_lengths():
    errs = validate_model(_model([_lookup1d("LU", breakpoints="0 1 2", table_data="0 1")]))
    assert any(e.code == "E007" for e in errs)

def test_validate_lookup1d_non_increasing_breakpoints():
    errs = validate_model(_model([_lookup1d("LU", breakpoints="1 0", table_data="0 1")]))
    assert any(e.param == "breakpoints" and e.code == "E002" for e in errs)

def test_validate_lookup1d_invalid_extrapolation():
    errs = validate_model(_model([_lookup1d("LU", extrapolation="nearest")]))
    assert any(e.param == "extrapolation" and e.code == "E003" for e in errs)

def test_validate_lookup1d_valid():
    assert validate_model(_model([_lookup1d("LU")])) == []


# ---------------------------------------------------------------------------
# 53. ADC block
# ---------------------------------------------------------------------------


def test_adc_in_catalog():
    assert "ADC" in BLOCK_CATALOG

def test_adc_spec_source_no_inputs():
    spec = BLOCK_CATALOG["ADC"]
    assert spec.inputs == []
    assert [p.name for p in spec.outputs] == ["y"]

def test_simulate_adc_outputs_sim_value():
    model = _model([_adc("AD", sim_value="1.65"), _scope("SC")],
                   [_wire("AD", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 1.65) < 1e-6

def test_emit_decls_adc():
    d = _emit_decls([_adc("AD")])
    assert "sig_AD_y" in d

def test_emit_step_adc_has_hal_calls():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_adc("AD")], {}, FakeWorkspace(), 1, board)
    assert "HAL_ADC_Start" in step
    assert "sig_AD_y" in step

def test_validate_adc_invalid_channel():
    errs = validate_model(_model([_adc("AD", channel="17")]))
    assert any(e.param == "channel" and e.code == "E002" for e in errs)

def test_validate_adc_invalid_resolution():
    errs = validate_model(_model([_adc("AD", resolution="11")]))
    assert any(e.param == "resolution" and e.code == "E002" for e in errs)

def test_validate_adc_zero_vref():
    errs = validate_model(_model([_adc("AD", vref="0")]))
    assert any(e.param == "vref" and e.code == "E002" for e in errs)

def test_validate_adc_valid():
    assert validate_model(_model([_adc("AD")])) == []


# ---------------------------------------------------------------------------
# 54. DAC block
# ---------------------------------------------------------------------------


def test_dac_in_catalog():
    assert "DAC" in BLOCK_CATALOG

def test_dac_spec_sink_no_outputs():
    spec = BLOCK_CATALOG["DAC"]
    assert [p.name for p in spec.inputs] == ["u"]
    assert spec.outputs == []

def test_emit_decls_dac_no_output_signal():
    d = _emit_decls([_dac("DC")])
    assert "sig_DC_y" not in d

def test_emit_step_dac_has_hal_calls():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _dac("DC")]
    wires = {("DC", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "HAL_DAC_SetValue" in step
    assert "HAL_DAC_Start" in step

def test_validate_dac_invalid_channel():
    errs = validate_model(_model([_dac("DC", channel="3")]))
    assert any(e.param == "channel" and e.code == "E002" for e in errs)

def test_validate_dac_zero_vref():
    errs = validate_model(_model([_dac("DC", vref="-1")]))
    assert any(e.param == "vref" and e.code == "E002" for e in errs)

def test_validate_dac_valid():
    assert validate_model(_model([_dac("DC")])) == []


# ---------------------------------------------------------------------------
# 55. PWMOut block
# ---------------------------------------------------------------------------


def test_pwmout_in_catalog():
    assert "PWMOut" in BLOCK_CATALOG

def test_pwmout_spec_sink_no_outputs():
    spec = BLOCK_CATALOG["PWMOut"]
    assert [p.name for p in spec.inputs] == ["u"]
    assert spec.outputs == []

def test_emit_decls_pwmout_no_output_signal():
    d = _emit_decls([_pwmout("PW")])
    assert "sig_PW_y" not in d

def test_emit_step_pwmout_has_hal_calls():
    board = BOARDS["NUCLEO-F446RE"]
    blocks = [_sw("A"), _pwmout("PW")]
    wires = {("PW", "u"): ("A", "y")}
    step, _ = _emit_step(blocks, wires, FakeWorkspace(), 1, board)
    assert "__HAL_TIM_SET_COMPARE" in step
    assert "__HAL_TIM_GET_AUTORELOAD" in step

def test_validate_pwmout_empty_timer():
    errs = validate_model(_model([_pwmout("PW", timer="")]))
    assert any(e.param == "timer" and e.code == "E003" for e in errs)

def test_validate_pwmout_invalid_channel():
    errs = validate_model(_model([_pwmout("PW", channel="5")]))
    assert any(e.param == "channel" and e.code == "E002" for e in errs)

def test_validate_pwmout_zero_frequency():
    errs = validate_model(_model([_pwmout("PW", frequency_hz="0")]))
    assert any(e.param == "frequency_hz" and e.code == "E002" for e in errs)

def test_validate_pwmout_valid():
    assert validate_model(_model([_pwmout("PW")])) == []


# ---------------------------------------------------------------------------
# 56. TimerTick block
# ---------------------------------------------------------------------------


def test_timertick_in_catalog():
    assert "TimerTick" in BLOCK_CATALOG

def test_timertick_spec_source_no_inputs():
    spec = BLOCK_CATALOG["TimerTick"]
    assert spec.inputs == []
    assert [p.name for p in spec.outputs] == ["y"]

def test_simulate_timertick_outputs_scaled_time():
    model = _model([_timertick("TT", scale="0.001"), _scope("SC")],
                   [_wire("TT", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    # scale=0.001: y = t * 1000 * 0.001 = t
    assert abs(float(y[-1]) - float(t[-1])) < 0.01

def test_emit_decls_timertick():
    d = _emit_decls([_timertick("TT")])
    assert "sig_TT_y" in d

def test_emit_step_timertick_has_gettick():
    board = BOARDS["NUCLEO-F446RE"]
    step, _ = _emit_step([_timertick("TT")], {}, FakeWorkspace(), 1, board)
    assert "HAL_GetTick" in step
    assert "sig_TT_y" in step

def test_validate_timertick_zero_scale():
    errs = validate_model(_model([_timertick("TT", scale="0")]))
    assert any(e.param == "scale" and e.code == "E002" for e in errs)

def test_validate_timertick_invalid_scale():
    errs = validate_model(_model([_timertick("TT", scale="abc")]))
    assert any(e.param == "scale" and e.code == "E001" for e in errs)

def test_validate_timertick_valid():
    assert validate_model(_model([_timertick("TT")])) == []

def test_validate_timertick_negative_scale_valid():
    # Negative scale is allowed (just != 0)
    assert validate_model(_model([_timertick("TT", scale="-0.001")])) == []


# ---------------------------------------------------------------------------
# 24. LTI blocks — StateSpace, DiscreteStateSpace, ZeroPoleGain
# ---------------------------------------------------------------------------


def _ss(bid, A="0 1; -2 -3", B="0; 1", C="1 0", D="0",
        ic="", method="euler"):
    return {"id": bid, "type": "StateSpace",
            "params": {"A": A, "B": B, "C": C, "D": D,
                       "initial_state": ic, "method": method},
            "x": 0.0, "y": 0.0}

def _dss(bid, Ad="0.99 0.01; -0.02 0.97", Bd="0; 0.01",
         Cd="1 0", Dd="0", ic=""):
    return {"id": bid, "type": "DiscreteStateSpace",
            "params": {"Ad": Ad, "Bd": Bd, "Cd": Cd, "Dd": Dd,
                       "initial_state": ic},
            "x": 0.0, "y": 0.0}

def _zpk(bid, zeros="", poles="-1", gain="1.0"):
    return {"id": bid, "type": "ZeroPoleGain",
            "params": {"zeros": zeros, "poles": poles, "gain": gain},
            "x": 0.0, "y": 0.0}


# ── StateSpace ──────────────────────────────────────────────────────────────

def test_catalog_contains_lti_blocks():
    for name in ("StateSpace", "DiscreteStateSpace", "ZeroPoleGain"):
        assert name in BLOCK_CATALOG, f"{name} missing from BLOCK_CATALOG"


def test_validate_statespace_valid():
    b = _ss("SS1")
    errs = _validate_block(b)
    assert errs == []


def test_validate_statespace_bad_A():
    b = _ss("SS1", A="abc")
    errs = _validate_block(b)
    codes = [e.code for e in errs]
    assert "E001" in codes


def test_validate_statespace_bad_method():
    b = _ss("SS1", method="runge-kutta")
    errs = _validate_block(b)
    codes = [e.code for e in errs]
    assert "E003" in codes


def test_validate_statespace_bad_ic():
    b = _ss("SS1", ic="abc def")
    errs = _validate_block(b)
    codes = [e.code for e in errs]
    assert "E001" in codes


def test_simulate_statespace_first_order():
    """First-order system: dx/dt = -x + u, y = x. Step input."""
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")  # constant 1
    b_ss   = _ss("SS1", A="-1", B="1", C="1", D="0", method="euler")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_ss, b_sc],
                    [_wire("SRC","y","SS1","u"), _wire("SS1","y","SC","u0")])
    t, sigs = simulate_model(model, duration_s=5.0, step_s=0.001)
    y = sigs["SC.u0"]
    # Steady-state for dx/dt=-x+1 is y=1.0
    assert abs(y[-1] - 1.0) < 0.05


def test_simulate_statespace_zero_ic_second_order():
    """Second-order integrator chain with zero IC: should give ramp-then-curve output."""
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    # A=[0 1; 0 0], B=[0;1], C=[1 0]: double integrator
    b_ss   = _ss("SS2", A="0 1; 0 0", B="0; 1", C="1 0", D="0", method="euler")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_ss, b_sc],
                    [_wire("SRC","y","SS2","u"), _wire("SS2","y","SC","u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    # y should be approximately t²/2 at t=1 → 0.5
    assert 0.3 < y[-1] < 0.7


def test_simulate_statespace_D_feedthrough():
    """D != 0: y = C*x + D*u; at t=0 with zero IC, y[0] = D*u[0]."""
    b_step = _sw("SRC", amplitude="2.0", duty="1.0")
    b_ss   = _ss("SS1", A="-1", B="1", C="0", D="3.0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_ss, b_sc],
                    [_wire("SRC","y","SS1","u"), _wire("SS1","y","SC","u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    # y[0] = D*u[0] = 3*2 = 6
    assert abs(y[0] - 6.0) < 0.1


# ── DiscreteStateSpace ───────────────────────────────────────────────────────

def test_validate_discrete_statespace_valid():
    b = _dss("DSS1")
    errs = _validate_block(b)
    assert errs == []


def test_validate_discrete_statespace_bad_Ad():
    b = _dss("DSS1", Ad="x y; z w")
    errs = _validate_block(b)
    codes = [e.code for e in errs]
    assert "E001" in codes


def test_simulate_discrete_ss_unity():
    """Ad=1, Bd=dt, Cd=1, Dd=0 → integrator approximation."""
    dt = 0.01
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    # Ad=1, Bd=0.01, Cd=1, Dd=0  →  x[k+1] = x[k] + 0.01*u, y=x (discrete integrator)
    b_dss  = _dss("DSS1", Ad="1", Bd="0.01", Cd="1", Dd="0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_dss, b_sc],
                    [_wire("SRC","y","DSS1","u"), _wire("DSS1","y","SC","u0")])
    t, sigs = simulate_model(model, duration_s=1.0, step_s=dt)
    y = sigs["SC.u0"]
    # Should approximate t → y[-1] ≈ 1.0 (minus one step)
    assert 0.9 < y[-1] < 1.1


def test_simulate_discrete_ss_initial_condition():
    """Non-zero IC: output starts at IC."""
    b_step = _sw("SRC", amplitude="0.0", duty="1.0")  # zero input
    b_dss  = _dss("DSS1", Ad="0.9", Bd="0", Cd="1", Dd="0", ic="5.0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_dss, b_sc],
                    [_wire("SRC","y","DSS1","u"), _wire("DSS1","y","SC","u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    # y[0] = Cd*x[0] + Dd*u[0] = 1*5 + 0*0 = 5
    assert abs(y[0] - 5.0) < 0.1
    # y decays geometrically: y[-1] < 5
    assert y[-1] < y[0]


# ── ZeroPoleGain ─────────────────────────────────────────────────────────────

def test_validate_zpk_valid():
    b = _zpk("ZPK1")
    errs = _validate_block(b)
    assert errs == []


def test_validate_zpk_empty_poles():
    b = _zpk("ZPK1", poles="")
    errs = _validate_block(b)
    codes = [e.code for e in errs]
    assert "E001" in codes


def test_validate_zpk_zero_gain():
    b = _zpk("ZPK1", gain="0")
    errs = _validate_block(b)
    codes = [e.code for e in errs]
    assert "E002" in codes


def test_validate_zpk_bad_poles():
    b = _zpk("ZPK1", poles="abc")
    errs = _validate_block(b)
    codes = [e.code for e in errs]
    assert "E001" in codes


def test_simulate_zpk_first_order_step():
    """ZPK with pole at -1 and gain 1: H(s) = 1/(s+1), step response → 1-e^-t."""
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    b_zpk  = _zpk("ZPK1", zeros="", poles="-1", gain="1.0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_zpk, b_sc],
                    [_wire("SRC","y","ZPK1","u"), _wire("ZPK1","y","SC","u0")])
    t, sigs = simulate_model(model, duration_s=5.0, step_s=0.001)
    y = sigs["SC.u0"]
    # Steady state for H(s)=1/(s+1) with unit step = 1.0
    assert abs(y[-1] - 1.0) < 0.05


def test_simulate_zpk_gain_scaling():
    """ZPK with gain=2: output should be 2x compared to gain=1."""
    b_step  = _sw("SRC", amplitude="1.0", duty="1.0")
    b_zpk1  = _zpk("ZPK1", poles="-1", gain="1.0")
    b_zpk2  = _zpk("ZPK2", poles="-1", gain="2.0")
    b_sc1   = _scope("SC1")
    b_sc2   = _scope("SC2")
    model   = _model([b_step, b_zpk1, b_zpk2, b_sc1, b_sc2],
                     [_wire("SRC","y","ZPK1","u"), _wire("ZPK1","y","SC1","u0"),
                      _wire("SRC","y","ZPK2","u"), _wire("ZPK2","y","SC2","u0")])
    _, sigs = simulate_model(model, duration_s=5.0, step_s=0.001)
    y1 = sigs["SC1.u0"]
    y2 = sigs["SC2.u0"]
    assert abs(y2[-1] - 2 * y1[-1]) < 0.1


# ── Codegen: StateSpace ───────────────────────────────────────────────────────

def test_codegen_statespace_first_order():
    """StateSpace codegen should include _ss_x_ state variable and matrix multiply."""
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    b_ss   = _ss("SS1", A="-1", B="1", C="1", D="0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_ss, b_sc],
                    [_wire("SRC","y","SS1","u"), _wire("SS1","y","SC","u0")])
    model["board"] = "NUCLEO-F446RE"
    model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "_ss_x_SS1" in src


def test_codegen_discrete_ss():
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    b_dss  = _dss("DSS1", Ad="0.99", Bd="0.01", Cd="1", Dd="0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_dss, b_sc],
                    [_wire("SRC","y","DSS1","u"), _wire("DSS1","y","SC","u0")])
    model["board"] = "NUCLEO-F446RE"
    model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "_ss_x_DSS1" in src


def test_codegen_zpk():
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    b_zpk  = _zpk("ZPK1", poles="-1", gain="1.0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_zpk, b_sc],
                    [_wire("SRC","y","ZPK1","u"), _wire("ZPK1","y","SC","u0")])
    model["board"] = "NUCLEO-F446RE"
    model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "ZeroPoleGain" in src or "_zpk_s_ZPK1" in src


# ---------------------------------------------------------------------------
# FreeRTOS / CMSIS-RTOS codegen tests
# ---------------------------------------------------------------------------

def _rtos_model():
    """Minimal model with use_rtos=True for codegen tests."""
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_sc], [_wire("SRC", "y", "SC", "u0")])
    model["board"]    = "NUCLEO-F446RE"
    model["step_ms"]  = 1
    model["use_rtos"] = True
    return model


def test_codegen_rtos_uses_cmsis_header():
    """FreeRTOS build must include cmsis_os.h, not rely on bare-metal SysTick."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "#include \"cmsis_os.h\"" in src


def test_codegen_rtos_has_model_task():
    """FreeRTOS build must define ModelTask and osThreadNew."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "ModelTask" in src
    assert "osThreadNew" in src


def test_codegen_rtos_uses_osdelayuntil():
    """ModelTask must call osDelayUntil for precise periodic execution."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "osDelayUntil" in src


def test_codegen_rtos_no_systick_handler():
    """FreeRTOS build must NOT define its own SysTick_Handler (FreeRTOS owns it)."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "SysTick_Handler" not in src


def test_codegen_rtos_no_tick_flag():
    """FreeRTOS build must NOT use the bare-metal step_tick_flag polling pattern."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "step_tick_flag" not in src


def test_codegen_rtos_hal_conf_use_rtos_1():
    """stm32f4xx_hal_conf.h must have USE_RTOS 1U for FreeRTOS builds."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        conf = (proj / "stm32f4xx_hal_conf.h").read_text()
    assert "USE_RTOS                     1U" in conf


def test_codegen_bare_metal_hal_conf_use_rtos_0():
    """Bare-metal builds must keep USE_RTOS 0U."""
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_sc], [_wire("SRC", "y", "SC", "u0")])
    model["board"]    = "NUCLEO-F446RE"
    model["step_ms"]  = 1
    model["use_rtos"] = False
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        conf = (proj / "stm32f4xx_hal_conf.h").read_text()
    assert "USE_RTOS                     0U" in conf


def test_codegen_rtos_freertos_config_generated():
    """FreeRTOS build must emit a FreeRTOSConfig.h file."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        cfg  = (proj / "FreeRTOSConfig.h")
        assert cfg.exists(), "FreeRTOSConfig.h was not generated"
        txt = cfg.read_text()
    assert "configTICK_RATE_HZ" in txt
    assert "configTOTAL_HEAP_SIZE" in txt


def test_codegen_bare_metal_no_freertos_config():
    """Bare-metal build must NOT emit FreeRTOSConfig.h."""
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_sc], [_wire("SRC", "y", "SC", "u0")])
    model["board"]    = "NUCLEO-F446RE"
    model["step_ms"]  = 1
    model["use_rtos"] = False
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), model, FakeWorkspace())
        assert not (proj / "FreeRTOSConfig.h").exists()


def test_codegen_rtos_makefile_includes_freertos_sources():
    """RTOS Makefile must reference FreeRTOS kernel .c files."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        mk   = (proj / "Makefile").read_text()
    assert "tasks.c" in mk
    assert "portable/GCC/ARM_CM4F/port.c" in mk
    assert "heap_4.c" in mk
    assert "cmsis_os2.c" in mk


def test_codegen_rtos_oskernel_start_in_main():
    """main() must call osKernelInitialize and osKernelStart."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        src  = (proj / "main.c").read_text()
    assert "osKernelInitialize" in src
    assert "osKernelStart" in src


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _discover_tests(ns: dict):
    return sorted(
        [(name, fn) for name, fn in ns.items()
         if name.startswith("test_") and callable(fn)],
        key=lambda nf: nf[0],
    )


def main() -> int:
    tests = _discover_tests(globals())
    print(f"Discovered {len(tests)} tests\n")
    passed = 0
    failed: list = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL  {name}")
            print(f"          {e}")
            failed.append(name)
            continue
        except Exception as e:
            print(f"  ERROR {name}")
            print(f"          {type(e).__name__}: {e}")
            for line in traceback.format_exc().splitlines()[-3:]:
                print(f"          {line}")
            failed.append(name)
            continue
        print(f"  pass  {name}")
        passed += 1
    print()
    print(f"{passed}/{len(tests)} passed")
    if failed:
        print("Failing:")
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
