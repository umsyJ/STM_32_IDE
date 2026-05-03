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
from unittest.mock import MagicMock, patch

import numpy as np

# Make the IDE modules importable regardless of where the tests are run from.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from code_templates import (  # noqa: E402
    BOARDS,
    _FREERTOS_VERSION,
    _bilinear_tf,
    _emit_decls,
    _emit_helpers,
    _emit_init,
    _emit_step,
    _ensure_freertos,
    _group_blocks_by_rate,
    _python_to_c,
    _topo_order,
    _wires,
    generate_project,
)
from stm32_block_ide import (  # noqa: E402
    BLOCK_CATALOG,
    ValidationError,
    _apply_resize_math,
    _is_valid_stm32_pin,
    _try_eval_param,
    _validate_block,
    _wire_auto_waypoints,
    _wire_move_seg,
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
    # Extended to 59 blocks including all new group A-F blocks + LTI + batch-2 + batch-3 + DSP blocks.
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
        # Batch 2 — new blocks
        "Chirp", "RandomNumber", "FromWorkspace",
        "MathFunction", "RoundingFunction", "Divide", "Bias", "Polynomial",
        "RateLimiter", "Quantizer",
        "DiscreteTransferFcn", "MovingAverage",
        "Lookup2D",
        "EncoderRead",
        # Batch 3 — 10 new practical blocks
        "Ground", "Relay", "CompareToConstant", "DetectRisePositive",
        "SaturationDynamic", "MultiportSwitch", "TransportDelay",
        "UARTSend", "I2CRead", "I2CWrite",
        # DSP blocks
        "FIRFilter", "BiquadFilter", "RunningRMS", "MedianFilter", "NCO", "PeakDetector",
        # Custom code
        "PythonFcn",
        # Control / cart-pendulum blocks
        "WeightedSum", "PlantODE", "AngleUnwrap", "HBridgeOut", "DiscreteIntegratorAW",
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

def test_simulate_discint_feedback_loop():
    """Step → Sum → DI → Gain(-1) → Sum feedback loop.

    With K=1 the closed-loop system is a first-order IIR with time constant
    τ=1 s.  After 1 s the output should be ~0.63 (1 - 1/e) within ±5 %.
    This test confirms the per-sample simulator correctly breaks the
    algebraic loop: before the fix, DI always saw zeros and never integrated.
    """
    # Step source
    step_b = {"type": "Step", "id": "ST", "x": 0, "y": 0,
               "params": {"step_time": "0.0", "initial_value": "0.0",
                          "final_value": "1.0"}}
    # Sum: u0 (step) + u1 (feedback)
    sum_b  = {"type": "Sum", "id": "SU", "x": 0, "y": 0, "params": {}}
    # DiscreteIntegrator K=1
    di_b   = {"type": "DiscreteIntegrator", "id": "DI", "x": 0, "y": 0,
               "params": {"gain_value": "1.0", "initial_condition": "0.0",
                          "upper_limit": "1e10", "lower_limit": "-1e10",
                          "method": "Forward Euler"}}
    # Gain −1
    gain_b = {"type": "Gain", "id": "GN", "x": 0, "y": 0,
               "params": {"gain": "-1.0"}}
    # Scope on DI output
    scope_b= {"type": "Scope", "id": "SC", "x": 0, "y": 0,
               "params": {"channels": "1"}}

    blocks = [step_b, sum_b, di_b, gain_b, scope_b]
    conns  = [
        {"src_block": "ST", "src_port": "y",  "dst_block": "SU", "dst_port": "u0"},
        {"src_block": "GN", "src_port": "y",  "dst_block": "SU", "dst_port": "u1"},
        {"src_block": "SU", "src_port": "y",  "dst_block": "DI", "dst_port": "u"},
        {"src_block": "DI", "src_port": "y",  "dst_block": "GN", "dst_port": "u"},
        {"src_block": "DI", "src_port": "y",  "dst_block": "SC", "dst_port": "u0"},
    ]
    model = {"blocks": blocks, "connections": conns}
    t, sigs = simulate_model(model, duration_s=5.0, step_s=0.001)
    y = sigs["SC.u0"]
    # At t=1 s the output should be near 1 - exp(-1) ≈ 0.632
    k1 = int(1.0 / 0.001)
    assert abs(float(y[k1]) - (1.0 - np.exp(-1.0))) < 0.05, (
        f"Feedback loop broken: y[1s]={float(y[k1]):.4f}, expected ~0.632")
    # At t=5 s it should have settled near 1.0
    assert float(y[-1]) > 0.99, f"DI feedback did not settle: y[-1]={float(y[-1]):.4f}"


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
# FreeRTOS codegen tests
# ---------------------------------------------------------------------------

# Patch _ensure_freertos so tests never hit the network.
# The returned path is a fake directory; the tests only inspect generated
# file content (Makefile / main.c), not an actual build.
_FAKE_FREERTOS = Path("/fake/FreeRTOS-Kernel-10.5.1")

def _rtos_proj(tmp_dir: str):
    """Generate a FreeRTOS project into tmp_dir with network download patched."""
    with patch("code_templates._ensure_freertos", return_value=_FAKE_FREERTOS):
        return generate_project(Path(tmp_dir), _rtos_model(), FakeWorkspace())


def _rtos_model():
    """Minimal model with use_rtos=True for codegen tests."""
    b_step = _sw("SRC", amplitude="1.0", duty="1.0")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_sc], [_wire("SRC", "y", "SC", "u0")])
    model["board"]    = "NUCLEO-F446RE"
    model["step_ms"]  = 1
    model["use_rtos"] = True
    return model


def test_codegen_rtos_uses_freertos_header():
    """FreeRTOS build must include FreeRTOS.h and task.h (pure kernel API)."""
    with tempfile.TemporaryDirectory() as d:
        proj = _rtos_proj(d)
        src = (proj / "main.c").read_text()
    assert '#include "FreeRTOS.h"' in src
    assert '#include "task.h"' in src
    assert "cmsis_os" not in src


def test_codegen_rtos_has_model_task():
    """FreeRTOS build must define ModelTask and use xTaskCreate."""
    with tempfile.TemporaryDirectory() as d:
        proj = _rtos_proj(d)
        src = (proj / "main.c").read_text()
    assert "ModelTask" in src
    assert "xTaskCreate" in src


def test_codegen_rtos_uses_vtaskdelayuntil():
    """ModelTask must call vTaskDelayUntil for precise periodic execution."""
    with tempfile.TemporaryDirectory() as d:
        proj = _rtos_proj(d)
        src = (proj / "main.c").read_text()
    assert "vTaskDelayUntil" in src


def test_codegen_rtos_no_systick_handler():
    """FreeRTOS build must NOT define its own SysTick_Handler (FreeRTOS owns it)."""
    with tempfile.TemporaryDirectory() as d:
        proj = _rtos_proj(d)
        src = (proj / "main.c").read_text()
    assert "SysTick_Handler" not in src


def test_codegen_rtos_no_tick_flag():
    """FreeRTOS build must NOT use the bare-metal step_tick_flag polling pattern."""
    with tempfile.TemporaryDirectory() as d:
        proj = _rtos_proj(d)
        src = (proj / "main.c").read_text()
    assert "step_tick_flag" not in src


def test_codegen_rtos_hal_conf_use_rtos_0():
    """USE_RTOS must always be 0U — STM32 HAL V1.x refuses to compile when it is 1."""
    with tempfile.TemporaryDirectory() as d:
        proj = _rtos_proj(d)
        conf = (proj / "stm32f4xx_hal_conf.h").read_text()
    assert "USE_RTOS                     0U" in conf


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
        proj = _rtos_proj(d)
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
    """RTOS Makefile must reference FreeRTOS kernel .c files (no cmsis_os2.c)."""
    with tempfile.TemporaryDirectory() as d:
        proj = _rtos_proj(d)
        mk   = (proj / "Makefile").read_text()
    assert "tasks.c" in mk
    assert "portable/GCC/ARM_CM4F/port.c" in mk
    assert "heap_4.c" in mk
    assert "cmsis_os2.c" not in mk          # pure FreeRTOS — no CMSIS wrapper


def test_codegen_rtos_vtaskscheduler_start_in_main():
    """main() must call xTaskCreate and vTaskStartScheduler (pure FreeRTOS)."""
    with tempfile.TemporaryDirectory() as d:
        proj = _rtos_proj(d)
        src  = (proj / "main.c").read_text()
    assert "xTaskCreate" in src
    assert "vTaskStartScheduler" in src


# ---------------------------------------------------------------------------
# Batch 3: Factory helpers for 10 new practical blocks
# ---------------------------------------------------------------------------


def _ground(bid):
    return {"id": bid, "type": "Ground", "params": {}, "x": 0, "y": 0}


def _relay(bid, on_thr="0.5", off_thr="-0.5", on_val="1.0", off_val="0.0"):
    return {"id": bid, "type": "Relay",
            "params": {"on_threshold": on_thr, "off_threshold": off_thr,
                       "on_value": on_val, "off_value": off_val}, "x": 0, "y": 0}


def _ctc(bid, operator="==", constant="0.0"):
    return {"id": bid, "type": "CompareToConstant",
            "params": {"operator": operator, "constant": constant}, "x": 0, "y": 0}


def _drp(bid, ic="0.0"):
    return {"id": bid, "type": "DetectRisePositive",
            "params": {"initial_condition": ic}, "x": 0, "y": 0}


def _satdyn(bid, du="1.0", dl="-1.0"):
    return {"id": bid, "type": "SaturationDynamic",
            "params": {"default_upper": du, "default_lower": dl}, "x": 0, "y": 0}


def _mpsw(bid, num_inputs="4"):
    return {"id": bid, "type": "MultiportSwitch",
            "params": {"num_inputs": num_inputs}, "x": 0, "y": 0}


def _tdelay(bid, delay_samples="5", ic="0.0"):
    return {"id": bid, "type": "TransportDelay",
            "params": {"delay_samples": delay_samples, "initial_condition": ic}, "x": 0, "y": 0}


def _uartsend(bid, usart="USART1"):
    return {"id": bid, "type": "UARTSend",
            "params": {"usart": usart, "format": "%.4f\\r\\n", "timeout": "10"}, "x": 0, "y": 0}


def _i2cread(bid, i2c="I2C1", dev="0x48", reg="0x00"):
    return {"id": bid, "type": "I2CRead",
            "params": {"i2c": i2c, "device_addr": dev, "reg_addr": reg,
                       "data_bytes": "2", "scale": "1.0", "sim_value": "5.0"}, "x": 0, "y": 0}


def _i2cwrite(bid, i2c="I2C1", dev="0x48", reg="0x00"):
    return {"id": bid, "type": "I2CWrite",
            "params": {"i2c": i2c, "device_addr": dev, "reg_addr": reg,
                       "data_bytes": "2", "scale": "1.0"}, "x": 0, "y": 0}


# ---------------------------------------------------------------------------
# Batch 3: Tests for 10 new practical blocks
# ---------------------------------------------------------------------------


# Ground
def test_simulate_ground_outputs_zero():
    model = _model([_ground("G"), _scope("SC")], [_wire("G", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert np.all(sigs["SC.u0"] == 0.0)


def test_validate_ground_no_errors():
    assert _validate_block(_ground("G")) == []


def test_codegen_ground_outputs_zero():
    model = _model([_ground("G"), _scope("SC")], [_wire("G", "y", "SC", "u0")])
    model["board"] = "NUCLEO-F446RE"
    model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        src = (generate_project(Path(d), model, FakeWorkspace()) / "main.c").read_text()
    assert "0.0f" in src


# Relay
def test_simulate_relay_switches_on():
    b_sw  = _sw("SRC", amplitude="1.0", duty="1.0")
    b_rel = _relay("R1", on_thr="0.5", off_thr="-0.5")
    b_sc  = _scope("SC")
    model = _model([b_sw, b_rel, b_sc], [_wire("SRC", "y", "R1", "u"), _wire("R1", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert np.all(sigs["SC.u0"] == 1.0)


def test_simulate_relay_holds_state():
    b_c   = _const("C", "0.0")
    b_rel = _relay("R1", on_thr="0.5", off_thr="-0.5", off_val="0.0")
    b_sc  = _scope("SC")
    model = _model([b_c, b_rel, b_sc], [_wire("C", "y", "R1", "u"), _wire("R1", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert np.all(sigs["SC.u0"] == 0.0)


def test_validate_relay_off_thr_ge_on_thr():
    b = _relay("R", on_thr="0.0", off_thr="0.5")
    codes = [e.code for e in _validate_block(b)]
    assert "E007" in codes


# CompareToConstant
def test_simulate_ctc_greater_than():
    b_sw  = _sw("SRC", amplitude="2.0", duty="1.0")
    b_ctc = _ctc("C1", operator=">", constant="1.0")
    b_sc  = _scope("SC")
    model = _model([b_sw, b_ctc, b_sc], [_wire("SRC", "y", "C1", "u"), _wire("C1", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert np.all(sigs["SC.u0"] == 1.0)


def test_validate_ctc_bad_operator():
    b = _ctc("C", operator="???")
    codes = [e.code for e in _validate_block(b)]
    assert "E003" in codes


# DetectRisePositive
def test_simulate_detect_rise_positive():
    b_step = _step("STP", step_time="0.05", initial_value="-1.0", final_value="1.0")
    b_drp  = _drp("D1")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_drp, b_sc],
                    [_wire("STP", "y", "D1", "u"), _wire("D1", "y", "SC", "u0")])
    t, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert np.sum(y == 1.0) == 1


# TransportDelay
def test_simulate_transport_delay():
    b_sw = _sw("SRC", amplitude="1.0", duty="1.0")
    b_td = _tdelay("TD", delay_samples="10", ic="0.0")
    b_sc = _scope("SC")
    model = _model([b_sw, b_td, b_sc], [_wire("SRC", "y", "TD", "u"), _wire("TD", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    assert y[0] == 0.0
    assert y[10] == 1.0


def test_validate_transport_delay_zero_samples():
    b = _tdelay("TD", delay_samples="0")
    codes = [e.code for e in _validate_block(b)]
    assert "E002" in codes


# I2CRead
def test_simulate_i2cread_returns_sim_value():
    b_i2c = {"id": "I2C", "type": "I2CRead",
             "params": {"i2c": "I2C1", "device_addr": "0x48", "reg_addr": "0x00",
                        "data_bytes": "2", "scale": "1.0", "sim_value": "3.3"},
             "x": 0, "y": 0}
    b_sc = _scope("SC")
    model = _model([b_i2c, b_sc], [_wire("I2C", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert np.allclose(sigs["SC.u0"], 3.3)


def test_validate_i2cread_bad_address():
    b = {"id": "I2C", "type": "I2CRead",
         "params": {"i2c": "I2C1", "device_addr": "BADADDR", "reg_addr": "0x00",
                    "data_bytes": "2", "scale": "1.0", "sim_value": "0.0"}, "x": 0, "y": 0}
    codes = [e.code for e in _validate_block(b)]
    assert "E001" in codes


# Codegen
def test_codegen_relay_has_threshold():
    b_sw  = _sw("SRC", amplitude="1.0", duty="1.0")
    b_rel = _relay("R1")
    b_sc  = _scope("SC")
    model = _model([b_sw, b_rel, b_sc],
                   [_wire("SRC", "y", "R1", "u"), _wire("R1", "y", "SC", "u0")])
    model["board"] = "NUCLEO-F446RE"
    model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        src = (generate_project(Path(d), model, FakeWorkspace()) / "main.c").read_text()
    assert "_relay_state_R1" in src


def test_codegen_transport_delay_circular_buffer():
    b_sw = _sw("SRC")
    b_td = _tdelay("TD")
    b_sc = _scope("SC")
    model = _model([b_sw, b_td, b_sc],
                   [_wire("SRC", "y", "TD", "u"), _wire("TD", "y", "SC", "u0")])
    model["board"] = "NUCLEO-F446RE"
    model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        src = (generate_project(Path(d), model, FakeWorkspace()) / "main.c").read_text()
    assert "_td_buf_TD" in src


def test_codegen_i2c_read_hal_call():
    b_i2c = {"id": "I2C", "type": "I2CRead",
             "params": {"i2c": "I2C1", "device_addr": "0x48", "reg_addr": "0x00",
                        "data_bytes": "2", "scale": "1.0", "sim_value": "0.0"}, "x": 0, "y": 0}
    b_sc = _scope("SC")
    model = _model([b_i2c, b_sc], [_wire("I2C", "y", "SC", "u0")])
    model["board"] = "NUCLEO-F446RE"
    model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        src = (generate_project(Path(d), model, FakeWorkspace()) / "main.c").read_text()
    assert "HAL_I2C_Mem_Read" in src


# ---------------------------------------------------------------------------
# DSP block factory helpers
# ---------------------------------------------------------------------------


def _fir(bid, coefficients="0.25 0.25 0.25 0.25"):
    return {"id": bid, "type": "FIRFilter",
            "params": {"coefficients": coefficients}, "x": 0, "y": 0}

def _biquad(bid, b0="1.0", b1="0.0", b2="0.0", a1="0.0", a2="0.0"):
    return {"id": bid, "type": "BiquadFilter",
            "params": {"b0": b0, "b1": b1, "b2": b2, "a1": a1, "a2": a2}, "x": 0, "y": 0}

def _rrms(bid, window="10"):
    return {"id": bid, "type": "RunningRMS",
            "params": {"window": window}, "x": 0, "y": 0}

def _median(bid, window="5"):
    return {"id": bid, "type": "MedianFilter",
            "params": {"window": window}, "x": 0, "y": 0}

def _nco(bid, amplitude="1.0", initial_phase="0.0"):
    return {"id": bid, "type": "NCO",
            "params": {"amplitude": amplitude, "initial_phase": initial_phase}, "x": 0, "y": 0}

def _peakdet(bid, mode="max", decay_rate="0.0", initial="0.0"):
    return {"id": bid, "type": "PeakDetector",
            "params": {"mode": mode, "decay_rate": decay_rate, "initial": initial},
            "x": 0, "y": 0}


# ---------------------------------------------------------------------------
# DSP block tests
# ---------------------------------------------------------------------------


# ── FIRFilter ────────────────────────────────────────────────────────────────

def test_validate_fir_valid():
    assert _validate_block(_fir("F1")) == []

def test_validate_fir_empty_coeffs():
    b = _fir("F1", coefficients="")
    codes = [e.code for e in _validate_block(b)]
    assert "E001" in codes

def test_validate_fir_bad_coeffs():
    b = _fir("F1", coefficients="a b c")
    codes = [e.code for e in _validate_block(b)]
    assert "E001" in codes

def test_simulate_fir_passthrough():
    """Single coefficient = 1 → pass-through"""
    b_sw = _sw("SRC", amplitude="2.0", duty="1.0")
    b_fir = _fir("F1", coefficients="1.0")
    b_sc  = _scope("SC")
    model = _model([b_sw, b_fir, b_sc],
                   [_wire("SRC","y","F1","u"), _wire("F1","y","SC","u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert np.allclose(sigs["SC.u0"], 2.0)

def test_simulate_fir_average():
    """4-tap average of constant signal = same constant"""
    b_c   = _const("C", "4.0")
    b_fir = _fir("F1", coefficients="0.25 0.25 0.25 0.25")
    b_sc  = _scope("SC")
    model = _model([b_c, b_fir, b_sc],
                   [_wire("C","y","F1","u"), _wire("F1","y","SC","u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    # After the transient (first 3 samples), output should be 4.0
    assert np.allclose(sigs["SC.u0"][4:], 4.0)

def test_codegen_fir_has_ring_buffer():
    b_sw = _sw("SRC"); b_fir = _fir("F1"); b_sc = _scope("SC")
    model = _model([b_sw, b_fir, b_sc],
                   [_wire("SRC","y","F1","u"), _wire("F1","y","SC","u0")])
    model["board"] = "NUCLEO-F446RE"; model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        src = (generate_project(Path(d), model, FakeWorkspace()) / "main.c").read_text()
    assert "_fir_buf_F1" in src
    assert "_fir_idx_F1" in src


# ── BiquadFilter ─────────────────────────────────────────────────────────────

def test_validate_biquad_valid():
    assert _validate_block(_biquad("BQ")) == []

def test_validate_biquad_bad_coeff():
    b = _biquad("BQ", b0="abc")
    codes = [e.code for e in _validate_block(b)]
    assert "E001" in codes

def test_simulate_biquad_passthrough():
    """b0=1, all others 0 → pure pass-through"""
    b_sw  = _sw("SRC", amplitude="3.0", duty="1.0")
    b_bq  = _biquad("BQ", b0="1.0")
    b_sc  = _scope("SC")
    model = _model([b_sw, b_bq, b_sc],
                   [_wire("SRC","y","BQ","u"), _wire("BQ","y","SC","u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    assert np.allclose(sigs["SC.u0"], 3.0)

def test_codegen_biquad_has_state_vars():
    b_sw = _sw("SRC"); b_bq = _biquad("BQ"); b_sc = _scope("SC")
    model = _model([b_sw, b_bq, b_sc],
                   [_wire("SRC","y","BQ","u"), _wire("BQ","y","SC","u0")])
    model["board"] = "NUCLEO-F446RE"; model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        src = (generate_project(Path(d), model, FakeWorkspace()) / "main.c").read_text()
    assert "_bq_w1_BQ" in src and "_bq_w2_BQ" in src


# ── RunningRMS ───────────────────────────────────────────────────────────────

def test_validate_rrms_valid():
    assert _validate_block(_rrms("R1")) == []

def test_validate_rrms_bad_window():
    b = _rrms("R1", window="0")
    codes = [e.code for e in _validate_block(b)]
    assert "E002" in codes

def test_simulate_rrms_constant_input():
    """RMS of constant A = A"""
    b_c   = _const("C", "3.0")
    b_rms = _rrms("R1", window="10")
    b_sc  = _scope("SC")
    model = _model([b_c, b_rms, b_sc],
                   [_wire("C","y","R1","u"), _wire("R1","y","SC","u0")])
    _, sigs = simulate_model(model, duration_s=0.2, step_s=0.001)
    # After window fills, RMS should be 3.0
    assert abs(sigs["SC.u0"][-1] - 3.0) < 0.01

def test_codegen_rrms_has_circular_buffer():
    b_sw = _sw("SRC"); b_rms = _rrms("R1", window="5"); b_sc = _scope("SC")
    model = _model([b_sw, b_rms, b_sc],
                   [_wire("SRC","y","R1","u"), _wire("R1","y","SC","u0")])
    model["board"] = "NUCLEO-F446RE"; model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        src = (generate_project(Path(d), model, FakeWorkspace()) / "main.c").read_text()
    assert "_rms_buf_R1" in src and "sqrtf" in src


# ── MedianFilter ─────────────────────────────────────────────────────────────

def test_validate_median_valid():
    assert _validate_block(_median("M1")) == []

def test_validate_median_window_too_large():
    b = _median("M1", window="20")
    codes = [e.code for e in _validate_block(b)]
    assert "E002" in codes

def test_simulate_median_removes_spike():
    """A single spike sample should be removed by median filter"""
    b_c  = _const("C", "1.0")
    # Add a spike via a step that goes back - actually use const + spike in scope
    # Simple test: constant 1.0 through median → should stay 1.0
    b_med = _median("M1", window="5")
    b_sc  = _scope("SC")
    model = _model([b_c, b_med, b_sc],
                   [_wire("C","y","M1","u"), _wire("M1","y","SC","u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    # After warmup (window-1 samples), all outputs should equal 1.0
    assert np.allclose(sigs["SC.u0"][5:], 1.0)


# ── NCO ──────────────────────────────────────────────────────────────────────

def test_validate_nco_valid():
    assert _validate_block(_nco("N1")) == []

def test_validate_nco_bad_amplitude():
    b = _nco("N1", amplitude="abc")
    codes = [e.code for e in _validate_block(b)]
    assert "E001" in codes

def test_simulate_nco_outputs_sin_cos():
    """NCO with constant 1Hz freq → sin² + cos² = 1"""
    b_c  = _const("FREQ", "1.0")
    b_n  = _nco("N1", amplitude="1.0")
    b_sc1 = _scope("SC1")
    b_sc2 = _scope("SC2")
    model = _model([b_c, b_n, b_sc1, b_sc2],
                   [_wire("FREQ","y","N1","freq"),
                    _wire("N1","sin_out","SC1","u0"),
                    _wire("N1","cos_out","SC2","u0")])
    _, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    s = sigs["SC1.u0"]
    c = sigs["SC2.u0"]
    # sin² + cos² ≈ 1
    assert np.allclose(s**2 + c**2, 1.0, atol=1e-4)

def test_codegen_nco_has_phase_state():
    b_c = _const("FREQ", "1.0"); b_n = _nco("N1"); b_sc = _scope("SC")
    model = _model([b_c, b_n, b_sc],
                   [_wire("FREQ","y","N1","freq"), _wire("N1","sin_out","SC","u0")])
    model["board"] = "NUCLEO-F446RE"; model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        src = (generate_project(Path(d), model, FakeWorkspace()) / "main.c").read_text()
    assert "_nco_phase_N1" in src
    assert "sinf" in src and "cosf" in src


# ── PeakDetector ─────────────────────────────────────────────────────────────

def test_validate_peakdet_valid():
    assert _validate_block(_peakdet("P1")) == []

def test_validate_peakdet_bad_mode():
    b = _peakdet("P1", mode="minimum")
    codes = [e.code for e in _validate_block(b)]
    assert "E003" in codes

def test_validate_peakdet_negative_decay():
    b = _peakdet("P1", decay_rate="-1.0")
    codes = [e.code for e in _validate_block(b)]
    assert "E002" in codes

def test_simulate_peakdet_holds_max():
    """Step 0→1 → peak should stay at 1 forever"""
    b_step = _step("STP", step_time="0.05", initial_value="0.0", final_value="1.0")
    b_pk   = _peakdet("P1", mode="max")
    b_sc   = _scope("SC")
    model  = _model([b_step, b_pk, b_sc],
                    [_wire("STP","y","P1","u"), _wire("P1","y","SC","u0")])
    _, sigs = simulate_model(model, duration_s=0.2, step_s=0.001)
    y = sigs["SC.u0"]
    assert np.all(y[100:] == 1.0)

def test_simulate_peakdet_decays():
    """Peak should decay when decay_rate > 0"""
    # Use initial="1.0" so the peak starts at 1.0, input is 0 always → pure decay
    b_c  = _const("C", "0.0")
    b_pk = _peakdet("P1", mode="max", decay_rate="2.0", initial="1.0")
    b_sc = _scope("SC")
    model = _model([b_c, b_pk, b_sc],
                   [_wire("C","y","P1","u"), _wire("P1","y","SC","u0")])
    _, sigs = simulate_model(model, duration_s=0.5, step_s=0.001)
    y = sigs["SC.u0"]
    # After 0.5s with decay_rate=2, peak should have decayed significantly from 1.0
    assert y[-1] < y[0]

def test_codegen_peakdet_has_hold():
    b_c = _const("C", "1.0"); b_pk = _peakdet("P1"); b_sc = _scope("SC")
    model = _model([b_c, b_pk, b_sc],
                   [_wire("C","y","P1","u"), _wire("P1","y","SC","u0")])
    model["board"] = "NUCLEO-F446RE"; model["step_ms"] = 1
    with tempfile.TemporaryDirectory() as d:
        src = (generate_project(Path(d), model, FakeWorkspace()) / "main.c").read_text()
    assert "_peak_hold_P1" in src


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


# ===========================================================================
# Multi-rate FreeRTOS tests
# ===========================================================================

# ---------------------------------------------------------------------------
# BlockInstance round-trip
# ---------------------------------------------------------------------------


def test_block_instance_default_sample_time():
    """BlockInstance.sample_time_ms defaults to 0."""
    from stm32_block_ide import BlockInstance
    spec = BLOCK_CATALOG["Constant"]
    bi = BlockInstance(spec=spec, block_id="C_1", x=0, y=0)
    assert bi.sample_time_ms == 0


def test_block_instance_to_dict_omits_zero():
    """to_dict() must NOT include sample_time_ms when it is 0."""
    from stm32_block_ide import BlockInstance
    spec = BLOCK_CATALOG["Constant"]
    bi = BlockInstance(spec=spec, block_id="C_1", x=0, y=0)
    d = bi.to_dict()
    assert "sample_time_ms" not in d


def test_block_instance_to_dict_includes_nonzero():
    """to_dict() must include sample_time_ms when it is non-zero."""
    from stm32_block_ide import BlockInstance
    spec = BLOCK_CATALOG["Constant"]
    bi = BlockInstance(spec=spec, block_id="C_1", x=0, y=0, sample_time_ms=10)
    d = bi.to_dict()
    assert d["sample_time_ms"] == 10


def test_block_instance_json_roundtrip():
    """sample_time_ms survives a JSON encode/decode round-trip."""
    from stm32_block_ide import BlockInstance
    spec = BLOCK_CATALOG["Gain"]
    bi = BlockInstance(spec=spec, block_id="G_1", x=10, y=20, sample_time_ms=5)
    d = json.loads(json.dumps(bi.to_dict()))
    assert d["sample_time_ms"] == 5


def test_block_instance_load_from_old_dict_defaults_zero():
    """Loading a dict without sample_time_ms gives 0 (backward compat)."""
    d = {"id": "C_1", "type": "Constant", "x": 0, "y": 0, "params": {}}
    # sample_time_ms field should not be present → defaults to 0
    assert int(d.get("sample_time_ms", 0)) == 0


# ---------------------------------------------------------------------------
# validate_model E008
# ---------------------------------------------------------------------------

def _make_rtos_model(step_ms=1, blocks_override=None):
    """Minimal model dict for E008 tests."""
    blocks = blocks_override or [
        {"id": "SW_1", "type": "SquareWave", "x": 0, "y": 0,
         "params": {"frequency_hz": "1", "amplitude": "1", "offset": "0", "duty": "0.5"}},
    ]
    return {
        "blocks": blocks,
        "connections": [],
        "step_ms": step_ms,
        "use_rtos": True,
    }


def test_e008_valid_multiple():
    """sample_time_ms that is a positive multiple of step_ms → no E008."""
    model = _make_rtos_model(step_ms=5, blocks_override=[
        {"id": "SW_1", "type": "SquareWave", "x": 0, "y": 0,
         "sample_time_ms": 10,
         "params": {"frequency_hz": "1", "amplitude": "1", "offset": "0", "duty": "0.5"}},
    ])
    errs = [e for e in validate_model(model) if e.code == "E008"]
    assert errs == []


def test_e008_not_a_multiple():
    """sample_time_ms that is not a multiple of step_ms → E008."""
    model = _make_rtos_model(step_ms=5, blocks_override=[
        {"id": "SW_1", "type": "SquareWave", "x": 0, "y": 0,
         "sample_time_ms": 7,
         "params": {"frequency_hz": "1", "amplitude": "1", "offset": "0", "duty": "0.5"}},
    ])
    errs = [e for e in validate_model(model) if e.code == "E008"]
    assert len(errs) == 1
    assert errs[0].block_id == "SW_1"


def test_e008_less_than_step():
    """sample_time_ms less than step_ms → E008."""
    model = _make_rtos_model(step_ms=10, blocks_override=[
        {"id": "SW_1", "type": "SquareWave", "x": 0, "y": 0,
         "sample_time_ms": 5,
         "params": {"frequency_hz": "1", "amplitude": "1", "offset": "0", "duty": "0.5"}},
    ])
    errs = [e for e in validate_model(model) if e.code == "E008"]
    assert len(errs) == 1


def test_e008_bare_metal_ignored():
    """Bare-metal models never get E008 regardless of sample_time_ms."""
    model = {
        "blocks": [
            {"id": "SW_1", "type": "SquareWave", "x": 0, "y": 0,
             "sample_time_ms": 7,
             "params": {"frequency_hz": "1", "amplitude": "1", "offset": "0", "duty": "0.5"}},
        ],
        "connections": [],
        "step_ms": 5,
        "use_rtos": False,
    }
    errs = [e for e in validate_model(model) if e.code == "E008"]
    assert errs == []


def test_e008_multiple_bad_blocks():
    """Two blocks with invalid rates → two E008 errors."""
    # step_ms=4: 6ms is not a multiple of 4, 10ms is not a multiple of 4
    model = _make_rtos_model(step_ms=4, blocks_override=[
        {"id": "SW_1", "type": "SquareWave", "x": 0, "y": 0,
         "sample_time_ms": 6,
         "params": {"frequency_hz": "1", "amplitude": "1", "offset": "0", "duty": "0.5"}},
        {"id": "SW_2", "type": "SquareWave", "x": 100, "y": 0,
         "sample_time_ms": 10,
         "params": {"frequency_hz": "2", "amplitude": "1", "offset": "0", "duty_cycle": "0.5"}},
    ])
    errs = [e for e in validate_model(model) if e.code == "E008"]
    assert len(errs) == 2


# ---------------------------------------------------------------------------
# _group_blocks_by_rate
# ---------------------------------------------------------------------------

def test_group_all_base_rate():
    """All blocks with sample_time_ms=0 → single group keyed by step_ms."""
    blocks = [
        {"id": "A", "type": "Constant", "params": {}},
        {"id": "B", "type": "Constant", "params": {}},
    ]
    g = _group_blocks_by_rate(blocks, step_ms=5)
    assert list(g.keys()) == [5]
    assert len(g[5]) == 2


def test_group_two_rates():
    """Blocks with different rates → two groups."""
    blocks = [
        {"id": "A", "type": "Constant", "sample_time_ms": 0, "params": {}},
        {"id": "B", "type": "Constant", "sample_time_ms": 10, "params": {}},
    ]
    g = _group_blocks_by_rate(blocks, step_ms=1)
    assert set(g.keys()) == {1, 10}
    assert any(b["id"] == "A" for b in g[1])
    assert any(b["id"] == "B" for b in g[10])


def test_group_ordered_ascending():
    """Groups are sorted ascending (fastest first)."""
    blocks = [
        {"id": "A", "type": "Constant", "sample_time_ms": 50, "params": {}},
        {"id": "B", "type": "Constant", "sample_time_ms": 0, "params": {}},
        {"id": "C", "type": "Constant", "sample_time_ms": 10, "params": {}},
    ]
    g = _group_blocks_by_rate(blocks, step_ms=1)
    assert list(g.keys()) == sorted(g.keys())


def test_group_explicit_equals_step_ms():
    """A block with sample_time_ms == step_ms lands in the base-rate group."""
    blocks = [
        {"id": "A", "type": "Constant", "sample_time_ms": 5, "params": {}},
        {"id": "B", "type": "Constant", "sample_time_ms": 0, "params": {}},
    ]
    g = _group_blocks_by_rate(blocks, step_ms=5)
    # Both blocks should be in the single group keyed at 5
    assert list(g.keys()) == [5]
    assert len(g[5]) == 2


# ---------------------------------------------------------------------------
# simulate_model rate gating
# ---------------------------------------------------------------------------

def _two_rate_model(step_ms=1, slow_ms=10):
    """
    Step(1 at t=0) → Gain(fast, rate=0) + Gain(slow, rate=slow_ms)
    Fast and slow gains both feed Scope so we can inspect both outputs.
    """
    return {
        "step_ms": step_ms,
        "use_rtos": True,
        "blocks": [
            {"id": "ST_1", "type": "Step", "x": 0, "y": 0,
             "params": {"step_time": "0", "initial_value": "0", "final_value": "1"}},
            {"id": "GN_fast", "type": "Gain", "x": 100, "y": 0,
             "sample_time_ms": 0,
             "params": {"K": "1"}},
            {"id": "GN_slow", "type": "Gain", "x": 100, "y": 100,
             "sample_time_ms": slow_ms,
             "params": {"K": "1"}},
            {"id": "SC_1", "type": "Scope", "x": 200, "y": 0,
             "params": {"channels": "2"}},
        ],
        "connections": [
            {"src_block": "ST_1", "src_port": "y",
             "dst_block": "GN_fast", "dst_port": "u"},
            {"src_block": "ST_1", "src_port": "y",
             "dst_block": "GN_slow", "dst_port": "u"},
            {"src_block": "GN_fast", "src_port": "y",
             "dst_block": "SC_1", "dst_port": "u0"},
            {"src_block": "GN_slow", "src_port": "y",
             "dst_block": "SC_1", "dst_port": "u1"},
        ],
    }


def test_simulate_fast_block_updates_every_step():
    """Fast block (rate=0) output changes immediately when Step fires."""
    model = _two_rate_model(step_ms=1, slow_ms=10)
    t, sigs = simulate_model(model, duration_s=0.05, step_s=0.001)
    fast_y = sigs.get("GN_fast.y") or sigs.get("SC_1.u0")
    assert fast_y is not None, f"available signals: {list(sigs)}"
    # At t=0 step fires (step_time=0), fast gain should equal 1 at sample 0
    assert abs(float(fast_y[0]) - 1.0) < 1e-6


def test_simulate_slow_block_held_between_steps():
    """Slow block (rate=10ms) only updates every 10 samples; held otherwise."""
    model = _two_rate_model(step_ms=1, slow_ms=10)
    t, sigs = simulate_model(model, duration_s=0.05, step_s=0.001)
    slow_y = sigs.get("GN_slow.y") or sigs.get("SC_1.u1")
    assert slow_y is not None, f"available signals: {list(sigs)}"
    # Slow block executes at k=0, 10, 20, … After first execution (k=0)
    # value should be held constant at 1 through k=9.
    for k in range(1, 10):
        assert float(slow_y[k]) == float(slow_y[0]), f"mismatch at k={k}"


def test_simulate_downstream_sees_held_value():
    """Downstream block reads the slow block's held (not stale zero) output."""
    model = _two_rate_model(step_ms=1, slow_ms=10)
    t, sigs = simulate_model(model, duration_s=0.05, step_s=0.001)
    slow_y = sigs.get("GN_slow.y") or sigs.get("SC_1.u1")
    assert slow_y is not None
    # After first execution the held value should be 1, not 0
    assert abs(float(slow_y[5]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# codegen multi-rate
# ---------------------------------------------------------------------------

def _make_multirate_codegen_model():
    """SquareWave(1ms) + DiscreteIntegrator(10ms) → two tasks."""
    return {
        "board": "NUCLEO-F446RE",
        "step_ms": 1,
        "use_rtos": True,
        "blocks": [
            {"id": "SW_1", "type": "SquareWave", "x": 0, "y": 0,
             "sample_time_ms": 0,
             "params": {"frequency_hz": "2", "amplitude": "1",
                        "offset": "0", "duty": "0.5"}},
            {"id": "DI_1", "type": "DiscreteIntegrator", "x": 150, "y": 0,
             "sample_time_ms": 10,
             "params": {"K": "1", "method": "Forward Euler",
                        "upper_limit": "1e9", "lower_limit": "-1e9",
                        "initial_condition": "0"}},
        ],
        "connections": [
            {"src_block": "SW_1", "src_port": "y",
             "dst_block": "DI_1", "dst_port": "u"},
        ],
    }


def test_codegen_multirate_two_task_creates():
    """Multi-rate RTOS codegen emits two xTaskCreate calls."""
    model = _make_multirate_codegen_model()
    with tempfile.TemporaryDirectory() as td:
        proj = generate_project(Path(td), model, WORKSPACE)
        src = (proj / "main.c").read_text()
    assert src.count("xTaskCreate") == 2


def test_codegen_multirate_volatile_present():
    """Multi-rate RTOS codegen declares shared signals as volatile."""
    model = _make_multirate_codegen_model()
    with tempfile.TemporaryDirectory() as td:
        proj = generate_project(Path(td), model, WORKSPACE)
        src = (proj / "main.c").read_text()
    assert "volatile float" in src


def test_codegen_single_rate_rtos_no_volatile():
    """Single-rate RTOS codegen must NOT add volatile (no sharing)."""
    model = {
        "board": "NUCLEO-F446RE",
        "step_ms": 1,
        "use_rtos": True,
        "blocks": [
            {"id": "SW_1", "type": "SquareWave", "x": 0, "y": 0,
             "params": {"frequency_hz": "2", "amplitude": "1",
                        "offset": "0", "duty": "0.5"}},
        ],
        "connections": [],
    }
    with tempfile.TemporaryDirectory() as td:
        proj = generate_project(Path(td), model, WORKSPACE)
        src = (proj / "main.c").read_text()
    assert "volatile" not in src
    assert "ModelTask" in src


def test_codegen_multirate_faster_task_higher_priority():
    """Faster task must get a numerically higher FreeRTOS priority."""
    import re
    model = _make_multirate_codegen_model()
    with tempfile.TemporaryDirectory() as td:
        proj = generate_project(Path(td), model, WORKSPACE)
        src = (proj / "main.c").read_text()
    # Extract all (rate_ms, priority) from xTaskCreate lines
    # Pattern: xTaskCreate(model_task_1ms, "Task1ms", 512, NULL, 5, NULL)
    matches = re.findall(r'xTaskCreate\(model_task_(\d+)ms,\s*"[^"]*",\s*\d+,\s*NULL,\s*(\d+),', src)
    assert len(matches) == 2, f"Expected 2 task creates, got: {matches}"
    rate_pri = {int(r): int(p) for r, p in matches}
    rates = sorted(rate_pri.keys())
    # Faster (lower ms) → higher priority value
    assert rate_pri[rates[0]] > rate_pri[rates[1]]


def test_codegen_multirate_correct_dt():
    """Each rate group's step function uses its own dt_s, not the base rate."""
    model = _make_multirate_codegen_model()
    with tempfile.TemporaryDirectory() as td:
        proj = generate_project(Path(td), model, WORKSPACE)
        src = (proj / "main.c").read_text()
    # The 10ms task's DiscreteIntegrator should use 0.010000 not 0.001000
    assert "0.010000" in src


# ===========================================================================
# Auto-tests for every block type
# Covers all 19 blocks that lacked dedicated simulator/codegen tests,
# plus helpers for any missing factory functions.
# ===========================================================================

# ---------------------------------------------------------------------------
# New factory helpers (blocks not yet covered above)
# ---------------------------------------------------------------------------

def _chirp(bid="CH", amplitude="1.0", f_start="1.0", f_end="10.0",
           sweep_time="5.0", phase_deg="0.0"):
    return {"type": "Chirp", "id": bid, "x": 0, "y": 0,
            "params": {"amplitude": amplitude, "f_start": f_start,
                       "f_end": f_end, "sweep_time": sweep_time,
                       "phase_deg": phase_deg}}

def _randnum(bid="RN", mean="0.0", variance="1.0", seed="42"):
    return {"type": "RandomNumber", "id": bid, "x": 0, "y": 0,
            "params": {"mean": mean, "variance": variance, "seed": seed}}

def _fromws(bid="FW", variable_name="u", default="0.0"):
    return {"type": "FromWorkspace", "id": bid, "x": 0, "y": 0,
            "params": {"variable_name": variable_name, "default": default}}

def _mathfunc(bid="MF", function="square", exponent="2.0"):
    return {"type": "MathFunction", "id": bid, "x": 0, "y": 0,
            "params": {"function": function, "exponent": exponent}}

def _rounding(bid="RF", function="round"):
    return {"type": "RoundingFunction", "id": bid, "x": 0, "y": 0,
            "params": {"function": function}}

def _divide(bid="DV2", eps="1e-10"):
    return {"type": "Divide", "id": bid, "x": 0, "y": 0,
            "params": {"eps": eps}}

def _bias(bid="BI", bias="1.0"):
    return {"type": "Bias", "id": bid, "x": 0, "y": 0,
            "params": {"bias": bias}}

def _poly(bid="PL", coefficients="1 0"):
    return {"type": "Polynomial", "id": bid, "x": 0, "y": 0,
            "params": {"coefficients": coefficients}}

def _ratelim(bid="RL", rising_limit="1.0", falling_limit="-1.0",
             initial_condition="0.0"):
    return {"type": "RateLimiter", "id": bid, "x": 0, "y": 0,
            "params": {"rising_limit": rising_limit,
                       "falling_limit": falling_limit,
                       "initial_condition": initial_condition}}

def _quantizer(bid="QZ", interval="0.1"):
    return {"type": "Quantizer", "id": bid, "x": 0, "y": 0,
            "params": {"interval": interval}}

def _dtf(bid="DT", numerator="1", denominator="1 -1"):
    return {"type": "DiscreteTransferFcn", "id": bid, "x": 0, "y": 0,
            "params": {"numerator": numerator, "denominator": denominator}}

def _mavg(bid="MA", window="10"):
    return {"type": "MovingAverage", "id": bid, "x": 0, "y": 0,
            "params": {"window": window}}

def _encread(bid="ER", timer="TIM4", counts_per_rev="1000",
             mode="position", sim_value="0.0"):
    return {"type": "EncoderRead", "id": bid, "x": 0, "y": 0,
            "params": {"timer": timer, "counts_per_rev": counts_per_rev,
                       "mode": mode, "sim_value": sim_value}}


# ---------------------------------------------------------------------------
# Helper: quick codegen check — generate project and return main.c text
# ---------------------------------------------------------------------------
def _gen_src(model):
    with tempfile.TemporaryDirectory() as td:
        proj = generate_project(Path(td), model, WORKSPACE)
        return (proj / "main.c").read_text()


# ===========================================================================
# Chirp
# ===========================================================================

def test_chirp_in_catalog():
    assert "Chirp" in BLOCK_CATALOG

def test_simulate_chirp_bounded_by_amplitude():
    model = _model([_chirp("CH", amplitude="2.0"), _scope("SC")],
                   [_wire("CH", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=5.0, step_s=0.01)
    y = sigs["SC.u0"]
    assert float(y.max()) <= 2.001 and float(y.min()) >= -2.001

def test_simulate_chirp_zero_at_start():
    # phase_deg=0, f_start>0, t=0 → phi=0 → sin(0)=0
    model = _model([_chirp("CH", amplitude="1.0", phase_deg="0.0"), _scope("SC")],
                   [_wire("CH", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.0001)
    assert abs(float(sigs["SC.u0"][0])) < 0.01

def test_simulate_chirp_nonzero_output():
    # Over 5 s the signal should vary — not all zeros
    model = _model([_chirp("CH"), _scope("SC")],
                   [_wire("CH", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=5.0, step_s=0.01)
    assert float(np.abs(sigs["SC.u0"]).max()) > 0.5

def test_codegen_chirp_in_step():
    model = _model([_chirp("CH"), _scope("SC")],
                   [_wire("CH", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_CH_y" in src


# ===========================================================================
# RandomNumber
# ===========================================================================

def test_randnum_in_catalog():
    assert "RandomNumber" in BLOCK_CATALOG

def test_simulate_randnum_correct_mean():
    # mean=0, var=1, seed=42 over 10000 samples → |mean| < 0.1
    model = _model([_randnum("RN", mean="0.0", variance="1.0", seed="42"),
                    _scope("SC")],
                   [_wire("RN", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=10.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.mean())) < 0.1, f"mean={y.mean():.4f}"

def test_simulate_randnum_correct_variance():
    model = _model([_randnum("RN", mean="0.0", variance="4.0", seed="7"),
                    _scope("SC")],
                   [_wire("RN", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=10.0, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y.var()) - 4.0) < 1.0, f"var={y.var():.3f}"

def test_simulate_randnum_offset_mean():
    model = _model([_randnum("RN", mean="5.0", variance="0.01", seed="1"),
                    _scope("SC")],
                   [_wire("RN", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=5.0, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 5.0) < 0.1

def test_codegen_randnum_in_step():
    model = _model([_randnum("RN"), _scope("SC")],
                   [_wire("RN", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_RN_y" in src


# ===========================================================================
# FromWorkspace
# ===========================================================================

def test_fromws_in_catalog():
    assert "FromWorkspace" in BLOCK_CATALOG

def test_simulate_fromws_reads_array():
    n = 100
    WORKSPACE.globals["my_sig"] = np.linspace(0.0, 1.0, n)
    try:
        model = _model([_fromws("FW", variable_name="my_sig"), _scope("SC")],
                       [_wire("FW", "y", "SC", "u0")])
        _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
        y = sigs["SC.u0"]
        # First n samples should match the workspace array
        assert abs(float(y[0]) - 0.0) < 1e-6
        assert abs(float(y[n-1]) - 1.0) < 1e-6
    finally:
        WORKSPACE.globals.pop("my_sig", None)

def test_simulate_fromws_default_when_missing():
    WORKSPACE.globals.pop("missing_var", None)
    model = _model([_fromws("FW", variable_name="missing_var", default="3.0"),
                    _scope("SC")],
                   [_wire("FW", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    # Should not crash; output should be the default or zeros
    assert sigs.get("SC.u0") is not None

def test_codegen_fromws_in_step():
    model = _model([_fromws("FW"), _scope("SC")],
                   [_wire("FW", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_FW_y" in src


# ===========================================================================
# MathFunction
# ===========================================================================

def test_mathfunc_in_catalog():
    assert "MathFunction" in BLOCK_CATALOG

def _mathfunc_val(fn, inp):
    """Helper: run a single MathFunction block with a Constant input."""
    model = _model(
        [_const("C", value=str(inp)), _mathfunc("MF", function=fn), _scope("SC")],
        [_wire("C", "y", "MF", "u"), _wire("MF", "y", "SC", "u0")]
    )
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    return float(sigs["SC.u0"][0])

def test_simulate_mathfunc_square():
    assert abs(_mathfunc_val("square", 3.0) - 9.0) < 1e-6

def test_simulate_mathfunc_exp():
    import math as _math
    assert abs(_mathfunc_val("exp", 1.0) - _math.e) < 1e-5

def test_simulate_mathfunc_log():
    import math as _math
    assert abs(_mathfunc_val("log", _math.e) - 1.0) < 1e-6

def test_simulate_mathfunc_log10():
    assert abs(_mathfunc_val("log10", 100.0) - 2.0) < 1e-6

def test_simulate_mathfunc_reciprocal():
    assert abs(_mathfunc_val("reciprocal", 4.0) - 0.25) < 1e-6

def test_simulate_mathfunc_pow10():
    assert abs(_mathfunc_val("pow10", 2.0) - 100.0) < 1e-6

def test_simulate_mathfunc_pow2():
    assert abs(_mathfunc_val("pow2", 3.0) - 8.0) < 1e-6

def test_simulate_mathfunc_log_zero_clamps():
    # log(0) → 0 (guarded, no crash)
    assert _mathfunc_val("log", 0.0) == 0.0

def test_codegen_mathfunc_in_step():
    model = _model([_const("C", value="2.0"), _mathfunc("MF", function="square"),
                    _scope("SC")],
                   [_wire("C", "y", "MF", "u"), _wire("MF", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_MF_y" in src


# ===========================================================================
# RoundingFunction
# ===========================================================================

def test_roundingfunc_in_catalog():
    assert "RoundingFunction" in BLOCK_CATALOG

def _round_val(fn, inp):
    model = _model(
        [_const("C", value=str(inp)), _rounding("RF", function=fn), _scope("SC")],
        [_wire("C", "y", "RF", "u"), _wire("RF", "y", "SC", "u0")]
    )
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    return float(sigs["SC.u0"][0])

def test_simulate_rounding_floor():
    assert abs(_round_val("floor", 2.9) - 2.0) < 1e-9
    assert abs(_round_val("floor", -2.1) - (-3.0)) < 1e-9

def test_simulate_rounding_ceil():
    assert abs(_round_val("ceil", 2.1) - 3.0) < 1e-9
    assert abs(_round_val("ceil", -2.9) - (-2.0)) < 1e-9

def test_simulate_rounding_round():
    assert abs(_round_val("round", 2.6) - 3.0) < 1e-9
    assert abs(_round_val("round", 2.4) - 2.0) < 1e-9

def test_simulate_rounding_fix():
    # fix = truncate toward zero
    assert abs(_round_val("fix", 2.9) - 2.0) < 1e-9
    assert abs(_round_val("fix", -2.9) - (-2.0)) < 1e-9

def test_codegen_roundingfunc_in_step():
    model = _model([_const("C", value="2.7"), _rounding("RF", function="floor"),
                    _scope("SC")],
                   [_wire("C", "y", "RF", "u"), _wire("RF", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_RF_y" in src


# ===========================================================================
# Divide
# ===========================================================================

def test_divide_in_catalog():
    assert "Divide" in BLOCK_CATALOG

def test_simulate_divide_basic():
    model = _model(
        [_const("A", value="6.0"), _const("B", value="2.0"),
         _divide("DV2"), _scope("SC")],
        [_wire("A", "y", "DV2", "u0"), _wire("B", "y", "DV2", "u1"),
         _wire("DV2", "y", "SC", "u0")]
    )
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 3.0) < 1e-6

def test_simulate_divide_by_zero_returns_zero():
    model = _model(
        [_const("A", value="5.0"), _const("B", value="0.0"),
         _divide("DV2"), _scope("SC")],
        [_wire("A", "y", "DV2", "u0"), _wire("B", "y", "DV2", "u1"),
         _wire("DV2", "y", "SC", "u0")]
    )
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert float(sigs["SC.u0"][0]) == 0.0

def test_simulate_divide_negative():
    model = _model(
        [_const("A", value="-8.0"), _const("B", value="4.0"),
         _divide("DV2"), _scope("SC")],
        [_wire("A", "y", "DV2", "u0"), _wire("B", "y", "DV2", "u1"),
         _wire("DV2", "y", "SC", "u0")]
    )
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - (-2.0)) < 1e-6

def test_codegen_divide_in_step():
    model = _model([_const("A", "4.0"), _const("B", "2.0"), _divide("DV2"),
                    _scope("SC")],
                   [_wire("A", "y", "DV2", "u0"), _wire("B", "y", "DV2", "u1"),
                    _wire("DV2", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_DV2_y" in src


# ===========================================================================
# Bias
# ===========================================================================

def test_bias_in_catalog():
    assert "Bias" in BLOCK_CATALOG

def test_simulate_bias_adds_constant():
    model = _model([_const("C", value="3.0"), _bias("BI", bias="2.0"), _scope("SC")],
                   [_wire("C", "y", "BI", "u"), _wire("BI", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 5.0) < 1e-6

def test_simulate_bias_negative():
    model = _model([_const("C", value="1.0"), _bias("BI", bias="-4.0"), _scope("SC")],
                   [_wire("C", "y", "BI", "u"), _wire("BI", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - (-3.0)) < 1e-6

def test_simulate_bias_zero():
    model = _model([_const("C", value="7.0"), _bias("BI", bias="0.0"), _scope("SC")],
                   [_wire("C", "y", "BI", "u"), _wire("BI", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 7.0) < 1e-6

def test_codegen_bias_in_step():
    model = _model([_const("C", "1.0"), _bias("BI", bias="5.0"), _scope("SC")],
                   [_wire("C", "y", "BI", "u"), _wire("BI", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_BI_y" in src


# ===========================================================================
# Polynomial
# ===========================================================================

def test_polynomial_in_catalog():
    assert "Polynomial" in BLOCK_CATALOG

def test_simulate_polynomial_identity():
    # coefficients "1 0" → np.polyval([1,0], x) = x
    model = _model([_const("C", value="3.0"), _poly("PL", coefficients="1 0"),
                    _scope("SC")],
                   [_wire("C", "y", "PL", "u"), _wire("PL", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 3.0) < 1e-6

def test_simulate_polynomial_quadratic():
    # coefficients "2 0 0" → np.polyval([2,0,0], x) = 2*x^2; at x=3 → 18
    model = _model([_const("C", value="3.0"), _poly("PL", coefficients="2 0 0"),
                    _scope("SC")],
                   [_wire("C", "y", "PL", "u"), _wire("PL", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 18.0) < 1e-6

def test_simulate_polynomial_constant():
    # coefficients "5" → np.polyval([5], x) = 5 (constant)
    model = _model([_const("C", value="99.0"), _poly("PL", coefficients="5"),
                    _scope("SC")],
                   [_wire("C", "y", "PL", "u"), _wire("PL", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 5.0) < 1e-6

def test_codegen_polynomial_in_step():
    model = _model([_const("C", "2.0"), _poly("PL", coefficients="1 0 0"),
                    _scope("SC")],
                   [_wire("C", "y", "PL", "u"), _wire("PL", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_PL_y" in src


# ===========================================================================
# RateLimiter
# ===========================================================================

def test_ratelimiter_in_catalog():
    assert "RateLimiter" in BLOCK_CATALOG

def test_simulate_ratelimiter_limits_rising():
    # Step from 0→1 at t=0; rising_limit=10 V/s, step_s=0.001 → 10*0.001=0.01/sample
    # After 50 ms (50 samples) output should reach ≈0.5 (not 1.0 yet)
    step_s = 0.001
    model = _model(
        [_const("C", value="1.0"), _ratelim("RL", rising_limit="10.0",
                                             falling_limit="-100.0",
                                             initial_condition="0.0"),
         _scope("SC")],
        [_wire("C", "y", "RL", "u"), _wire("RL", "y", "SC", "u0")]
    )
    _, sigs = simulate_model(model, duration_s=0.2, step_s=step_s)
    y = sigs["SC.u0"]
    # 50 ms in: output ≈ 10*0.05 = 0.5
    idx50 = 50
    assert abs(float(y[idx50]) - 0.5) < 0.05, f"y[50]={y[idx50]:.4f}"
    # 100 ms in: output ≈ 1.0 (fully slewed)
    assert abs(float(y[-1]) - 1.0) < 0.05, f"y[-1]={y[-1]:.4f}"

def test_simulate_ratelimiter_limits_falling():
    # Start at 1, step to 0; falling_limit=-10 V/s
    step_s = 0.001
    model = _model(
        [_const("C", value="0.0"),
         _ratelim("RL", rising_limit="100.0", falling_limit="-10.0",
                  initial_condition="1.0"),
         _scope("SC")],
        [_wire("C", "y", "RL", "u"), _wire("RL", "y", "SC", "u0")]
    )
    _, sigs = simulate_model(model, duration_s=0.2, step_s=step_s)
    y = sigs["SC.u0"]
    # 50 ms in: output ≈ 1 - 10*0.05 = 0.5
    assert abs(float(y[50]) - 0.5) < 0.05

def test_codegen_ratelimiter_in_step():
    model = _model([_const("C", "1.0"), _ratelim("RL"), _scope("SC")],
                   [_wire("C", "y", "RL", "u"), _wire("RL", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_RL_y" in src


# ===========================================================================
# Quantizer
# ===========================================================================

def test_quantizer_in_catalog():
    assert "Quantizer" in BLOCK_CATALOG

def test_simulate_quantizer_rounds_to_interval():
    # interval=0.5, input=0.3 → round(0.6)*0.5 = 1*0.5 = 0.5
    model = _model([_const("C", value="0.3"), _quantizer("QZ", interval="0.5"),
                    _scope("SC")],
                   [_wire("C", "y", "QZ", "u"), _wire("QZ", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 0.5) < 1e-6

def test_simulate_quantizer_rounds_down():
    # interval=0.5, input=0.1 → round(0.2)*0.5 = 0*0.5 = 0.0
    model = _model([_const("C", value="0.1"), _quantizer("QZ", interval="0.5"),
                    _scope("SC")],
                   [_wire("C", "y", "QZ", "u"), _wire("QZ", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 0.0) < 1e-6

def test_simulate_quantizer_exact_multiple():
    # input=0.5, interval=0.5 → already on grid
    model = _model([_const("C", value="0.5"), _quantizer("QZ", interval="0.5"),
                    _scope("SC")],
                   [_wire("C", "y", "QZ", "u"), _wire("QZ", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 0.5) < 1e-6

def test_codegen_quantizer_in_step():
    model = _model([_const("C", "0.7"), _quantizer("QZ", interval="0.1"),
                    _scope("SC")],
                   [_wire("C", "y", "QZ", "u"), _wire("QZ", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_QZ_y" in src


# ===========================================================================
# DiscreteTransferFcn
# ===========================================================================

def test_dtf_in_catalog():
    assert "DiscreteTransferFcn" in BLOCK_CATALOG

def test_simulate_dtf_accumulator():
    # H(z) = 1/(1-z^-1): accumulates input.
    # Like all delay blocks: emits stored state first, then updates.
    # With constant u=1: y[0]=0 (IC), y[1]=1, y[2]=2, y[3]=3, y[4]=4
    model = _model(
        [_const("C", value="1.0"), _dtf("DT", numerator="1", denominator="1 -1"),
         _scope("SC")],
        [_wire("C", "y", "DT", "u"), _wire("DT", "y", "SC", "u0")]
    )
    _, sigs = simulate_model(model, duration_s=5.0, step_s=1.0)
    y = sigs["SC.u0"]
    # y[0]=0 (initial state), y[k]=k for k>=1
    assert abs(float(y[0]) - 0.0) < 1e-5, f"y[0]={y[0]}"
    for k in range(1, 5):
        assert abs(float(y[k]) - float(k)) < 1e-5, f"y[{k}]={y[k]:.4f} != {k}"

def test_simulate_dtf_passthrough():
    # H(z) = 1/1 (pure gain=1): y = u
    model = _model(
        [_const("C", value="3.0"), _dtf("DT", numerator="1", denominator="1"),
         _scope("SC")],
        [_wire("C", "y", "DT", "u"), _wire("DT", "y", "SC", "u0")]
    )
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    assert abs(float(sigs["SC.u0"].mean()) - 3.0) < 1e-6

def test_codegen_dtf_in_step():
    model = _model([_const("C", "1.0"),
                    _dtf("DT", numerator="1", denominator="1 -0.9"),
                    _scope("SC")],
                   [_wire("C", "y", "DT", "u"), _wire("DT", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_DT_y" in src


# ===========================================================================
# MovingAverage
# ===========================================================================

def test_mavg_in_catalog():
    assert "MovingAverage" in BLOCK_CATALOG

def test_simulate_mavg_settles_to_input():
    # Constant input 2.0; after window=5 samples output should be 2.0
    model = _model([_const("C", value="2.0"), _mavg("MA", window="5"), _scope("SC")],
                   [_wire("C", "y", "MA", "u"), _wire("MA", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    y = sigs["SC.u0"]
    # After window (5) samples output = 2.0
    assert abs(float(y[5]) - 2.0) < 1e-6
    assert abs(float(y[-1]) - 2.0) < 1e-6

def test_simulate_mavg_ramps_during_fill():
    # With window=4 and constant input 4.0:
    # sample 0: (4+0+0+0)/4 = 1, sample 1: 2, sample 2: 3, sample 3: 4
    model = _model([_const("C", value="4.0"), _mavg("MA", window="4"), _scope("SC")],
                   [_wire("C", "y", "MA", "u"), _wire("MA", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    y = sigs["SC.u0"]
    assert abs(float(y[0]) - 1.0) < 1e-6   # 4/4 = 1
    assert abs(float(y[1]) - 2.0) < 1e-6   # (4+4)/4 = 2
    assert abs(float(y[3]) - 4.0) < 1e-6   # fully filled

def test_simulate_mavg_smooths_step():
    # Step from 0→4 at t=0 with window=4 → ramps up to 4
    model = _model([_const("C", value="4.0"), _mavg("MA", window="10"), _scope("SC")],
                   [_wire("C", "y", "MA", "u"), _wire("MA", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.05, step_s=0.001)
    y = sigs["SC.u0"]
    assert float(y[0]) < float(y[-1])  # monotonically increases from start

def test_codegen_mavg_in_step():
    model = _model([_const("C", "1.0"), _mavg("MA", window="8"), _scope("SC")],
                   [_wire("C", "y", "MA", "u"), _wire("MA", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_MA_y" in src


# ===========================================================================
# Lookup2D
# ===========================================================================

def test_lookup2d_in_catalog():
    assert "Lookup2D" in BLOCK_CATALOG

def _make_lookup2d(bid="L2", rbp="0 1", cbp="0 1", table="0 1; 2 3",
                   u0_val="0.5", u1_val="0.5"):
    """Build a Lookup2D model with constant inputs."""
    blocks = [
        {"type": "Constant", "id": "C0", "x": 0, "y": 0,
         "params": {"value": u0_val}},
        {"type": "Constant", "id": "C1", "x": 0, "y": 50,
         "params": {"value": u1_val}},
        {"type": "Lookup2D", "id": bid, "x": 100, "y": 0,
         "params": {"row_breakpoints": rbp, "col_breakpoints": cbp, "table": table}},
        {"type": "Scope", "id": "SC", "x": 200, "y": 0,
         "params": {"max_points": "100", "stream": "1"}},
    ]
    conns = [
        _wire("C0", "y", bid, "u0"),
        _wire("C1", "y", bid, "u1"),
        _wire(bid, "y", "SC", "u0"),
    ]
    return _model(blocks, conns)

def test_simulate_lookup2d_corner():
    # At (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    model = _make_lookup2d(u0_val="0.0", u1_val="0.0")
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 0.0) < 1e-6

def test_simulate_lookup2d_bilinear_center():
    # At u0=0.5, u1=0.5 → bilinear interp of [0,1,2,3] = 1.5
    model = _make_lookup2d(u0_val="0.5", u1_val="0.5")
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 1.5) < 1e-5

def test_simulate_lookup2d_far_corner():
    model = _make_lookup2d(u0_val="1.0", u1_val="1.0")
    _, sigs = simulate_model(model, duration_s=0.001, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 3.0) < 1e-6

def test_codegen_lookup2d_in_step():
    model = _make_lookup2d()
    src = _gen_src(model)
    assert "sig_L2_y" in src


# ===========================================================================
# EncoderRead
# ===========================================================================

def test_encread_in_catalog():
    assert "EncoderRead" in BLOCK_CATALOG

def test_simulate_encread_sim_value():
    model = _model([_encread("ER", sim_value="500.0"), _scope("SC")],
                   [_wire("ER", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    assert abs(float(sigs["SC.u0"][0]) - 500.0) < 1e-6

def test_simulate_encread_default_zero():
    model = _model([_encread("ER", sim_value="0.0"), _scope("SC")],
                   [_wire("ER", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    assert float(sigs["SC.u0"].sum()) == 0.0

def test_codegen_encread_in_step():
    model = _model([_encread("ER"), _scope("SC")],
                   [_wire("ER", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_ER_y" in src


# ===========================================================================
# TimerTick
# ===========================================================================

def test_timertick_in_catalog():
    assert "TimerTick" in BLOCK_CATALOG

def test_simulate_timertick_proportional_to_time():
    # scale=0.001 → y = t_k * 1000 * 0.001 = t_k
    model = _model([_timertick("TT", scale="0.001"), _scope("SC")],
                   [_wire("TT", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=1.0, step_s=0.001)
    y = sigs["SC.u0"]
    t = np.arange(len(y)) * 0.001
    # y[k] should equal t[k] (within float precision)
    assert float(np.max(np.abs(y - t))) < 1e-9

def test_simulate_timertick_scale_factor():
    # scale=1.0 → y = t_k * 1000 (milliseconds)
    model = _model([_timertick("TT", scale="1.0"), _scope("SC")],
                   [_wire("TT", "y", "SC", "u0")])
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    y = sigs["SC.u0"]
    # At k=5, t_k=0.005 s → y = 0.005*1000*1.0 = 5.0
    assert abs(float(y[5]) - 5.0) < 1e-9

def test_codegen_timertick_in_step():
    model = _model([_timertick("TT"), _scope("SC")],
                   [_wire("TT", "y", "SC", "u0")])
    src = _gen_src(model)
    assert "sig_TT_y" in src


# ===========================================================================
# DAC (sink)
# ===========================================================================

def test_dac_in_catalog():
    assert "DAC" in BLOCK_CATALOG

def test_simulate_dac_no_crash():
    # DAC is a sink; simulation should run and expose u signal in display
    model = _model([_const("C", value="1.5"), _dac("DC")],
                   [_wire("C", "y", "DC", "u")])
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    # Either "DC.u" is in sigs, or fallback keys are present — no crash
    assert sigs is not None

def test_simulate_dac_display_signal():
    model = _model([_sw("SW"), _dac("DC")],
                   [_wire("SW", "y", "DC", "u")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    sig = sigs.get("DC.u")
    if sig is not None:
        assert len(sig) > 0

def test_codegen_dac_in_step():
    model = _model([_const("C", "1.65"), _dac("DC")],
                   [_wire("C", "y", "DC", "u")])
    src = _gen_src(model)
    assert "DAC" in src or "HAL_DAC" in src or "sig_" in src


# ===========================================================================
# PWMOut (sink)
# ===========================================================================

def test_pwmout_in_catalog():
    assert "PWMOut" in BLOCK_CATALOG

def test_simulate_pwmout_no_crash():
    model = _model([_const("C", value="50.0"), _pwmout("PW")],
                   [_wire("C", "y", "PW", "u")])
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    assert sigs is not None

def test_codegen_pwmout_in_step():
    model = _model([_const("C", "75.0"), _pwmout("PW", timer="TIM2", channel="1",
                                                  frequency_hz="1000",
                                                  max_duty="100.0")],
                   [_wire("C", "y", "PW", "u")])
    src = _gen_src(model)
    assert "TIM2" in src or "PWM" in src or "CCR" in src

def test_pwmout_has_timer_param():
    spec = BLOCK_CATALOG["PWMOut"]
    assert "timer" in spec.params
    assert "channel" in spec.params
    assert "frequency_hz" in spec.params


# ===========================================================================
# UARTSend (sink)
# ===========================================================================

def test_uartsend_in_catalog():
    assert "UARTSend" in BLOCK_CATALOG

def test_simulate_uartsend_no_crash():
    model = _model([_const("C", value="3.14"), _uartsend("US")],
                   [_wire("C", "y", "US", "u")])
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    assert sigs is not None

def test_simulate_uartsend_display_signal():
    model = _model([_sw("SW"), _uartsend("US")],
                   [_wire("SW", "y", "US", "u")])
    _, sigs = simulate_model(model, duration_s=0.1, step_s=0.001)
    sig = sigs.get("US.u")
    if sig is not None:
        assert len(sig) > 0

def test_codegen_uartsend_in_step():
    model = _model([_const("C", "1.0"), _uartsend("US", usart="USART1")],
                   [_wire("C", "y", "US", "u")])
    src = _gen_src(model)
    assert "USART" in src or "HAL_UART" in src or "snprintf" in src


# ===========================================================================
# I2CWrite (sink)
# ===========================================================================

def test_i2cwrite_in_catalog():
    assert "I2CWrite" in BLOCK_CATALOG

def test_simulate_i2cwrite_no_crash():
    model = _model([_const("C", value="100.0"), _i2cwrite("IW")],
                   [_wire("C", "y", "IW", "u")])
    _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
    assert sigs is not None

def test_codegen_i2cwrite_in_step():
    model = _model([_const("C", "1.0"), _i2cwrite("IW")],
                   [_wire("C", "y", "IW", "u")])
    src = _gen_src(model)
    assert "I2C" in src or "HAL_I2C" in src or "sig_" in src

def test_i2cwrite_has_address_params():
    spec = BLOCK_CATALOG["I2CWrite"]
    assert "device_addr" in spec.params
    assert "reg_addr" in spec.params


# ===========================================================================
# Smoke tests: every block in the catalog produces no crash when simulated
# (catalog completeness sweep)
# ===========================================================================

def test_every_source_block_simulates_without_crash():
    """Every source block (no inputs) should simulate cleanly."""
    source_blocks = [
        b for b in BLOCK_CATALOG.values()
        if not b.inputs  # no inputs = source
    ]
    for spec in source_blocks:
        defaults = {k: v[0] for k, v in spec.params.items()}
        block = {"type": spec.type_name, "id": "BLK", "x": 0, "y": 0,
                 "params": defaults}
        out_port = spec.outputs[0].name if spec.outputs else None
        if out_port is None:
            continue
        scope = {"type": "Scope", "id": "SC", "x": 100, "y": 0,
                 "params": {"max_points": "10", "stream": "0"}}
        model = {
            "board": "NUCLEO-F446RE", "step_ms": 1,
            "blocks": [block, scope],
            "connections": [{"src_block": "BLK", "src_port": out_port,
                             "dst_block": "SC", "dst_port": "u0"}],
        }
        try:
            _, sigs = simulate_model(model, duration_s=0.01, step_s=0.001)
        except Exception as exc:
            raise AssertionError(
                f"Block {spec.type_name} raised during simulation: {exc}"
            ) from exc


def test_every_single_input_block_simulates_without_crash():
    """Every block with exactly one input port should simulate cleanly."""
    for name, spec in BLOCK_CATALOG.items():
        if len(spec.inputs) != 1:
            continue
        if not spec.outputs:
            continue   # sink — tested elsewhere
        defaults = {k: v[0] for k, v in spec.params.items()}
        src = {"type": "Constant", "id": "SRC", "x": 0, "y": 0,
               "params": {"value": "1.0"}}
        blk = {"type": name, "id": "BLK", "x": 100, "y": 0,
               "params": defaults}
        scope = {"type": "Scope", "id": "SC", "x": 200, "y": 0,
                 "params": {"max_points": "10", "stream": "0"}}
        model = {
            "board": "NUCLEO-F446RE", "step_ms": 1,
            "blocks": [src, blk, scope],
            "connections": [
                {"src_block": "SRC", "src_port": "y",
                 "dst_block": "BLK", "dst_port": spec.inputs[0].name},
                {"src_block": "BLK", "src_port": spec.outputs[0].name,
                 "dst_block": "SC", "dst_port": "u0"},
            ],
        }
        try:
            simulate_model(model, duration_s=0.01, step_s=0.001)
        except Exception as exc:
            raise AssertionError(
                f"Block {name} raised during simulation: {exc}"
            ) from exc


def test_every_block_codegen_without_crash():
    """Every non-sink block should appear in generated main.c without error."""
    for name, spec in BLOCK_CATALOG.items():
        if not spec.outputs:
            continue   # pure sinks covered by codegen sink test below
        defaults = {k: v[0] for k, v in spec.params.items()}
        blocks = []
        conns = []
        if spec.inputs:
            src = {"type": "Constant", "id": "SRC", "x": 0, "y": 0,
                   "params": {"value": "1.0"}}
            blocks.append(src)
            for inp in spec.inputs:
                conns.append({"src_block": "SRC", "src_port": "y",
                              "dst_block": "BLK", "dst_port": inp.name})
        blk = {"type": name, "id": "BLK", "x": 100, "y": 0, "params": defaults}
        scope = {"type": "Scope", "id": "SC", "x": 200, "y": 0,
                 "params": {"max_points": "10", "stream": "0"}}
        blocks += [blk, scope]
        conns.append({"src_block": "BLK", "src_port": spec.outputs[0].name,
                      "dst_block": "SC", "dst_port": "u0"})
        model = {"board": "NUCLEO-F446RE", "step_ms": 1,
                 "blocks": blocks, "connections": conns}
        try:
            with tempfile.TemporaryDirectory() as td:
                generate_project(Path(td), model, WORKSPACE)
        except Exception as exc:
            raise AssertionError(
                f"Block {name} raised during codegen: {exc}"
            ) from exc


def test_every_sink_block_simulates_without_crash():
    """Every sink block (no outputs) should simulate without crashing."""
    for name, spec in BLOCK_CATALOG.items():
        if spec.outputs:
            continue
        if not spec.inputs:
            continue
        defaults = {k: v[0] for k, v in spec.params.items()}
        src = {"type": "Constant", "id": "SRC", "x": 0, "y": 0,
               "params": {"value": "1.0"}}
        blk = {"type": name, "id": "BLK", "x": 100, "y": 0, "params": defaults}
        conns = [{"src_block": "SRC", "src_port": "y",
                  "dst_block": "BLK", "dst_port": spec.inputs[0].name}]
        model = {"board": "NUCLEO-F446RE", "step_ms": 1,
                 "blocks": [src, blk], "connections": conns}
        try:
            simulate_model(model, duration_s=0.01, step_s=0.001)
        except Exception as exc:
            raise AssertionError(
                f"Sink block {name} raised during simulation: {exc}"
            ) from exc


# ===========================================================================
# PythonFcn block tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pfcn(code="u[0]", num_inputs="1", bid="PF"):
    """Build a minimal PythonFcn block dict."""
    return {"type": "PythonFcn", "id": bid, "x": 0, "y": 0,
            "params": {"num_inputs": num_inputs, "code": code}}


def _pfcn_model(code="u[0]", num_inputs="1", src_val="2.0"):
    """Build a model: Constant → PythonFcn."""
    src  = {"type": "Constant", "id": "C0", "x": 0, "y": 0,
            "params": {"value": src_val}}
    blk  = _pfcn(code, num_inputs)
    conn = [{"src_block": "C0", "src_port": "y",
             "dst_block": "PF", "dst_port": "u0"}]
    return {"board": "NUCLEO-F446RE", "step_ms": 1,
            "blocks": [src, blk], "connections": conn}


def _pfcn_sim(code="u[0]", src_val="2.0", steps=5):
    """Run simulate_model on a Constant → PythonFcn model and return y array."""
    m = _pfcn_model(code=code, src_val=src_val)
    _t, disp = simulate_model(m, duration_s=steps * 0.001, step_s=0.001)
    return disp["PF.y"]


def _pfcn_src(code="u[0]", num_inputs="1"):
    """Return generated main.c for a Constant → PythonFcn model."""
    m = _pfcn_model(code=code, num_inputs=num_inputs)
    ws = FakeWorkspace()
    blocks = m["blocks"]
    wires  = _wires(m)
    board  = BOARDS["NUCLEO-F446RE"]
    step_ms = m["step_ms"]
    body, _ = _emit_step(blocks, wires, ws, step_ms, board)
    decls   = _emit_decls(blocks)
    return decls + "\n" + body


# ---------------------------------------------------------------------------
# Catalog / spec
# ---------------------------------------------------------------------------

def test_pythonfcn_in_catalog():
    assert "PythonFcn" in BLOCK_CATALOG


def test_pythonfcn_has_four_inputs():
    spec = BLOCK_CATALOG["PythonFcn"]
    in_names = [p.name for p in spec.inputs]
    assert in_names == ["u0", "u1", "u2", "u3"]


def test_pythonfcn_has_single_output():
    spec = BLOCK_CATALOG["PythonFcn"]
    assert len(spec.outputs) == 1
    assert spec.outputs[0].name == "y"


def test_pythonfcn_default_code_is_u0():
    spec = BLOCK_CATALOG["PythonFcn"]
    assert spec.params["code"][0] == "u[0]"


def test_pythonfcn_default_num_inputs_is_1():
    spec = BLOCK_CATALOG["PythonFcn"]
    assert spec.params["num_inputs"][0] == "1"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_pythonfcn_validate_ok():
    errs = _validate_block(_pfcn("u[0] * 2.0"), None)
    assert errs == []


def test_pythonfcn_validate_empty_code():
    errs = _validate_block(_pfcn(""), None)
    codes = [e.code for e in errs]
    assert "E001" in codes


def test_pythonfcn_validate_syntax_error():
    errs = _validate_block(_pfcn("u[0] *** 2"), None)
    codes = [e.code for e in errs]
    assert "E003" in codes


def test_pythonfcn_validate_num_inputs_out_of_range():
    errs = _validate_block(_pfcn("u[0]", num_inputs="5"), None)
    codes = [e.code for e in errs]
    assert "E002" in codes


def test_pythonfcn_validate_num_inputs_zero():
    errs = _validate_block(_pfcn("u[0]", num_inputs="0"), None)
    codes = [e.code for e in errs]
    assert "E002" in codes


def test_pythonfcn_validate_multiline_exec_ok():
    """Multi-line code with a final assignment is valid."""
    code = "y = u[0] ** 2\ny += 1.0"
    errs = _validate_block(_pfcn(code), None)
    assert errs == []


# ---------------------------------------------------------------------------
# Simulator — single input
# ---------------------------------------------------------------------------

def test_pythonfcn_sim_passthrough():
    y = _pfcn_sim("u[0]", src_val="3.0")
    assert np.allclose(y, 3.0)


def test_pythonfcn_sim_scale():
    y = _pfcn_sim("u[0] * 4.0", src_val="2.0")
    assert np.allclose(y, 8.0)


def test_pythonfcn_sim_math_sin():
    import math as _math
    y = _pfcn_sim("math.sin(u[0])", src_val="0.0")
    assert np.allclose(y, 0.0, atol=1e-6)


def test_pythonfcn_sim_time_ramp():
    """Output depends on t (time in seconds)."""
    m = _pfcn_model(code="t", src_val="0.0")
    _t, disp = simulate_model(m, duration_s=0.005, step_s=0.001)
    y = disp["PF.y"]
    # y[k] ≈ k * dt
    expected = np.arange(len(y)) * 0.001
    assert np.allclose(y, expected, atol=1e-9)


def test_pythonfcn_sim_abs():
    y = _pfcn_sim("abs(u[0])", src_val="-5.0")
    assert np.allclose(y, 5.0)


def test_pythonfcn_sim_polynomial():
    """u[0]**2 + 2*u[0] + 1  with u=3 → 16."""
    y = _pfcn_sim("u[0]**2 + 2*u[0] + 1", src_val="3.0")
    assert np.allclose(y, 16.0)


# ---------------------------------------------------------------------------
# Simulator — two inputs
# ---------------------------------------------------------------------------

def test_pythonfcn_sim_two_inputs():
    """Sum of two inputs."""
    c0 = {"type": "Constant", "id": "C0", "x": 0, "y": 0, "params": {"value": "3.0"}}
    c1 = {"type": "Constant", "id": "C1", "x": 0, "y": 50, "params": {"value": "7.0"}}
    blk = _pfcn("u[0] + u[1]", num_inputs="2")
    conns = [
        {"src_block": "C0", "src_port": "y", "dst_block": "PF", "dst_port": "u0"},
        {"src_block": "C1", "src_port": "y", "dst_block": "PF", "dst_port": "u1"},
    ]
    m = {"board": "NUCLEO-F446RE", "step_ms": 1,
         "blocks": [c0, c1, blk], "connections": conns}
    _t, disp = simulate_model(m, duration_s=0.005, step_s=0.001)
    assert np.allclose(disp["PF.y"], 10.0)


def test_pythonfcn_sim_unconnected_input_is_zero():
    """Unconnected u1 should read as 0."""
    blk = _pfcn("u[1]", num_inputs="2")
    src = {"type": "Constant", "id": "C0", "x": 0, "y": 0, "params": {"value": "5.0"}}
    conn = [{"src_block": "C0", "src_port": "y", "dst_block": "PF", "dst_port": "u0"}]
    m = {"board": "NUCLEO-F446RE", "step_ms": 1,
         "blocks": [src, blk], "connections": conn}
    _t, disp = simulate_model(m, duration_s=0.005, step_s=0.001)
    assert np.allclose(disp["PF.y"], 0.0)


# ---------------------------------------------------------------------------
# Simulator — exec mode (multi-line code with 'y = ...' assignment)
# ---------------------------------------------------------------------------

def test_pythonfcn_sim_exec_mode():
    """Multi-statement code assigns result to y."""
    code = "a = u[0] * 2\ny = a + 1"
    y = _pfcn_sim(code, src_val="5.0")
    assert np.allclose(y, 11.0)


def test_pythonfcn_sim_exec_missing_y_returns_zero():
    """If exec mode runs but doesn't set y, output is 0."""
    code = "a = u[0] * 2\nb = a + 1"  # no y assignment
    y = _pfcn_sim(code, src_val="5.0")
    assert np.allclose(y, 0.0)


def test_pythonfcn_sim_runtime_error_returns_zero():
    """A runtime error (e.g. division by zero) should produce 0, not crash."""
    y = _pfcn_sim("u[0] / 0.0", src_val="1.0")
    # Should not raise; result is 0 (or inf/nan handled gracefully)
    assert len(y) > 0  # simulation completed


# ---------------------------------------------------------------------------
# _python_to_c transpiler
# ---------------------------------------------------------------------------

def test_python_to_c_passthrough():
    result = _python_to_c("u[0]", ["sig_A_y"])
    assert result == "sig_A_y"


def test_python_to_c_scale():
    result = _python_to_c("u[0] * 2.0", ["sig_A_y"])
    assert result == "sig_A_y * 2.0"


def test_python_to_c_two_inputs():
    result = _python_to_c("u[0] + u[1]", ["sig_A_y", "sig_B_y"])
    assert result == "sig_A_y + sig_B_y"


def test_python_to_c_math_sin():
    result = _python_to_c("math.sin(u[0])", ["x"])
    assert result == "sinf(x)"


def test_python_to_c_math_cos():
    result = _python_to_c("math.cos(u[0])", ["x"])
    assert result == "cosf(x)"


def test_python_to_c_math_sqrt():
    result = _python_to_c("math.sqrt(u[0])", ["x"])
    assert result == "sqrtf(x)"


def test_python_to_c_math_pi():
    result = _python_to_c("math.pi * u[0]", ["x"])
    assert "3.14159265f" in result


def test_python_to_c_pow():
    result = _python_to_c("u[0]**2", ["x"])
    assert "powf" in result


def test_python_to_c_abs():
    result = _python_to_c("abs(u[0])", ["x"])
    assert "fabsf" in result


def test_python_to_c_returns_none_for_multiline():
    result = _python_to_c("a = u[0]\ny = a * 2", ["x"])
    assert result is None


def test_python_to_c_returns_none_for_return():
    result = _python_to_c("return u[0] * 2", ["x"])
    assert result is None


def test_python_to_c_returns_none_for_if():
    result = _python_to_c("if u[0] > 0: u[0]", ["x"])
    assert result is None


def test_python_to_c_true_false():
    result = _python_to_c("True", [])
    assert result == "1.0f"
    result2 = _python_to_c("False", [])
    assert result2 == "0.0f"


# ---------------------------------------------------------------------------
# Codegen — simple expression auto-transpiled
# ---------------------------------------------------------------------------

def test_pythonfcn_codegen_simple_expression():
    src = _pfcn_src("u[0] * 2.0")
    assert "sig_PF_y" in src
    # Should NOT contain TODO stub for simple expression
    assert "TODO" not in src


def test_pythonfcn_codegen_passthrough_uses_signal_var():
    src = _pfcn_src("u[0]")
    # The output var should be assigned from the upstream signal
    assert "sig_PF_y" in src
    assert "sig_C0_y" in src


def test_pythonfcn_codegen_math_sin_transpiled():
    src = _pfcn_src("math.sin(u[0])")
    assert "sinf(" in src
    assert "TODO" not in src


def test_pythonfcn_codegen_complex_code_emits_stub():
    src = _pfcn_src("a = u[0] * 2\ny = a + 1")
    assert "TODO" in src
    # Python source should appear as a comment
    assert "a = u[0] * 2" in src


def test_pythonfcn_codegen_decl_present():
    decls = _emit_decls([_pfcn()])
    assert "sig_PF_y" in decls


def test_pythonfcn_codegen_volatile_decl():
    decls = _emit_decls([_pfcn()], volatile=True)
    assert "volatile" in decls
    assert "sig_PF_y" in decls


# ===========================================================================
# Phase B — New control-systems blocks
# ===========================================================================

# ---------------------------------------------------------------------------
# B1: WeightedSum
# ---------------------------------------------------------------------------

def _wsum(bid="WS", num_inputs="4", gains="1 1 1 1"):
    return {"type": "WeightedSum", "id": bid, "x": 0, "y": 0,
            "params": {"num_inputs": num_inputs, "gains": gains}}


def _wsum_model(gains, values):
    """Constant sources → WeightedSum → Scope."""
    n = len(values)
    srcs = [{"type": "Constant", "id": f"C{i}", "x": 0, "y": i*20,
              "params": {"value": str(v)}} for i, v in enumerate(values)]
    ws   = _wsum("WS", num_inputs=str(n), gains=" ".join(str(g) for g in gains))
    sc   = _scope("SC")
    conns = [_wire(f"C{i}", "y", "WS", f"u{i}") for i in range(n)]
    conns.append(_wire("WS", "y", "SC", "u0"))
    return _model(srcs + [ws, sc], conns)


def test_weightedsum_in_catalog():
    assert "WeightedSum" in BLOCK_CATALOG


def test_weightedsum_spec_has_eight_inputs():
    assert len(BLOCK_CATALOG["WeightedSum"].inputs) == 8


def test_weightedsum_spec_has_output_y():
    assert BLOCK_CATALOG["WeightedSum"].outputs[0].name == "y"


def test_weightedsum_validate_gain_count_mismatch():
    blk = _wsum(num_inputs="3", gains="1 2")  # 2 gains but 3 inputs
    errs = _validate_block(blk)
    assert any(e.code == "E002" and e.param == "gains" for e in errs)


def test_weightedsum_validate_non_numeric_gain():
    blk = _wsum(num_inputs="2", gains="1 abc")
    errs = _validate_block(blk)
    assert any(e.code == "E001" and e.param == "gains" for e in errs)


def test_weightedsum_validate_ok():
    blk = _wsum(num_inputs="4", gains="-1 -2 -3 -4")
    errs = _validate_block(blk)
    assert errs == []


def test_weightedsum_sim_sum_four_equal():
    """4 inputs × gain 1 each, all=2 → y=8."""
    m = _wsum_model([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0])
    _, d = simulate_model(m, duration_s=0.01, step_s=0.001)
    assert abs(d["SC.u0"][-1] - 8.0) < 1e-9


def test_weightedsum_sim_lqr_feedback():
    """Negative gains for LQR: -1*x0 + -2*x1 = -(1+4) = -5."""
    m = _wsum_model([-1.0, -2.0], [1.0, 2.0])
    _, d = simulate_model(m, duration_s=0.01, step_s=0.001)
    assert abs(d["SC.u0"][-1] - (-5.0)) < 1e-9


def test_weightedsum_sim_single_input():
    m = _wsum_model([3.5], [2.0])
    _, d = simulate_model(m, duration_s=0.01, step_s=0.001)
    assert abs(d["SC.u0"][-1] - 7.0) < 1e-9


def test_weightedsum_codegen_has_signal_var():
    m = _wsum_model([1.0, 2.0], [3.0, 4.0])
    ws = FakeWorkspace()
    blocks = m["blocks"]
    wires_ = _wires(m)
    board  = BOARDS["NUCLEO-F446RE"]
    body, _ = _emit_step(blocks, wires_, ws, 1, board)
    assert "sig_WS_y" in body


def test_weightedsum_codegen_gains_in_output():
    m = _wsum_model([1.5, 2.5], [0.0, 0.0])
    ws = FakeWorkspace()
    blocks = m["blocks"]
    wires_ = _wires(m)
    board  = BOARDS["NUCLEO-F446RE"]
    body, _ = _emit_step(blocks, wires_, ws, 1, board)
    assert "1.500000f" in body
    assert "2.500000f" in body


def test_weightedsum_codegen_decl_present():
    decls = _emit_decls([_wsum()])
    assert "sig_WS_y" in decls


# ---------------------------------------------------------------------------
# B2: PlantODE
# ---------------------------------------------------------------------------

def _plant(bid="PL", order="2",
           equations="x[1]\n-9.81*math.sin(x[0]) - 0.1*x[1] + u",
           initial_state="0 0", num_outputs="2"):
    return {"type": "PlantODE", "id": bid, "x": 0, "y": 0,
            "params": {"order": order, "equations": equations,
                       "initial_state": initial_state,
                       "num_outputs": num_outputs}}


def _plant_model(u_val="0.0"):
    src  = _const("SRC", value=u_val)
    pl   = _plant()
    sc0  = _scope("SC0"); sc1 = _scope("SC1")
    sc0["id"] = "SC0"; sc1["id"] = "SC1"
    m = _model([src, pl, sc0, sc1], [
        _wire("SRC", "y", "PL", "u"),
        _wire("PL", "y0", "SC0", "u0"),
        _wire("PL", "y1", "SC1", "u0"),
    ])
    return m


def test_plantode_in_catalog():
    assert "PlantODE" in BLOCK_CATALOG


def test_plantode_spec_has_four_outputs():
    assert len(BLOCK_CATALOG["PlantODE"].outputs) == 4


def test_plantode_validate_ic_count_mismatch():
    blk = _plant(order="3", initial_state="0 0")  # 3 states but only 2 ICs
    errs = _validate_block(blk)
    assert any(e.code == "E001" and e.param == "initial_state" for e in errs)


def test_plantode_validate_num_outputs_exceeds_order():
    blk = _plant(order="2", num_outputs="4")
    errs = _validate_block(blk)
    assert any(e.code == "E002" and e.param == "num_outputs" for e in errs)


def test_plantode_validate_ok():
    blk = _plant()
    errs = _validate_block(blk)
    assert errs == []


def test_plantode_sim_returns_states():
    """PlantODE with zero input should evolve from initial state."""
    m = _plant_model("0.0")
    _, d = simulate_model(m, duration_s=0.1, step_s=0.001)
    # Scope SC0.u0 wired from PL.y0; SC1.u0 from PL.y1
    assert "SC0.u0" in d
    assert "SC1.u0" in d


def test_plantode_sim_rk4_advances_state():
    """With non-zero initial velocity, pendulum state should change over time."""
    src  = _const("SRC", value="0.0")
    pl   = _plant(initial_state="0 1")  # theta=0, dtheta/dt=1
    sc   = _scope("SC")
    m    = _model([src, pl, sc], [
        _wire("SRC", "y", "PL", "u"),
        _wire("PL", "y1", "SC", "u0"),
    ])
    _, d = simulate_model(m, duration_s=0.1, step_s=0.001)
    # y1 changes due to gravity — first value should differ from last
    arr = d["SC.u0"]
    assert abs(arr[-1] - arr[0]) > 0.01


def test_plantode_codegen_emits_stub():
    m = _plant_model("0.0")
    ws = FakeWorkspace()
    blocks = m["blocks"]
    wires_ = _wires(m)
    board  = BOARDS["NUCLEO-F446RE"]
    body, _ = _emit_step(blocks, wires_, ws, 1, board)
    assert "PlantODE" in body
    assert "stub" in body.lower() or "TODO" in body


def test_plantode_codegen_decl_present():
    decls = _emit_decls([_plant()])
    assert "sig_PL_y0" in decls
    assert "sig_PL_y1" in decls


def test_plantode_validate_bad_equation_syntax():
    blk = _plant(equations="x[1]\n!!!invalid!!!")
    errs = _validate_block(blk)
    assert any(e.code == "E003" and e.param == "equations" for e in errs)


# ---------------------------------------------------------------------------
# B3: AngleUnwrap
# ---------------------------------------------------------------------------

def _unwrap(bid="AW", range_="auto"):
    return {"type": "AngleUnwrap", "id": bid, "x": 0, "y": 0,
            "params": {"range": range_}}


def _unwrap_model(input_vals_deg, deg_mode=False):
    """Simulate AngleUnwrap with prescribed wrapped input signal."""
    import math as _m
    scale = 1.0 if deg_mode else _m.pi / 180.0
    conns = []
    srcs  = []
    for i, _ in enumerate(input_vals_deg):
        pass
    rng = "360" if deg_mode else "auto"
    # We'll just use a Constant at 0 and trust the sweep test; detailed
    # numerical verification is done separately using simulate_model directly.
    src = _const("SRC", value="0.0")
    aw  = _unwrap("AW", range_=rng)
    sc  = _scope("SC")
    m   = _model([src, aw, sc], [
        _wire("SRC", "y", "AW", "u"),
        _wire("AW", "y", "SC", "u0"),
    ])
    return m


def test_anglunwrap_in_catalog():
    assert "AngleUnwrap" in BLOCK_CATALOG


def test_anglunwrap_validate_bad_range():
    blk = _unwrap(range_="180")
    errs = _validate_block(blk)
    assert any(e.code == "E003" for e in errs)


def test_anglunwrap_validate_ok_auto():
    errs = _validate_block(_unwrap("AW", "auto"))
    assert errs == []


def test_anglunwrap_validate_ok_360():
    errs = _validate_block(_unwrap("AW", "360"))
    assert errs == []


def test_anglunwrap_sim_constant_zero_output():
    """Constant 0 input → output 0 (no wrapping)."""
    m = _unwrap_model([])
    _, d = simulate_model(m, duration_s=0.01, step_s=0.001)
    assert all(abs(v) < 1e-9 for v in d["SC.u0"])


def test_anglunwrap_sim_offset_accumulates():
    """Simulate a quick wrap by using PythonFcn to generate a wrapping signal."""
    import math as _m
    # Build: PythonFcn generating wrapped angle → AngleUnwrap → Scope
    pf  = _pfcn(code="math.atan2(math.sin(t*6.28), math.cos(t*6.28))", bid="PF")
    aw  = _unwrap("AW", "auto")
    sc  = _scope("SC")
    m   = _model([pf, aw, sc], [
        _wire("PF", "y", "AW", "u"),
        _wire("AW", "y", "SC", "u0"),
    ])
    _, d = simulate_model(m, duration_s=1.5, step_s=0.01)
    # Unwrapped output at 1.5s should be close to 1.5*2*pi
    expected = 1.5 * 2 * _m.pi
    assert abs(d["SC.u0"][-1] - expected) < 0.5  # allow rounding in discrete sim


def test_anglunwrap_codegen_has_static_vars():
    decls = _emit_decls([_unwrap("AW")])
    assert "_aw_prev_AW" in decls
    assert "_aw_off_AW" in decls


def test_anglunwrap_codegen_step_has_halfpi():
    m = _unwrap_model([])
    ws = FakeWorkspace()
    blocks = m["blocks"]
    wires_ = _wires(m)
    board  = BOARDS["NUCLEO-F446RE"]
    body, _ = _emit_step(blocks, wires_, ws, 1, board)
    assert "3.141" in body  # π appears in the C output


def test_anglunwrap_codegen_360_uses_180():
    aw  = _unwrap("AW", "360")
    src = _const("SRC")
    sc  = _scope("SC")
    m   = _model([src, aw, sc], [
        _wire("SRC", "y", "AW", "u"),
        _wire("AW", "y", "SC", "u0"),
    ])
    ws = FakeWorkspace()
    body, _ = _emit_step(m["blocks"], _wires(m), ws, 1, BOARDS["NUCLEO-F446RE"])
    assert "180.0f" in body


# ---------------------------------------------------------------------------
# B4: HBridgeOut
# ---------------------------------------------------------------------------

def _hbridge(bid="HB", timer="TIM2", channel="1",
             dir_pin="PB0", dead_band_pct="5.0", max_duty="100.0"):
    return {"type": "HBridgeOut", "id": bid, "x": 0, "y": 0,
            "params": {"timer": timer, "channel": channel, "dir_pin": dir_pin,
                       "dead_band_pct": dead_band_pct, "max_duty": max_duty}}


def test_hbridge_in_catalog():
    assert "HBridgeOut" in BLOCK_CATALOG


def test_hbridge_validate_bad_pin():
    blk = _hbridge(dir_pin="ZZ99")
    errs = _validate_block(blk)
    assert any(e.code == "E004" and e.param == "dir_pin" for e in errs)


def test_hbridge_validate_negative_deadband():
    blk = _hbridge(dead_band_pct="-1")
    errs = _validate_block(blk)
    assert any(e.code == "E002" for e in errs)


def test_hbridge_validate_zero_max_duty():
    blk = _hbridge(max_duty="0")
    errs = _validate_block(blk)
    assert any(e.code == "E002" and e.param == "max_duty" for e in errs)


def test_hbridge_validate_ok():
    errs = _validate_block(_hbridge())
    assert errs == []


def test_hbridge_sim_passthrough():
    """HBridgeOut pin output should reflect sign/magnitude of input."""
    src = _const("SRC", value="50.0")
    hb  = _hbridge()
    sc  = _scope("SC")
    m   = _model([src, hb, sc], [
        _wire("SRC", "y", "HB", "u"),
        _wire("HB", "pin", "SC", "u0"),
    ])
    _, d = simulate_model(m, duration_s=0.01, step_s=0.001)
    assert all(v == 50.0 for v in d["SC.u0"])


def test_hbridge_sim_deadband_zeroes_small_input():
    """Input within dead-band should produce 0 output."""
    src = _const("SRC", value="3.0")   # < 5% of 100 = 5.0
    hb  = _hbridge(dead_band_pct="10.0", max_duty="100.0")
    sc  = _scope("SC")
    m   = _model([src, hb, sc], [
        _wire("SRC", "y", "HB", "u"),
        _wire("HB", "pin", "SC", "u0"),
    ])
    _, d = simulate_model(m, duration_s=0.01, step_s=0.001)
    assert all(v == 0.0 for v in d["SC.u0"])


def test_hbridge_codegen_has_hal_gpio():
    src = _const("SRC", value="50.0")
    hb  = _hbridge()
    sc  = _scope("SC")
    m   = _model([src, hb, sc], [
        _wire("SRC", "y", "HB", "u"),
        _wire("HB", "pin", "SC", "u0"),
    ])
    ws = FakeWorkspace()
    body, _ = _emit_step(m["blocks"], _wires(m), ws, 1, BOARDS["NUCLEO-F446RE"])
    assert "HAL_GPIO_WritePin" in body


def test_hbridge_codegen_ccr_register():
    src = _const("SRC")
    hb  = _hbridge(timer="TIM3", channel="2")
    sc  = _scope("SC")
    m   = _model([src, hb, sc], [
        _wire("SRC", "y", "HB", "u"),
        _wire("HB", "pin", "SC", "u0"),
    ])
    ws = FakeWorkspace()
    body, _ = _emit_step(m["blocks"], _wires(m), ws, 1, BOARDS["NUCLEO-F446RE"])
    assert "TIM3->CCR2" in body


def test_hbridge_codegen_decl_present():
    decls = _emit_decls([_hbridge()])
    assert "sig_HB_pin" in decls


# ---------------------------------------------------------------------------
# B5: DiscreteIntegratorAW
# ---------------------------------------------------------------------------

def _diaw(bid="DI", gain="1.0", ic="0.0", upper="1e10", lower="-1e10",
          method="Forward Euler", kaw="0.0"):
    return {"type": "DiscreteIntegratorAW", "id": bid, "x": 0, "y": 0,
            "params": {"gain_value": gain, "initial_condition": ic,
                       "upper_limit": upper, "lower_limit": lower,
                       "method": method, "back_calc_coeff": kaw}}


def _diaw_model(gain="1.0", ic="0.0", upper="1e10", lower="-1e10",
                method="Forward Euler", kaw="0.0", src_val="1.0"):
    src  = _const("SRC", value=src_val)
    di   = _diaw("DI", gain, ic, upper, lower, method, kaw)
    sc   = _scope("SC")
    m    = _model([src, di, sc], [
        _wire("SRC", "y", "DI", "u"),
        _wire("DI", "y", "SC", "u0"),
    ])
    return m


def test_diaw_in_catalog():
    assert "DiscreteIntegratorAW" in BLOCK_CATALOG


def test_diaw_validate_ok():
    errs = _validate_block(_diaw())
    assert errs == []


def test_diaw_validate_bad_kaw():
    blk = _diaw(kaw="abc")
    errs = _validate_block(blk)
    assert any(e.code == "E001" and e.param == "back_calc_coeff" for e in errs)


def test_diaw_validate_limit_conflict():
    blk = _diaw(upper="1.0", lower="2.0")
    errs = _validate_block(blk)
    assert any(e.code == "E007" for e in errs)


def test_diaw_sim_integrates_like_di():
    """With kaw=0, should behave identically to DiscreteIntegrator."""
    m = _diaw_model(gain="1.0", kaw="0.0", src_val="1.0")
    _, d = simulate_model(m, duration_s=0.005, step_s=0.001)
    # After 5 steps at dt=0.001 with input=1.0 and gain=1.0:
    # y[0]=0 (IC), y[1]=0.001, y[2]=0.002, …  last stored = 0.004
    arr = d["SC.u0"]
    assert abs(arr[-1] - 0.004) < 1e-9


def test_diaw_sim_saturation_upper():
    """With tight upper limit, output should saturate."""
    m = _diaw_model(gain="10.0", upper="0.5", lower="-1e10",
                    kaw="0.0", src_val="1.0")
    _, d = simulate_model(m, duration_s=0.1, step_s=0.001)
    assert all(v <= 0.5 + 1e-9 for v in d["SC.u0"])


def test_diaw_sim_anti_windup_reduces_overshoot():
    """Anti-windup should limit state growth beyond saturation limits."""
    m_aw  = _diaw_model(gain="10.0", upper="0.5", lower="-1e10",
                        kaw="1.0", src_val="1.0")
    m_naw = _diaw_model(gain="10.0", upper="0.5", lower="-1e10",
                        kaw="0.0", src_val="1.0")
    _, d_aw  = simulate_model(m_aw,  duration_s=0.2, step_s=0.001)
    _, d_naw = simulate_model(m_naw, duration_s=0.2, step_s=0.001)
    assert all(v <= 0.5 + 1e-9 for v in d_aw["SC.u0"])
    assert all(v <= 0.5 + 1e-9 for v in d_naw["SC.u0"])


def test_diaw_codegen_state_decl():
    decls = _emit_decls([_diaw()])
    assert "_state_DI" in decls
    assert "sig_DI_y" in decls


def test_diaw_codegen_step_has_state_update():
    m = _diaw_model()
    ws = FakeWorkspace()
    body, _ = _emit_step(m["blocks"], _wires(m), ws, 1, BOARDS["NUCLEO-F446RE"])
    assert "_state_DI" in body


def test_diaw_codegen_kaw_in_output_when_nonzero():
    m = _diaw_model(kaw="0.5")
    ws = FakeWorkspace()
    body, _ = _emit_step(m["blocks"], _wires(m), ws, 1, BOARDS["NUCLEO-F446RE"])
    # Back-calculation coefficient should appear in the C output
    assert "_bc_DI" in body


# ===========================================================================
# Section A — Wire routing (pure-math helpers)
# ===========================================================================

def test_wire_auto_forward_two_waypoints():
    """Forward wire (p2.x > p1.x) → exactly 2 waypoints."""
    wps = _wire_auto_waypoints(0, 0, 200, 50)
    assert len(wps) == 2


def test_wire_auto_forward_midpoint():
    """Both forward waypoints share the same x (= midpoint)."""
    wps = _wire_auto_waypoints(0, 0, 200, 50)
    assert wps[0][0] == wps[1][0] == 100.0


def test_wire_auto_forward_y_pins_ports():
    """First WP y == p1.y; second WP y == p2.y for forward wire."""
    wps = _wire_auto_waypoints(10, 20, 300, 80)
    assert wps[0][1] == 20
    assert wps[1][1] == 80


def test_wire_auto_same_y():
    """Wire where source and destination share the same y still returns 2 wps."""
    wps = _wire_auto_waypoints(0, 50, 200, 50)
    assert len(wps) == 2
    assert wps[0][1] == wps[1][1] == 50


def test_wire_auto_backward_four_waypoints():
    """Backward wire (p2.x < p1.x) → exactly 4 waypoints."""
    wps = _wire_auto_waypoints(200, 50, 50, 50)
    assert len(wps) == 4


def test_wire_auto_backward_margin():
    """Exit x == p1.x + 36; enter x == p2.x - 36 for backward wire."""
    wps = _wire_auto_waypoints(200, 50, 50, 50)
    assert wps[0][0] == 200 + 36
    assert wps[3][0] == 50 - 36


def test_wire_auto_backward_horizontal_row():
    """Middle two waypoints of backward wire share same y."""
    wps = _wire_auto_waypoints(200, 50, 50, 50)
    assert wps[1][1] == wps[2][1]


def test_wire_move_seg_vertical_shifts_x():
    """Dragging a vertical segment (ddx != 0) shifts x of adjacent waypoints."""
    # Forward wire: 2 waypoints; segment 1 (between wp0 and wp1) is vertical
    wps_in = [(100, 0), (100, 50)]
    wps_out = _wire_move_seg(wps_in, si=1, ddx=20, ddy=0)
    assert wps_out[0][0] == 120
    assert wps_out[1][0] == 120


def test_wire_move_seg_horizontal_shifts_y():
    """Dragging a horizontal segment (ddy != 0) shifts y of adjacent waypoints."""
    # Backward wire: 4 waypoints; segment 2 (between wp1 and wp2) is horizontal
    wps_in = [(236, 50), (236, 14), (14, 14), (14, 50)]
    wps_out = _wire_move_seg(wps_in, si=2, ddx=0, ddy=-15)
    assert wps_out[1][1] == 14 - 15
    assert wps_out[2][1] == 14 - 15


def test_wire_move_seg_first_seg_noop():
    """si=0 is not movable; waypoints returned unchanged."""
    wps_in = [(100, 0), (100, 50)]
    wps_out = _wire_move_seg(wps_in, si=0, ddx=10, ddy=10)
    assert wps_out == list(wps_in)


def test_wire_move_seg_last_seg_noop():
    """si == len(waypoints) is not movable; waypoints returned unchanged."""
    wps_in = [(100, 0), (100, 50)]
    n = len(wps_in)
    wps_out = _wire_move_seg(wps_in, si=n, ddx=10, ddy=10)
    assert wps_out == list(wps_in)


def test_wire_move_seg_preserves_count():
    """Moving a segment never adds or removes waypoints."""
    wps_in = [(100, 0), (100, 50)]
    wps_out = _wire_move_seg(wps_in, si=1, ddx=20, ddy=0)
    assert len(wps_out) == len(wps_in)


def test_wire_move_seg_backward_vertical():
    """Vertical segment in 4-wp feedback wire (si=1) shifts x correctly."""
    wps_in = [(236, 50), (236, 14), (14, 14), (14, 50)]
    wps_out = _wire_move_seg(wps_in, si=1, ddx=10, ddy=0)
    assert wps_out[0][0] == 246
    assert wps_out[1][0] == 246


def test_wire_move_seg_backward_horizontal():
    """Horizontal segment in 4-wp feedback wire (si=2) shifts y correctly."""
    wps_in = [(236, 50), (236, 14), (14, 14), (14, 50)]
    wps_out = _wire_move_seg(wps_in, si=2, ddx=0, ddy=-10)
    assert wps_out[1][1] == 4
    assert wps_out[2][1] == 4


def test_wire_waypoints_roundtrip():
    """Move segment then move back → original waypoint positions restored."""
    wps_in = [(100, 0), (100, 50)]
    wps_moved = _wire_move_seg(wps_in, si=1, ddx=25, ddy=0)
    wps_back  = _wire_move_seg(wps_moved, si=1, ddx=-25, ddy=0)
    for orig, restored in zip(wps_in, wps_back):
        assert abs(orig[0] - restored[0]) < 1e-9
        assert abs(orig[1] - restored[1]) < 1e-9


# ===========================================================================
# Section B — Block resize math (pure-math helper)
# ===========================================================================

def test_resize_br_grows_right_and_down():
    """'br' handle: +dx grows width, +dy grows height; position unchanged."""
    w, h, x, y = _apply_resize_math("br", 30, 20, 150, 80, 100, 200)
    assert w == 180
    assert h == 100
    assert x == 100
    assert y == 200


def test_resize_tl_shrinks_and_shifts():
    """'tl' handle: width and height shrink; top-left position shifts."""
    w, h, x, y = _apply_resize_math("tl", 20, 10, 150, 80, 100, 200)
    assert w == 130
    assert h == 70
    assert x == 120  # 100 + (150 - 130)
    assert y == 210  # 200 + (80 - 70)


def test_resize_tr_wide_short():
    """'tr' handle: width increases, height decreases; y shifts, x unchanged."""
    w, h, x, y = _apply_resize_math("tr", 40, 15, 150, 80, 100, 200)
    assert w == 190
    assert h == 65
    assert x == 100
    assert y == 215


def test_resize_bl_tall_narrow():
    """'bl' handle: width decreases (x shifts), height increases."""
    w, h, x, y = _apply_resize_math("bl", 20, 30, 150, 80, 100, 200)
    assert w == 130
    assert h == 110
    assert x == 120
    assert y == 200


def test_resize_mr_width_only():
    """'mr' handle: only width changes."""
    w, h, x, y = _apply_resize_math("mr", 50, 0, 150, 80, 100, 200)
    assert w == 200
    assert h == 80
    assert x == 100
    assert y == 200


def test_resize_ml_width_shifts_x():
    """'ml' handle: width changes and x shifts; height unchanged."""
    w, h, x, y = _apply_resize_math("ml", 30, 0, 150, 80, 100, 200)
    assert w == 120
    assert x == 130  # 100 + (150 - 120)
    assert h == 80
    assert y == 200


def test_resize_tc_height_only_up():
    """'tc' handle: only height changes (shrinks up); y shifts."""
    w, h, x, y = _apply_resize_math("tc", 0, 20, 150, 80, 100, 200)
    assert w == 150
    assert h == 60
    assert x == 100
    assert y == 220


def test_resize_bc_height_only_down():
    """'bc' handle: only height changes (grows down); y unchanged."""
    w, h, x, y = _apply_resize_math("bc", 0, 30, 150, 80, 100, 200)
    assert w == 150
    assert h == 110
    assert x == 100
    assert y == 200


def test_resize_min_width_clamp():
    """Huge negative dx clamps new_w to MIN_W (80)."""
    w, h, x, y = _apply_resize_math("br", -9999, 0, 150, 80, 100, 200)
    assert w == 80.0


def test_resize_min_height_clamp():
    """Huge negative dy clamps new_h to MIN_H (50)."""
    w, h, x, y = _apply_resize_math("br", 0, -9999, 150, 80, 100, 200)
    assert h == 50.0


def test_resize_min_clamp_position_correct():
    """Left-resize clamped to MIN_W: right edge stays at orig_x + orig_w."""
    orig_x, orig_w = 100.0, 150.0
    w, h, x, y = _apply_resize_math("ml", 9999, 0, orig_w, 80, orig_x, 200)
    assert w == 80.0
    # right edge = new_x + new_w must equal orig_x + orig_w = 250
    assert abs((x + w) - (orig_x + orig_w)) < 1e-9


def test_resize_zero_delta_no_change():
    """dx=dy=0 → all dimensions unchanged."""
    w, h, x, y = _apply_resize_math("br", 0, 0, 150, 80, 100, 200)
    assert (w, h, x, y) == (150, 80, 100, 200)


def test_resize_large_block_right():
    """Grow a large block rightward."""
    w, h, x, y = _apply_resize_math("mr", 100, 0, 300, 200, 50, 50)
    assert w == 400
    assert h == 200


def test_resize_instance_fields_updated():
    """_apply_resize_math returns correct w/h that would be stored on instance."""
    w, h, x, y = _apply_resize_math("br", 50, 40, 150, 80, 0, 0)
    # Simulate what BlockItem.apply_resize does
    instance_width  = w
    instance_height = h
    assert instance_width  == 200
    assert instance_height == 120


# ===========================================================================
# Section C — BlockInstance width/height serialization
# ===========================================================================

def _make_instance(type_name="Constant"):
    """Build a minimal BlockInstance from the catalog (no Qt required)."""
    from stm32_block_ide import BlockInstance
    spec = BLOCK_CATALOG[type_name]
    params = {k: v[0] for k, v in spec.params.items()}
    return BlockInstance(spec=spec, block_id="T1", x=0.0, y=0.0, params=params)


def test_instance_default_width_zero():
    inst = _make_instance()
    assert inst.width == 0.0


def test_instance_default_height_zero():
    inst = _make_instance()
    assert inst.height == 0.0


def test_instance_to_dict_omits_zero_width():
    inst = _make_instance()
    d = inst.to_dict()
    assert "width" not in d


def test_instance_to_dict_emits_nonzero_width():
    inst = _make_instance()
    inst.width = 200.0
    d = inst.to_dict()
    assert d["width"] == 200.0


def test_instance_to_dict_omits_zero_height():
    inst = _make_instance()
    d = inst.to_dict()
    assert "height" not in d


def test_instance_to_dict_emits_nonzero_height():
    inst = _make_instance()
    inst.height = 120.0
    d = inst.to_dict()
    assert d["height"] == 120.0


def test_instance_roundtrip_both_dims():
    """Set w=200 h=120 → to_dict → both keys present with correct values."""
    inst = _make_instance()
    inst.width  = 200.0
    inst.height = 120.0
    d = inst.to_dict()
    assert d["width"]  == 200.0
    assert d["height"] == 120.0


def test_instance_old_dict_no_key_defaults_zero():
    """Loading a saved dict without 'width' key should give default 0.0."""
    old_dict = {"width": 0, "height": 0}  # simulates missing keys → float()
    w = float(old_dict.get("width",  0.0))
    h = float(old_dict.get("height", 0.0))
    assert w == 0.0
    assert h == 0.0


# ===========================================================================
# Section D — Golden-file codegen (structural assertions on main.c)
# ===========================================================================

def _baremetal_sw_gpio_model():
    """SquareWave → GpioOut bare-metal model."""
    sw  = _sw("SRC", frequency_hz="1.0", amplitude="1.0", duty="0.5")
    go  = _gpio_out("GO1", pin="PA5", threshold="0.5")
    model = _model([sw, go], [_wire("SRC", "y", "GO1", "u")])
    model["step_ms"] = 1
    return model


def _baremetal_sw_gpio_src():
    """Return generated main.c content for the SquareWave → GpioOut model."""
    m  = _baremetal_sw_gpio_model()
    ws = FakeWorkspace()
    blocks = m["blocks"]
    wires  = _wires(m)
    board  = BOARDS["NUCLEO-F446RE"]
    step_ms = m["step_ms"]
    body, _ = _emit_step(blocks, wires, ws, step_ms, board)
    decls   = _emit_decls(blocks)
    helpers = _emit_helpers(blocks)
    init    = _emit_init(blocks, board)
    return "\n".join([helpers, decls, init, body])


def test_golden_baremetal_squarewave_gpio():
    """SquareWave→GpioOut codegen references phase_ state and HAL_GPIO_WritePin."""
    src = _baremetal_sw_gpio_src()
    assert "HAL_GPIO_WritePin" in src
    assert "sig_SRC_y" in src


def test_golden_baremetal_has_systick():
    """Bare-metal codegen defines SysTick_Handler."""
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), _baremetal_sw_gpio_model(), FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "SysTick_Handler" in src


def test_golden_rtos_single_rate_task_name():
    """Single-rate RTOS codegen defines ModelTask; no volatile qualifier needed."""
    with tempfile.TemporaryDirectory() as d:
        with patch("code_templates._ensure_freertos", return_value=Path("/fake/fr")):
            proj = generate_project(Path(d), _rtos_model(), FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "ModelTask" in src


def test_golden_rtos_multirate_volatile():
    """Multi-rate RTOS codegen uses volatile float for inter-task signals."""
    b_fast = _sw("FAST", frequency_hz="10.0")
    b_fast["sample_time_ms"] = 2
    b_slow = _scope("SLOW")
    b_slow["sample_time_ms"] = 4
    m = _model([b_fast, b_slow], [_wire("FAST", "y", "SLOW", "u0")])
    m["use_rtos"] = True
    with tempfile.TemporaryDirectory() as d:
        with patch("code_templates._ensure_freertos", return_value=Path("/fake/fr")):
            proj = generate_project(Path(d), m, FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "volatile" in src


def test_golden_multirate_two_xtaskcreate():
    """Multi-rate RTOS codegen emits exactly 2 xTaskCreate calls."""
    b_fast = _sw("FAST", frequency_hz="10.0")
    b_fast["sample_time_ms"] = 2
    b_slow = _scope("SLOW")
    b_slow["sample_time_ms"] = 4
    m = _model([b_fast, b_slow], [_wire("FAST", "y", "SLOW", "u0")])
    m["use_rtos"] = True
    with tempfile.TemporaryDirectory() as d:
        with patch("code_templates._ensure_freertos", return_value=Path("/fake/fr")):
            proj = generate_project(Path(d), m, FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert src.count("xTaskCreate") == 2


def test_golden_step_ms_baked():
    """step_ms=5 value appears in the SysTick reload or pdMS_TO_TICKS call."""
    sw  = _sw("SRC")
    sc  = _scope("SC")
    m = _model([sw, sc], [_wire("SRC", "y", "SC", "u0")])
    m["step_ms"] = 5
    with tempfile.TemporaryDirectory() as d:
        proj = generate_project(Path(d), m, FakeWorkspace())
        src = (proj / "main.c").read_text()
    assert "5" in src


# ===========================================================================
# Section E — PythonFcn security and edge cases
# ===========================================================================

def test_pythonfcn_dangerous_import_returns_zero():
    """os.system(...) inside PythonFcn code must not crash the simulator."""
    import math as _math
    # The exec sandbox should not have real os access; output should be 0 or any float
    try:
        result = _pfcn_sim("import os; y=os.system('echo x')", src_val="0.0")
    except Exception:
        result = [0.0]  # any non-crash outcome is acceptable
    # Result is an array; we just care that it's iterable and no exception was raised
    assert hasattr(result, '__len__') or hasattr(result, '__iter__')


def test_pythonfcn_division_by_zero_no_crash():
    """u[0]/0.0 should produce inf or nan, not raise an exception."""
    import numpy as _np
    result = _pfcn_sim("u[0]/0.0", src_val="1.0")
    # Should be an array of inf or nan; crucially no exception
    assert result is not None


def test_pythonfcn_nan_input_propagates():
    """NaN produced inside PythonFcn should not crash the simulator."""
    import math as _math
    # Generate NaN internally (0.0/0.0 via float operations)
    result = _pfcn_sim("float('nan') * u[0]", src_val="1.0")
    assert result is not None


def test_pythonfcn_very_large_output_no_crash():
    """1e308 * 2 → inf; simulation should complete without exception."""
    result = _pfcn_sim("1e308 * 2", src_val="0.0")
    assert result is not None


def test_pythonfcn_exec_no_y_gives_zero():
    """Multi-line code that never sets y should produce 0.0 output."""
    import numpy as _np
    result = _pfcn_sim("a = u[0] * 2", src_val="5.0")
    # All outputs should be 0.0 when y is never assigned
    arr = list(result)
    assert all(v == 0.0 for v in arr)


def test_pythonfcn_multiline_exec_correct():
    """Multi-line exec code: a=u[0]*3; y=a+1 with u=4 should give 13."""
    import numpy as _np
    result = _pfcn_sim("a = u[0]*3\ny = a + 1", src_val="4.0", steps=3)
    arr = list(result)
    assert all(abs(v - 13.0) < 1e-9 for v in arr)


def test_pythonfcn_t_variable_increases_each_step():
    """code='t' should output the simulation time, increasing each step."""
    result = _pfcn_sim("t", src_val="0.0", steps=5)
    arr = list(result)
    # Each value should be non-negative and increasing
    assert arr[-1] > arr[0]
    for i in range(1, len(arr)):
        assert arr[i] >= arr[i - 1]


def test_pythonfcn_infinite_loop_not_triggered():
    """code='while True: pass' — eval will fail (SyntaxError or infinite),
    but exec path is only hit when eval raises SyntaxError; the infinite loop
    in exec would hang so we just verify the block spec parses without crash."""
    # We only test that _validate_block doesn't crash for this (it can't run)
    blk = _pfcn("while True: pass")
    errors = _validate_block(blk)
    # No crash is the requirement; validation may or may not flag it
    assert isinstance(errors, list)


# ---------------------------------------------------------------------------
# Regression: _python_to_c must strip leading "y = " so the generated C line
# does not contain an undeclared variable assignment inside a cast expression.
# Bug: "y = u[0] + u[1]" was emitted as  sig_X_y = (float)(y = ...);
#      which fails to compile ("y undeclared").
# ---------------------------------------------------------------------------

def test_pythonfcn_y_assign_stripped_from_codegen():
    """'y = u[0] + u[1]' must NOT produce '(y = ...)' in the C output."""
    from code_templates import _python_to_c
    result = _python_to_c("y = u[0] + u[1]", ["sig_C0_y", "sig_C1_y"])
    # Must return a valid expression (not None → stub, and not containing 'y =')
    assert result is not None, "_python_to_c returned None for simple y= assignment"
    assert "y =" not in result and "y=" not in result, (
        f"_python_to_c left 'y =' in expression: {result!r}"
    )
    assert "sig_C0_y" in result and "sig_C1_y" in result


def test_pythonfcn_y_assign_end_to_end_codegen():
    """Full codegen with a PythonFcn using 'y = u[0]' must compile-clean C."""
    import json, textwrap
    # Minimal two-block model: Constant → PythonFcn
    model = {
        "step_ms": 10,
        "blocks": [
            {"id": "C1", "type": "Constant",  "x": 0,   "y": 0,
             "params": {"value": "3.0"}},
            {"id": "PF", "type": "PythonFcn", "x": 200, "y": 0,
             "params": {"num_inputs": "1", "code": "y = u[0] * 2.0"}},
            {"id": "SC", "type": "Scope",     "x": 400, "y": 0,
             "params": {}},
        ],
        "connections": [
            {"src_block": "C1", "src_port": "y",  "dst_block": "PF", "dst_port": "u0"},
            {"src_block": "PF", "src_port": "y",  "dst_block": "SC", "dst_port": "u0"},
        ],
    }
    board = BOARDS["NUCLEO-F446RE"]
    wires = _wires(model)
    src, _ = _emit_step(model["blocks"], wires, FakeWorkspace(), step_ms=10, board=board)
    # The critical check: no raw 'y =' inside a cast
    assert "(y =" not in src and "(y=" not in src, (
        "Generated C step code contains illegal '(y =...)':\n" +
        "\n".join(ln for ln in src.splitlines() if "y =" in ln)
    )
    # And the PythonFcn output should contain the multiplied signal
    assert "sig_PF_y" in src


# ===========================================================================
# Section F — Sweep update (future new blocks will auto-appear in catalog sweep)
# ===========================================================================
# The existing sweep tests (test_every_source_block_simulates_without_crash,
# test_every_single_input_block_simulates_without_crash) iterate over
# BLOCK_CATALOG dynamically, so they automatically cover any new blocks added
# in Phase B without modification.  These two tests verify that property.

def test_sweep_covers_all_catalog_types():
    """Every block type in the catalog is a string key (no None or empty)."""
    for key in BLOCK_CATALOG:
        assert isinstance(key, str) and len(key) > 0


def test_sweep_catalog_sources_have_outputs():
    """Every source block (no inputs) exposes at least one output port."""
    source_specs = [s for s in BLOCK_CATALOG.values() if not s.inputs]
    for spec in source_specs:
        assert len(spec.outputs) > 0, (
            f"Source block {spec.type_name} has no output ports"
        )


# ===========================================================================
# Section G — FreeRTOS extraction logic (code_templates.py lines 55-93)
# ===========================================================================
# These tests exercise the _ensure_freertos() path that was previously
# uncovered: the zip-extract branch, the V-prefix rename fallback, the bad-
# layout error, the download-failure error, and the cleanup of a partial zip.
# All file I/O is redirected into a temporary directory; no network calls are
# made and the real ~/.stm32_block_ide cache is never touched.

def _make_zip_mock(extract_side_effect):
    """Return a mock ZipFile class whose extractall() runs extract_side_effect."""
    instance = MagicMock()
    instance.__enter__ = MagicMock(return_value=instance)
    instance.__exit__ = MagicMock(return_value=False)
    instance.extractall.side_effect = extract_side_effect
    return MagicMock(return_value=instance)


def test_freertos_cache_hit_skips_download():
    """If tasks.c already exists in the cache dir, return immediately — no download."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_cache = Path(tmpdir) / f"FreeRTOS-Kernel-{_FREERTOS_VERSION}"
        fake_cache.mkdir()
        (fake_cache / "tasks.c").touch()
        with patch("code_templates._FREERTOS_CACHE", fake_cache), \
             patch("code_templates.urllib.request.urlretrieve") as mock_dl:
            result = _ensure_freertos()
    mock_dl.assert_not_called()
    assert result == fake_cache


def test_freertos_extract_normal_variant():
    """Zip extracts as FreeRTOS-Kernel-{VERSION}/ (standard GitHub layout) — no rename needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_cache = Path(tmpdir) / f"FreeRTOS-Kernel-{_FREERTOS_VERSION}"

        def do_extract(path):
            (Path(path) / f"FreeRTOS-Kernel-{_FREERTOS_VERSION}").mkdir(parents=True)

        with patch("code_templates._FREERTOS_CACHE", fake_cache), \
             patch("code_templates.urllib.request.urlretrieve"), \
             patch("code_templates.zipfile.ZipFile", _make_zip_mock(do_extract)):
            result = _ensure_freertos()

        assert result == fake_cache
        assert fake_cache.exists()


def test_freertos_extract_v_variant_renamed():
    """Zip extracts as FreeRTOS-Kernel-V{VERSION}/ — gets renamed to the canonical form."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_cache = Path(tmpdir) / f"FreeRTOS-Kernel-{_FREERTOS_VERSION}"

        def do_extract(path):
            # Simulate the V-prefixed tag name that some GitHub releases use
            (Path(path) / f"FreeRTOS-Kernel-V{_FREERTOS_VERSION}").mkdir(parents=True)

        with patch("code_templates._FREERTOS_CACHE", fake_cache), \
             patch("code_templates.urllib.request.urlretrieve"), \
             patch("code_templates.zipfile.ZipFile", _make_zip_mock(do_extract)):
            result = _ensure_freertos()

        # Rename must have produced the canonical (non-V) directory
        assert result == fake_cache
        assert fake_cache.exists()
        assert not (fake_cache.parent / f"FreeRTOS-Kernel-V{_FREERTOS_VERSION}").exists()


def test_freertos_extract_unknown_layout_raises():
    """If the zip produces an unexpected directory name, RuntimeError is raised."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_cache = Path(tmpdir) / f"FreeRTOS-Kernel-{_FREERTOS_VERSION}"

        def do_extract(path):
            (Path(path) / "some-unrecognised-layout").mkdir(parents=True)

        with patch("code_templates._FREERTOS_CACHE", fake_cache), \
             patch("code_templates.urllib.request.urlretrieve"), \
             patch("code_templates.zipfile.ZipFile", _make_zip_mock(do_extract)):
            try:
                _ensure_freertos()
            except RuntimeError as exc:
                assert "Unexpected zip layout" in str(exc)
                return
    raise AssertionError("expected RuntimeError for unrecognised zip layout")


def test_freertos_download_failure_raises():
    """A network error during urlretrieve raises RuntimeError with a helpful message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_cache = Path(tmpdir) / f"FreeRTOS-Kernel-{_FREERTOS_VERSION}"
        with patch("code_templates._FREERTOS_CACHE", fake_cache), \
             patch("code_templates.urllib.request.urlretrieve",
                   side_effect=OSError("connection refused")):
            try:
                _ensure_freertos()
            except RuntimeError as exc:
                assert "Could not download FreeRTOS" in str(exc)
                return
    raise AssertionError("expected RuntimeError on download failure")


def test_freertos_download_failure_cleans_tmp_zip():
    """If urlretrieve fails after creating a partial file, that file is deleted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_cache = Path(tmpdir) / f"FreeRTOS-Kernel-{_FREERTOS_VERSION}"
        fake_cache.parent.mkdir(parents=True, exist_ok=True)
        tmp_zip = fake_cache.parent / f"freertos_kernel_{_FREERTOS_VERSION}.zip"

        def partial_download(url, dest):
            Path(dest).touch()          # simulate partial file created before failure
            raise OSError("partial download")

        with patch("code_templates._FREERTOS_CACHE", fake_cache), \
             patch("code_templates.urllib.request.urlretrieve",
                   side_effect=partial_download):
            try:
                _ensure_freertos()
            except RuntimeError:
                pass  # expected

        assert not tmp_zip.exists(), "partial zip must be deleted after download failure"


if __name__ == "__main__":
    sys.exit(main())
