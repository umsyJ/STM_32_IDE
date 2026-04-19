"""Unit tests for the STM32 Block IDE.

Covers:
- Block catalog integrity (every block has a spec, params are well-formed)
- Host-side simulator numerics (simulate_model)
- C-code generator internals (_topo_order, _emit_decls, _emit_init,
  _emit_step, _emit_helpers)
- End-to-end project generation (all files present, contents include
  expected symbols)
- Model JSON round-trip

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
    _emit_decls,
    _emit_helpers,
    _emit_init,
    _emit_step,
    _topo_order,
    _wires,
    generate_project,
)
from stm32_block_ide import BLOCK_CATALOG, simulate_model  # noqa: E402
from workspace_shared import WORKSPACE  # noqa: E402


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


def _wire(src_bid, src_port, dst_bid, dst_port):
    return {
        "src_block": src_bid, "src_port": src_port,
        "dst_block": dst_bid, "dst_port": dst_port,
    }


# ---------------------------------------------------------------------------
# 1. Block catalog integrity
# ---------------------------------------------------------------------------


def test_catalog_has_all_known_types():
    required = {"SquareWave", "GpioIn", "GpioOut", "Scope", "Ultrasonic"}
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
