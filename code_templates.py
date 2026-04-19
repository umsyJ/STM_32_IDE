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
        elif b["type"] == "GpioIn":
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        elif b["type"] == "Ultrasonic":
            decls.append(f"static float {_sig_var(b['id'],'d')} = 0.0f;")
        elif b["type"] in ("Sum", "Product"):
            decls.append(f"static float {_sig_var(b['id'],'y')} = 0.0f;")
        # GpioOut & Scope: no signal output
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
        if t == "SquareWave":
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
