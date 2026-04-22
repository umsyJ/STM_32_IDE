"""
STM32 Block IDE - A Simulink-like visual programming environment for STM32.

Main application module. Provides:
 - Block diagram editor (drag/drop blocks, wire them together)
 - Block parameter panel (values or workspace variable references)
 - Python workspace tab (MATLAB-like)
 - Scope / serial monitor tab
 - Board selector (NUCLEO-F446RE)
 - One-click build & flash (Play button)

Dependencies:
    pip install PyQt5 pyqtgraph pyserial numpy

Author: generated scaffold
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import (
    QLineF, QPointF, QRectF, Qt, QTimer, pyqtSignal, QObject, QThread,
)
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF,
    QTextCharFormat, QTextCursor,
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QComboBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
    QSplitter, QStatusBar, QTabWidget, QTextEdit, QToolBar, QVBoxLayout,
    QWidget, QCheckBox, QMenu,
)

import numpy as np

try:
    import pyqtgraph as pg
    HAVE_PYQTGRAPH = True
except Exception:
    HAVE_PYQTGRAPH = False

try:
    import serial
    import serial.tools.list_ports
    HAVE_SERIAL = True
except Exception:
    HAVE_SERIAL = False


from code_templates import generate_project, BOARDS, _topo_order as _ct_topo_order, _bilinear_tf
from workspace_shared import WORKSPACE
from matlab_workspace import MatlabWorkspace


APP_NAME = "STM32 Block IDE"
VERSION = "0.2.1"


# ---------------------------------------------------------------------------
# Block model
# ---------------------------------------------------------------------------

@dataclass
class PortSpec:
    name: str
    direction: str  # "in" or "out"


@dataclass
class BlockSpec:
    """Static description of a block type."""

    type_name: str
    display_name: str
    color: str
    inputs: List[PortSpec] = field(default_factory=list)
    outputs: List[PortSpec] = field(default_factory=list)
    # param name -> (default_value, help_text)
    params: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    description: str = ""


# Catalog of supported blocks. Each block's C codegen lives in code_templates.py
BLOCK_CATALOG: Dict[str, BlockSpec] = {
    "SquareWave": BlockSpec(
        type_name="SquareWave",
        display_name="Square Wave",
        color="#4a90e2",
        outputs=[PortSpec("y", "out")],
        params={
            "frequency_hz": ("1.0", "Frequency in Hz (may be a workspace variable)"),
            "amplitude":    ("1.0", "Output high value"),
            "offset":       ("0.0", "Output low value"),
            "duty":         ("0.5", "Duty cycle 0..1"),
        },
        description=(
            "Generates a square wave. Executed at the model step rate on the "
            "MCU. Output is a floating-point sample that downstream blocks "
            "consume as threshold logic or analog cues."
        ),
    ),
    "GpioIn": BlockSpec(
        type_name="GpioIn",
        display_name="GPIO In",
        color="#7ed321",
        outputs=[PortSpec("y", "out")],
        params={
            "pin":      ("PC13", "Board pin identifier, e.g. PC13 (user button on F446RE)"),
            "pull":     ("none", "Pull resistor: 'none', 'up', or 'down'"),
            "active_low": ("1", "1 = pressed is logic low"),
        },
        description=(
            "Reads a digital input pin each model step. Output is 0 or 1. "
            "Use for buttons, sensors, external logic levels."
        ),
    ),
    "GpioOut": BlockSpec(
        type_name="GpioOut",
        display_name="GPIO Out",
        color="#f5a623",
        inputs=[PortSpec("u", "in")],
        params={
            "pin":       ("PA5", "Board pin identifier, e.g. PA5 (green LED on F446RE)"),
            "threshold": ("0.5", "Input > threshold drives pin high"),
        },
        description=(
            "Drives a digital output pin each model step. Input is compared "
            "against a threshold to produce logic high/low."
        ),
    ),
    "Scope": BlockSpec(
        type_name="Scope",
        display_name="Scope",
        color="#bd10e0",
        inputs=[PortSpec("u0", "in"), PortSpec("u1", "in"), PortSpec("u2", "in")],
        params={
            "max_points": ("500", "Max samples kept in the scope view"),
            "stream":     ("1", "1 = stream this signal out over serial to host"),
        },
        description=(
            "Displays connected signals. When 'stream' is enabled the MCU "
            "sends samples over USART2 (115200 8-N-1) that the host scope "
            "tab visualises in real time."
        ),
    ),
    "Ultrasonic": BlockSpec(
        type_name="Ultrasonic",
        display_name="Ultrasonic (HC-SR04)",
        color="#00b8a9",
        outputs=[PortSpec("d", "out")],
        params={
            "trig_pin":   ("PA0",   "Trigger output pin. Wired to HC-SR04 TRIG."),
            "echo_pin":   ("PA1",   "Echo input pin. Wired to HC-SR04 ECHO "
                                    "(through a level shifter / divider - "
                                    "ECHO is 5V, STM32 pins are 3.3V tolerant only)."),
            "period_ms":  ("60",    "How often to fire a measurement, in ms. "
                                    ">= 50 ms is recommended per the HC-SR04 datasheet."),
            "timeout_us": ("30000", "Echo timeout in microseconds. ~30000 us "
                                    "corresponds to a max range of about 5 m."),
        },
        description=(
            "Reads an HC-SR04 ultrasonic range finder. Sends a 10 us trigger "
            "pulse on TRIG and times the ECHO pulse width with the DWT cycle "
            "counter. Output 'd' is distance in meters; 0 on timeout / no echo."
        ),
    ),
    "Constant": BlockSpec(
        type_name="Constant",
        display_name="Constant",
        color="#c0392b",
        outputs=[PortSpec("y", "out")],
        params={
            "value": ("1.0", "Constant output value (may be a workspace expression)"),
        },
        description=(
            "Outputs a fixed float value every model step. "
            "Use to set thresholds, scale factors, or any fixed signal."
        ),
    ),
    "Sum": BlockSpec(
        type_name="Sum",
        display_name="Sum",
        color="#e74c3c",
        inputs=[PortSpec("u0", "in"), PortSpec("u1", "in")],
        outputs=[PortSpec("y", "out")],
        params={},
        description=(
            "Adds two input signals each model step. Unconnected inputs "
            "contribute 0.0. Output 'y' = u0 + u1."
        ),
    ),
    "Product": BlockSpec(
        type_name="Product",
        display_name="Product",
        color="#8e44ad",
        inputs=[PortSpec("u0", "in"), PortSpec("u1", "in")],
        outputs=[PortSpec("y", "out")],
        params={},
        description=(
            "Multiplies two input signals each model step. Unconnected inputs "
            "contribute 1.0. Output 'y' = u0 * u1."
        ),
    ),
    "Step": BlockSpec(
        type_name="Step",
        display_name="Step",
        color="#e67e22",
        outputs=[PortSpec("y", "out")],
        params={
            "step_time":     ("1.0", "Time (s) at which the step occurs"),
            "initial_value": ("0.0", "Output before the step"),
            "final_value":   ("1.0", "Output at and after the step"),
        },
        description=(
            "Outputs initial_value while t < step_time, then final_value. "
            "On the MCU, time is tracked by counting model steps. "
            "Equivalent to Simulink's Step source block."
        ),
    ),
    "Integrator": BlockSpec(
        type_name="Integrator",
        display_name="Integrator",
        color="#2980b9",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "initial_value": ("0.0",   "Initial condition at t = 0"),
            "upper_limit":   ("1e10",  "Upper saturation limit"),
            "lower_limit":   ("-1e10", "Lower saturation limit"),
        },
        description=(
            "Discrete integrator using forward Euler: y[k] = y[k−1] + u[k−1]·dt. "
            "Output is clamped to [lower_limit, upper_limit] each step. "
            "Set initial_value to a non-zero IC if needed."
        ),
    ),
    "TransferFcn": BlockSpec(
        type_name="TransferFcn",
        display_name="Transfer Fcn",
        color="#6c3483",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "numerator":   ("1",   "s-domain numerator coefficients, highest power first "
                                   "(e.g. '1 2' → s+2)"),
            "denominator": ("1 1", "s-domain denominator coefficients, highest power first "
                                   "(e.g. '1 10' → s+10)"),
        },
        description=(
            "Implements a continuous-time transfer function discretized via "
            "the bilinear (Tustin) transform at the model step rate. "
            "Enter polynomials as space-separated coefficients, highest power "
            "first: '1 3 2' means s²+3s+2. "
            "The system must be proper (numerator degree ≤ denominator degree). "
            "Internally generates a Direct-Form-II-Transposed IIR filter."
        ),
    ),
    "PID": BlockSpec(
        type_name="PID",
        display_name="PID Controller",
        color="#17a589",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "Kp":          ("1.0",   "Proportional gain"),
            "Ki":          ("0.0",   "Integral gain"),
            "Kd":          ("0.0",   "Derivative gain"),
            "N":           ("100.0", "Derivative filter coefficient (rad/s). "
                                     "Must be < 2/step_s for stability."),
            "upper_limit": ("1e10",  "Output upper saturation limit"),
            "lower_limit": ("-1e10", "Output lower saturation limit"),
        },
        description=(
            "Parallel-form PID with a first-order derivative filter: "
            "y = Kp·e + Ki·∫e dt + Kd·N·s/(s+N)·e, "
            "discretized with forward Euler. "
            "N is the derivative filter bandwidth in rad/s (higher = closer "
            "to ideal differentiation). Output is clamped to the saturation "
            "limits. Set Ki=0 or Kd=0 to get PD or PI controllers."
        ),
    ),
    "ToWorkspace": BlockSpec(
        type_name="ToWorkspace",
        display_name="To Workspace",
        color="#16a085",
        inputs=[PortSpec("u", "in")],
        params={
            "variable_name": ("yout",   "Python workspace variable to write (e.g. 'yout')"),
            "max_points":    ("10000",  "Maximum samples stored (oldest are dropped)"),
            "decimation":    ("1",      "Store every Nth sample (1 = every sample)"),
            "save_time":     ("1",      "1 = also write a '<name>_t' time vector"),
        },
        description=(
            "Captures a signal during host simulation and writes it to the "
            "Python workspace as a NumPy array named by 'variable_name'. "
            "A matching time vector '<name>_t' is also written when "
            "'save_time' is 1. Use these arrays in the Python Workspace tab "
            "for post-processing, FFT, or plotting.\n\n"
            "On the MCU this block produces no code — it is a simulation-only "
            "sink, analogous to Simulink's To Workspace block."
        ),
    ),

    # ---- Group A: Sources --------------------------------------------------

    "SineWave": BlockSpec(
        type_name="SineWave",
        display_name="Sine Wave",
        color="#1a5276",
        outputs=[PortSpec("y", "out")],
        params={
            "frequency_hz": ("1.0",  "Frequency in Hz"),
            "amplitude":    ("1.0",  "Peak amplitude"),
            "phase_deg":    ("0.0",  "Phase offset in degrees"),
            "offset":       ("0.0",  "DC offset added to output"),
        },
        description=(
            "Generates A*sin(2*pi*f*t + phi) + offset. "
            "Phase is specified in degrees. "
            "Frequency must be > 0 Hz."
        ),
    ),

    "Ramp": BlockSpec(
        type_name="Ramp",
        display_name="Ramp",
        color="#1a5276",
        outputs=[PortSpec("y", "out")],
        params={
            "slope":          ("1.0", "Rate of change (units/second)"),
            "start_time":     ("0.0", "Time at which ramp begins (s)"),
            "initial_output": ("0.0", "Output before and at start_time"),
        },
        description=(
            "Outputs initial_output while t < start_time, then "
            "initial_output + slope*(t - start_time) after. "
            "Equivalent to Simulink's Ramp source block."
        ),
    ),

    "Clock": BlockSpec(
        type_name="Clock",
        display_name="Clock",
        color="#1a5276",
        outputs=[PortSpec("y", "out")],
        params={},
        description=(
            "Outputs the current simulation time t in seconds. "
            "On the MCU, time is tracked by counting model steps multiplied "
            "by the step period."
        ),
    ),

    "PulseGenerator": BlockSpec(
        type_name="PulseGenerator",
        display_name="Pulse Generator",
        color="#1a5276",
        outputs=[PortSpec("y", "out")],
        params={
            "amplitude":   ("1.0",  "High-level output value"),
            "period":      ("1.0",  "Pulse period in seconds (> 0)"),
            "pulse_width": ("50",   "Pulse width as % of period (0-100)"),
            "phase_delay": ("0.0",  "Delay before first pulse (s)"),
        },
        description=(
            "Generates a periodic rectangular pulse. "
            "pulse_width is a percentage of the period (0-100). "
            "Output is amplitude during the high phase, 0 otherwise."
        ),
    ),

    # ---- Group B: Math -----------------------------------------------------

    "Gain": BlockSpec(
        type_name="Gain",
        display_name="Gain",
        color="#e67e22",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "gain": ("1.0", "Scalar gain factor applied to input"),
        },
        description="Multiplies input u by gain: y = gain * u.",
    ),

    "Abs": BlockSpec(
        type_name="Abs",
        display_name="Abs",
        color="#e67e22",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={},
        description="Absolute value: y = |u|.",
    ),

    "Sign": BlockSpec(
        type_name="Sign",
        display_name="Sign",
        color="#e67e22",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={},
        description="Sign function: y = 1 if u > 0, -1 if u < 0, 0 if u == 0.",
    ),

    "Sqrt": BlockSpec(
        type_name="Sqrt",
        display_name="Sqrt",
        color="#e67e22",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "mode": ("sqrt", "sqrt: y=sqrt(|u|); signed_sqrt: y=sign(u)*sqrt(|u|)"),
        },
        description=(
            "Square root block. mode='sqrt': y = sqrt(|u|). "
            "mode='signed_sqrt': y = sign(u)*sqrt(|u|)."
        ),
    ),

    "Saturation": BlockSpec(
        type_name="Saturation",
        display_name="Saturation",
        color="#8e44ad",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "upper_limit": ("1.0",  "Maximum output value"),
            "lower_limit": ("-1.0", "Minimum output value"),
        },
        description="Clips input to [lower_limit, upper_limit]: y = clip(u, lower, upper).",
    ),

    "DeadZone": BlockSpec(
        type_name="DeadZone",
        display_name="Dead Zone",
        color="#8e44ad",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "lower_value": ("-0.5", "Lower dead-zone threshold"),
            "upper_value": ("0.5",  "Upper dead-zone threshold"),
        },
        description=(
            "Dead zone nonlinearity: y = u - upper if u > upper; "
            "y = u - lower if u < lower; y = 0 inside the zone."
        ),
    ),

    "MinMax": BlockSpec(
        type_name="MinMax",
        display_name="MinMax",
        color="#e67e22",
        inputs=[PortSpec("u0", "in"), PortSpec("u1", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "function": ("min", "min or max"),
        },
        description="Outputs min(u0,u1) or max(u0,u1) depending on function parameter.",
    ),

    # ---- Group C: Logic ----------------------------------------------------

    "RelationalOperator": BlockSpec(
        type_name="RelationalOperator",
        display_name="Relational Op",
        color="#27ae60",
        inputs=[PortSpec("u0", "in"), PortSpec("u1", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "operator": (">", "Comparison: >, <, >=, <=, ==, !="),
        },
        description=(
            "Compares u0 and u1 using the selected operator. "
            "Output y = 1.0 if true, 0.0 if false."
        ),
    ),

    "LogicalOperator": BlockSpec(
        type_name="LogicalOperator",
        display_name="Logical Op",
        color="#27ae60",
        inputs=[PortSpec("u0", "in"), PortSpec("u1", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "operator": ("AND", "Boolean operation: AND, OR, NOT, NAND, NOR, XOR"),
        },
        description=(
            "Boolean logic gate. NOT uses only u0. "
            "Inputs treated as boolean (nonzero = true). "
            "Output y = 1.0 or 0.0."
        ),
    ),

    "Switch": BlockSpec(
        type_name="Switch",
        display_name="Switch",
        color="#27ae60",
        inputs=[PortSpec("u0", "in"), PortSpec("u1", "in"), PortSpec("u2", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "threshold": ("0.5", "Threshold value for control input u1"),
            "criteria":  (">=",  "Comparison for u1 vs threshold: >, >=, =="),
        },
        description=(
            "Three-input switch: y = u0 if u1 criteria threshold else u2. "
            "u1 is the control input, u0 and u2 are data inputs."
        ),
    ),

    # ---- Group D: Discrete -------------------------------------------------

    "UnitDelay": BlockSpec(
        type_name="UnitDelay",
        display_name="Unit Delay",
        color="#2c3e50",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "initial_condition": ("0.0", "Output at first sample (k=0)"),
        },
        description="One-sample delay: y[k] = u[k-1]. Also known as z^-1.",
    ),

    "DiscreteIntegrator": BlockSpec(
        type_name="DiscreteIntegrator",
        display_name="Discrete Integrator",
        color="#2c3e50",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "gain_value":        ("1.0",   "Integrator gain K"),
            "initial_condition": ("0.0",   "Initial state"),
            "upper_limit":       ("1e10",  "Upper saturation limit"),
            "lower_limit":       ("-1e10", "Lower saturation limit"),
            "method":            ("Forward Euler", "Integration method: Forward Euler, Backward Euler, Trapezoidal"),
        },
        description=(
            "Discrete-time integrator: y[k] = clip(y[k-1] + K*u[k]*Ts, lower, upper). "
            "Supports Forward Euler, Backward Euler, and Trapezoidal methods."
        ),
    ),

    "ZeroOrderHold": BlockSpec(
        type_name="ZeroOrderHold",
        display_name="Zero-Order Hold",
        color="#2c3e50",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "sample_time": ("0.01", "Hold sample time in seconds (>= step time)"),
        },
        description=(
            "Samples input every sample_time seconds and holds the value "
            "between samples. Useful for rate transitions."
        ),
    ),

    "Derivative": BlockSpec(
        type_name="Derivative",
        display_name="Derivative",
        color="#2c3e50",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "initial_condition": ("0.0", "Assumed previous input value at k=0"),
        },
        description=(
            "Numerical derivative: y[k] = (u[k] - u[k-1]) / Ts. "
            "The initial_condition sets the assumed u[-1]."
        ),
    ),

    # ---- Group E: Lookup ---------------------------------------------------

    "Lookup1D": BlockSpec(
        type_name="Lookup1D",
        display_name="1-D Lookup",
        color="#f39c12",
        inputs=[PortSpec("u", "in")],
        outputs=[PortSpec("y", "out")],
        params={
            "breakpoints":    ("0 1",  "Space-separated breakpoint values (strictly increasing)"),
            "table_data":     ("0 1",  "Space-separated output values (same length as breakpoints)"),
            "extrapolation":  ("clip", "Extrapolation mode: clip or linear"),
        },
        description=(
            "Piecewise-linear 1-D lookup table. "
            "Interpolates between breakpoints; clips or extrapolates at boundaries."
        ),
    ),

    # ---- Group F: STM32 HAL ------------------------------------------------

    "ADC": BlockSpec(
        type_name="ADC",
        display_name="ADC",
        color="#00b8a9",
        outputs=[PortSpec("y", "out")],
        params={
            "channel":    ("1",   "ADC channel number (1-16)"),
            "resolution": ("12",  "ADC resolution bits: 6, 8, 10, or 12"),
            "vref":       ("3.3", "Reference voltage in volts (> 0)"),
            "sim_value":  ("0.0", "Constant voltage used during host simulation"),
        },
        description=(
            "Reads an ADC channel and outputs the voltage in volts. "
            "In host simulation, outputs the constant sim_value."
        ),
    ),

    "DAC": BlockSpec(
        type_name="DAC",
        display_name="DAC",
        color="#00b8a9",
        inputs=[PortSpec("u", "in")],
        params={
            "channel": ("1",   "DAC channel: 1 or 2"),
            "vref":    ("3.3", "Reference voltage in volts (> 0)"),
        },
        description=(
            "Converts input voltage to DAC output. "
            "Input u is clamped to [0, vref] before conversion."
        ),
    ),

    "PWMOut": BlockSpec(
        type_name="PWMOut",
        display_name="PWM Out",
        color="#00b8a9",
        inputs=[PortSpec("u", "in")],
        params={
            "timer":        ("TIM2",   "HAL timer handle name (e.g. TIM2)"),
            "channel":      ("1",      "Timer channel number (1-4)"),
            "frequency_hz": ("1000",   "PWM frequency in Hz (> 0)"),
            "max_duty":     ("100.0",  "Maximum duty cycle value (> 0)"),
        },
        description=(
            "Sets PWM duty cycle from input u. "
            "u is clamped to [0, max_duty] before setting the compare register."
        ),
    ),

    "TimerTick": BlockSpec(
        type_name="TimerTick",
        display_name="Timer Tick",
        color="#00b8a9",
        outputs=[PortSpec("y", "out")],
        params={
            "scale": ("0.001", "Scale factor: y = HAL_GetTick() * scale"),
        },
        description=(
            "Outputs HAL_GetTick() * scale. "
            "Default scale=0.001 converts milliseconds to seconds. "
            "In simulation, y = t * 1000 * scale."
        ),
    ),
}


@dataclass
class BlockInstance:
    """A placed block on the canvas."""

    spec: BlockSpec
    block_id: str
    x: float
    y: float
    params: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.spec.type_name,
            "id": self.block_id,
            "x": self.x, "y": self.y,
            "params": self.params,
        }


@dataclass
class Connection:
    src_block: str
    src_port: str
    dst_block: str
    dst_port: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ValidationError:
    """A parameter validation problem on a specific block."""
    block_id:   str   # which block instance
    block_type: str   # e.g. "SquareWave"
    param:      str   # which parameter field
    code:       str   # short code, e.g. "E001"
    message:    str   # human-readable description

    def __str__(self) -> str:
        return f"[{self.code}] {self.block_id} · {self.param}: {self.message}"


# ---------------------------------------------------------------------------
# Graphics items
# ---------------------------------------------------------------------------

PORT_RADIUS = 6.0


class PortItem(QGraphicsRectItem):
    def __init__(self, parent: "BlockItem", name: str, direction: str, index: int):
        super().__init__(-PORT_RADIUS, -PORT_RADIUS, 2*PORT_RADIUS, 2*PORT_RADIUS, parent)
        self.setBrush(QBrush(QColor("#222")))
        self.setPen(QPen(QColor("#eee"), 1))
        self.block = parent
        self.port_name = name
        self.direction = direction
        self.index = index
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip(f"{'▶ out' if direction == 'out' else '◀ in'}: {name}")

    def hoverEnterEvent(self, ev):
        self.setBrush(QBrush(QColor("#ffd54a")))
        self.setRect(-PORT_RADIUS * 1.5, -PORT_RADIUS * 1.5,
                     PORT_RADIUS * 3, PORT_RADIUS * 3)

    def hoverLeaveEvent(self, ev):
        self.setBrush(QBrush(QColor("#222")))
        self.setRect(-PORT_RADIUS, -PORT_RADIUS, PORT_RADIUS * 2, PORT_RADIUS * 2)

    def highlight_as_target(self, on: bool) -> None:
        self.setBrush(QBrush(QColor("#00e676") if on else QColor("#222")))

    def scenePos(self) -> QPointF:  # override for clarity
        return self.mapToScene(self.rect().center())


class BlockItem(QGraphicsRectItem):
    """Visual representation of a block."""

    WIDTH = 150.0
    HEIGHT = 80.0

    def __init__(self, instance: BlockInstance):
        super().__init__(0, 0, self.WIDTH, self.HEIGHT)
        self.instance = instance
        self.setBrush(QBrush(QColor(instance.spec.color)))
        self.setPen(QPen(QColor("#1b1b1b"), 1.5))
        self._has_error: bool = False
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setPos(instance.x, instance.y)
        self._make_ports()

    def set_error_highlight(self, on: bool) -> None:
        """Mark or unmark the block as having a validation error."""
        self._has_error = on
        if on:
            self.setPen(QPen(QColor("#ff4444"), 3))
        else:
            self.setPen(QPen(QColor("#1b1b1b"), 1.5))
        self.update()

    def _make_ports(self) -> None:
        self.input_ports: List[PortItem] = []
        self.output_ports: List[PortItem] = []
        ins = self.instance.spec.inputs
        outs = self.instance.spec.outputs
        for i, p in enumerate(ins):
            y = (i + 1) * self.HEIGHT / (len(ins) + 1)
            port = PortItem(self, p.name, "in", i)
            port.setPos(0, y)
            self.input_ports.append(port)
        for i, p in enumerate(outs):
            y = (i + 1) * self.HEIGHT / (len(outs) + 1)
            port = PortItem(self, p.name, "out", i)
            port.setPos(self.WIDTH, y)
            self.output_ports.append(port)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        painter.setPen(QPen(QColor("white")))
        f = QFont(); f.setBold(True); f.setPointSize(10)
        painter.setFont(f)
        # Shrink the name area slightly when an error badge is showing
        name_right_margin = -24 if self._has_error else -5
        painter.drawText(
            self.rect().adjusted(5, 5, name_right_margin, -self.HEIGHT/2),
            Qt.AlignLeft | Qt.AlignTop, self.instance.spec.display_name,
        )
        f2 = QFont(); f2.setPointSize(8)
        painter.setFont(f2)
        painter.drawText(
            self.rect().adjusted(5, self.HEIGHT/2, -5, -5),
            Qt.AlignLeft | Qt.AlignBottom, f"#{self.instance.block_id}",
        )
        # Error badge: red circle with "!" in the top-right corner
        if self._has_error:
            badge = QRectF(self.WIDTH - 22, 4, 18, 18)
            painter.setBrush(QBrush(QColor("#ff3333")))
            painter.setPen(QPen(QColor("#cc0000"), 1))
            painter.drawEllipse(badge)
            fb = QFont(); fb.setBold(True); fb.setPointSize(9)
            painter.setFont(fb)
            painter.setPen(QPen(QColor("white")))
            painter.drawText(badge, Qt.AlignCenter, "!")

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.instance.x = self.pos().x()
            self.instance.y = self.pos().y()
            if self.scene() is not None:
                for conn in getattr(self.scene(), "connection_items", []):
                    conn.update_path()
        return super().itemChange(change, value)


class ConnectionItem(QGraphicsPathItem):
    """Bezier wire between two ports."""

    _PEN_NORMAL   = QPen(QColor("#eeeeee"), 2)
    _PEN_HOVER    = QPen(QColor("#ffd54a"), 3)
    _PEN_SELECTED = QPen(QColor("#ff4444"), 3)

    def __init__(self, src_port: PortItem, dst_port: PortItem):
        super().__init__()
        self.src = src_port
        self.dst = dst_port
        self.setPen(self._PEN_NORMAL)
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.update_path()

    def update_path(self) -> None:
        p1 = self.src.scenePos()
        p2 = self.dst.scenePos()
        dx = max(40.0, abs(p2.x() - p1.x()) * 0.5)
        path = QPainterPath(p1)
        path.cubicTo(
            QPointF(p1.x() + dx, p1.y()),
            QPointF(p2.x() - dx, p2.y()),
            p2,
        )
        self.setPath(path)

    def shape(self):
        # Widen the hit area so the wire is easy to click.
        stroker = QPainterPath()
        from PyQt5.QtGui import QPainterPathStroker
        ps = QPainterPathStroker()
        ps.setWidth(12)
        return ps.createStroke(self.path())

    def hoverEnterEvent(self, ev):
        if not self.isSelected():
            self.setPen(self._PEN_HOVER)

    def hoverLeaveEvent(self, ev):
        if not self.isSelected():
            self.setPen(self._PEN_NORMAL)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            self.setPen(self._PEN_SELECTED if value else self._PEN_NORMAL)
        return super().itemChange(change, value)


# ---------------------------------------------------------------------------
# Block diagram scene
# ---------------------------------------------------------------------------

class BlockScene(QGraphicsScene):

    block_selected = pyqtSignal(object)  # BlockItem or None

    def __init__(self) -> None:
        super().__init__()
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.blocks: Dict[str, BlockItem] = {}
        self.connections: List[Connection] = []
        self.connection_items: List[ConnectionItem] = []
        self._pending_src: Optional[PortItem] = None
        self._drag_wire: Optional[QGraphicsPathItem] = None
        self._highlighted_port: Optional[PortItem] = None
        self._id_counter = 1

    # --- block management -------------------------------------------------

    def next_id(self, type_name: str) -> str:
        bid = f"{type_name}_{self._id_counter}"
        self._id_counter += 1
        return bid

    def add_block_by_type(self, type_name: str, pos: QPointF) -> BlockItem:
        spec = BLOCK_CATALOG[type_name]
        inst = BlockInstance(
            spec=spec,
            block_id=self.next_id(type_name),
            x=pos.x(), y=pos.y(),
            params={k: v[0] for k, v in spec.params.items()},
        )
        item = BlockItem(inst)
        self.addItem(item)
        self.blocks[inst.block_id] = item
        return item

    def remove_block(self, item: BlockItem) -> None:
        bid = item.instance.block_id
        # remove connections touching it
        kept = []
        for c, ci in list(zip(self.connections, self.connection_items)):
            if c.src_block == bid or c.dst_block == bid:
                self.removeItem(ci)
            else:
                kept.append((c, ci))
        self.connections = [c for c, _ in kept]
        self.connection_items = [ci for _, ci in kept]
        self.removeItem(item)
        self.blocks.pop(bid, None)

    def add_connection(self, src: PortItem, dst: PortItem) -> None:
        # Disallow same block, same direction
        if src.block is dst.block:
            return
        if src.direction == dst.direction:
            return
        # Normalize so source is an output
        if src.direction == "in":
            src, dst = dst, src
        # Remove any existing connection into the destination input
        for i, c in list(enumerate(self.connections)):
            if c.dst_block == dst.block.instance.block_id and c.dst_port == dst.port_name:
                self.removeItem(self.connection_items[i])
                del self.connections[i]
                del self.connection_items[i]
        conn = Connection(
            src_block=src.block.instance.block_id, src_port=src.port_name,
            dst_block=dst.block.instance.block_id, dst_port=dst.port_name,
        )
        item = ConnectionItem(src, dst)
        self.addItem(item)
        self.connections.append(conn)
        self.connection_items.append(item)

    # --- mouse handling (wiring + selection) ------------------------------

    def _is_valid_target(self, src: PortItem, dst: PortItem) -> bool:
        return dst.block is not src.block and dst.direction != src.direction

    def _set_highlighted_port(self, port: Optional[PortItem]) -> None:
        if self._highlighted_port is port:
            return
        if self._highlighted_port is not None:
            self._highlighted_port.highlight_as_target(False)
        self._highlighted_port = port
        if port is not None:
            port.highlight_as_target(True)

    def _update_drag_wire(self, cursor: QPointF) -> None:
        if self._drag_wire is None or self._pending_src is None:
            return
        p1 = self._pending_src.scenePos()
        p2 = cursor
        dx = max(40.0, abs(p2.x() - p1.x()) * 0.5)
        path = QPainterPath(p1)
        path.cubicTo(
            QPointF(p1.x() + dx, p1.y()),
            QPointF(p2.x() - dx, p2.y()),
            p2,
        )
        self._drag_wire.setPath(path)

    def _cancel_drag(self) -> None:
        if self._drag_wire is not None:
            self.removeItem(self._drag_wire)
            self._drag_wire = None
        self._set_highlighted_port(None)
        self._pending_src = None

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if isinstance(item, PortItem) and event.button() == Qt.LeftButton:
            self._pending_src = item
            wire = QGraphicsPathItem()
            wire.setPen(QPen(QColor("#ffd54a"), 2, Qt.DashLine))
            wire.setZValue(10)
            self.addItem(wire)
            self._drag_wire = wire
            self._update_drag_wire(event.scenePos())
            return
        super().mousePressEvent(event)
        sel = self.selectedItems()
        if sel and isinstance(sel[0], BlockItem):
            self.block_selected.emit(sel[0])
        else:
            self.block_selected.emit(None)

    def _port_at(self, scene_pos: QPointF) -> Optional[PortItem]:
        """Return the topmost PortItem at scene_pos, ignoring the drag wire."""
        for item in self.items(scene_pos):
            if isinstance(item, PortItem):
                return item
        return None

    def mouseMoveEvent(self, event):
        if self._pending_src is not None:
            self._update_drag_wire(event.scenePos())
            port = self._port_at(event.scenePos())
            if port is not None and self._is_valid_target(self._pending_src, port):
                self._set_highlighted_port(port)
            else:
                self._set_highlighted_port(None)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pending_src is not None:
            port = self._port_at(event.scenePos())
            src = self._pending_src
            self._cancel_drag()
            if port is not None and port is not src:
                self.add_connection(src, port)
            return
        super().mouseReleaseEvent(event)

    def remove_connection_item(self, ci: "ConnectionItem") -> None:
        idx = self.connection_items.index(ci)
        self.removeItem(ci)
        del self.connections[idx]
        del self.connection_items[idx]

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for it in list(self.selectedItems()):
                if isinstance(it, BlockItem):
                    self.remove_block(it)
                elif isinstance(it, ConnectionItem):
                    self.remove_connection_item(it)
            return
        super().keyPressEvent(event)

    # --- serialization ----------------------------------------------------

    def to_model(self) -> dict:
        return {
            "blocks": [b.instance.to_dict() for b in self.blocks.values()],
            "connections": [c.to_dict() for c in self.connections],
        }


class BlockView(QGraphicsView):
    def __init__(self, scene: BlockScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        type_name = event.mimeData().text()
        if type_name in BLOCK_CATALOG:
            self.scene().add_block_by_type(type_name, self.mapToScene(event.pos()))
            event.acceptProposedAction()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Palette dock (draggable list of block types)
# ---------------------------------------------------------------------------

class BlockPalette(QListWidget):
    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        for spec in BLOCK_CATALOG.values():
            item = QListWidgetItem(spec.display_name)
            item.setData(Qt.UserRole, spec.type_name)
            self.addItem(item)

    def mimeData(self, items):
        md = super().mimeData(items)
        if items:
            md.setText(items[0].data(Qt.UserRole))
        return md


# ---------------------------------------------------------------------------
# Parameter editor dock
# ---------------------------------------------------------------------------

class ParamPanel(QWidget):

    # Emitted when the user clicks an error row; carries the block_id.
    error_block_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._block: Optional[BlockItem] = None
        self.layout_ = QVBoxLayout(self)
        self.layout_.setSpacing(4)

        self.title = QLabel("<i>(no block selected)</i>")
        self.title.setTextFormat(Qt.RichText)
        self.layout_.addWidget(self.title)

        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.layout_.addWidget(self.form_host)

        self.help = QLabel("")
        self.help.setWordWrap(True)
        self.help.setStyleSheet("color: #888; font-size: 11px;")
        self.layout_.addWidget(self.help)

        self.layout_.addStretch(1)
        self._fields: Dict[str, QLineEdit] = {}

        # ---- Error panel (shown only when there are validation errors) ----
        self._error_frame = QFrame()
        self._error_frame.setFrameShape(QFrame.StyledPanel)
        self._error_frame.setStyleSheet(
            "QFrame { background: #2d0a0a; border: 1px solid #aa2222;"
            " border-radius: 4px; }"
        )
        _ef_layout = QVBoxLayout(self._error_frame)
        _ef_layout.setContentsMargins(6, 6, 6, 6)
        _ef_layout.setSpacing(4)

        _err_hdr = QHBoxLayout()
        _err_icon = QLabel("⚠")
        _err_icon.setStyleSheet("color: #ff6b6b; font-size: 14px;")
        _err_hdr.addWidget(_err_icon)
        _err_title = QLabel("Simulation Errors")
        _err_title.setStyleSheet(
            "color: #ff6b6b; font-weight: bold; font-size: 12px;"
        )
        _err_hdr.addWidget(_err_title)
        _err_hdr.addStretch()
        self._err_count_label = QLabel("")
        self._err_count_label.setStyleSheet("color: #ff9999; font-size: 11px;")
        _err_hdr.addWidget(self._err_count_label)
        _ef_layout.addLayout(_err_hdr)

        self._error_list = QListWidget()
        self._error_list.setStyleSheet(
            "QListWidget { background: transparent; border: none;"
            " color: #ffaaaa; font-size: 11px; }"
            "QListWidget::item { padding: 4px 2px; border-bottom: 1px solid #441111; }"
            "QListWidget::item:selected { background: #551111; color: #ffdddd; }"
            "QListWidget::item:hover { background: #3a0d0d; }"
        )
        self._error_list.setSelectionMode(QListWidget.SingleSelection)
        self._error_list.setWordWrap(True)
        self._error_list.itemClicked.connect(self._on_error_clicked)
        self._error_list.setMaximumHeight(220)
        _ef_layout.addWidget(self._error_list)

        _hint = QLabel("Click an error to highlight the block")
        _hint.setStyleSheet("color: #884444; font-size: 10px; font-style: italic;")
        _ef_layout.addWidget(_hint)

        self.layout_.addWidget(self._error_frame)
        self._error_frame.setVisible(False)
        self._error_block_ids: List[str] = []   # parallel to _error_list rows

    # ------------------------------------------------------------------
    # Error panel helpers
    # ------------------------------------------------------------------

    def show_errors(self, errors: List) -> None:
        """Populate and reveal the error panel with the given ValidationErrors."""
        self._error_list.clear()
        self._error_block_ids.clear()
        for err in errors:
            # Two-line display: code + location on first line, message on second
            text = f"{err.code}  {err.block_id} · {err.param}\n    {err.message}"
            item = QListWidgetItem(text)
            item.setToolTip(
                f"Block : {err.block_id}  ({err.block_type})\n"
                f"Param : {err.param}\n"
                f"Code  : {err.code}\n"
                f"Error : {err.message}"
            )
            self._error_list.addItem(item)
            self._error_block_ids.append(err.block_id)
        n = len(errors)
        self._err_count_label.setText(f"{n} error{'s' if n != 1 else ''}")
        self._error_frame.setVisible(True)

    def clear_errors(self) -> None:
        """Hide the error panel and remove all rows."""
        self._error_list.clear()
        self._error_block_ids.clear()
        self._err_count_label.setText("")
        self._error_frame.setVisible(False)

    def _on_error_clicked(self, item: QListWidgetItem) -> None:
        idx = self._error_list.row(item)
        if 0 <= idx < len(self._error_block_ids):
            self.error_block_selected.emit(self._error_block_ids[idx])

    # ------------------------------------------------------------------
    # Block parameter editing
    # ------------------------------------------------------------------

    def set_block(self, block: Optional[BlockItem]) -> None:
        self._block = block
        # Clear form
        while self.form.rowCount():
            self.form.removeRow(0)
        self._fields.clear()
        if block is None:
            self.title.setText("<i>(no block selected)</i>")
            self.help.setText("")
            return
        inst = block.instance
        self.title.setText(f"<b>{inst.spec.display_name}</b>  ({inst.block_id})")
        self.help.setText(inst.spec.description)
        for pname, (_default, helptxt) in inst.spec.params.items():
            edit = QLineEdit(str(inst.params.get(pname, _default)))
            edit.setToolTip(helptxt)
            edit.editingFinished.connect(lambda name=pname, w=edit: self._on_edit(name, w))
            self.form.addRow(pname, edit)
            self._fields[pname] = edit

    def _on_edit(self, name: str, widget: QLineEdit) -> None:
        if self._block is None:
            return
        self._block.instance.params[name] = widget.text()


# ---------------------------------------------------------------------------
# Python console tab — now provided by matlab_workspace.MatlabWorkspace
# (see that module for the MATLAB-like editor/command/variables layout).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scope tab (live plot + serial monitor)
# ---------------------------------------------------------------------------

class SerialReader(QThread):
    """Reads newline-delimited float samples from the MCU over serial."""

    sample = pyqtSignal(float, list)  # timestamp (s), channels
    status = pyqtSignal(str)

    def __init__(self, port: str, baud: int = 115200):
        super().__init__()
        self.port = port
        self.baud = baud
        self._stop = threading.Event()

    def run(self) -> None:
        if not HAVE_SERIAL:
            self.status.emit("pyserial not installed")
            return
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.5)
        except Exception as e:
            self.status.emit(f"open failed: {e}")
            return
        self.status.emit(f"connected {self.port} @ {self.baud}")
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                line = ser.readline().decode("ascii", errors="replace").strip()
            except Exception:
                break
            if not line:
                continue
            parts = line.split(",")
            try:
                ch = [float(x) for x in parts]
            except ValueError:
                continue
            self.sample.emit(time.monotonic() - t0, ch)
        try:
            ser.close()
        except Exception:
            pass
        self.status.emit("disconnected")

    def stop(self) -> None:
        self._stop.set()


def _make_plot_widget():
    if not HAVE_PYQTGRAPH:
        return None
    pw = pg.PlotWidget()
    pw.setBackground("#1e1e1e")
    pw.showGrid(x=True, y=True, alpha=0.3)
    pw.addLegend()
    pw.setClipToView(True)
    pw.setDownsampling(auto=True, mode="peak")
    return pw


class LiveScopeTab(QWidget):
    """Scope tab for live serial data from the MCU."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Port:"))
        self.port_box = QComboBox()
        self.refresh_ports()
        bar.addWidget(self.port_box)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        bar.addWidget(self.refresh_btn)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connect)
        bar.addWidget(self.connect_btn)
        bar.addSpacing(8)
        bar.addWidget(QLabel("Window (s):"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setDecimals(1)
        self.window_spin.setRange(0.5, 600.0)
        self.window_spin.setSingleStep(0.5)
        self.window_spin.setValue(5.0)
        self.window_spin.setToolTip("Rolling time window displayed, in seconds.")
        self.window_spin.valueChanged.connect(self._on_window_changed)
        bar.addWidget(self.window_spin)
        self.auto_y_check = QCheckBox("Auto-Y")
        self.auto_y_check.setChecked(True)
        self.auto_y_check.toggled.connect(lambda _: self._repaint())
        bar.addWidget(self.auto_y_check)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.plot = _make_plot_widget()
        if self.plot:
            lay.addWidget(self.plot, 1)
            self._curves: List = []
        else:
            lay.addWidget(QLabel("pyqtgraph is not installed."))

        self.status_lbl = QLabel("idle")
        lay.addWidget(self.status_lbl)

        self._reader: Optional[SerialReader] = None
        self._data: Dict[int, List[Tuple[float, float]]] = {}
        self._max_pts = 200_000
        self._dirty = False
        self._t0: Optional[float] = None  # timestamp of first sample after connect

        self._repaint_timer = QTimer(self)
        self._repaint_timer.setInterval(33)
        self._repaint_timer.timeout.connect(self._tick_repaint)
        self._repaint_timer.start()

    def _on_window_changed(self, _val: float) -> None:
        if not self._data:
            return
        t_latest = max((d[-1][0] for d in self._data.values() if d), default=0.0)
        t_min = t_latest - self.window_spin.value()
        for i in list(self._data.keys()):
            self._data[i] = [p for p in self._data[i] if p[0] >= t_min]
        self._repaint()

    def refresh_ports(self) -> None:
        self.port_box.clear()
        if HAVE_SERIAL:
            for p in serial.tools.list_ports.comports():
                self.port_box.addItem(p.device)
        if self.port_box.count() == 0:
            self.port_box.addItem("(none)")

    def toggle_connect(self) -> None:
        if self._reader is None:
            # Reconnecting — clear old data so the plot restarts from 0 s.
            self._data.clear()
            self._dirty = False
            self._t0 = None
            if self.plot:
                self.plot.clear()
            self._curves = []
            port = self.port_box.currentText()
            if port == "(none)":
                return
            self._reader = SerialReader(port)
            self._reader.sample.connect(self._on_sample)
            self._reader.status.connect(self.status_lbl.setText)
            self._reader.start()
            self.connect_btn.setText("Disconnect")
            self.status_lbl.setText(f"connected — {port}")
        else:
            self._reader.stop()
            self._reader.wait(1000)
            self._reader = None
            self.connect_btn.setText("Connect")
            self.status_lbl.setText("paused")

    def _on_sample(self, t: float, channels: List[float]) -> None:
        if self._t0 is None:
            self._t0 = t
        t_rel = t - self._t0
        for i, v in enumerate(channels):
            buf = self._data.setdefault(i, [])
            buf.append((t_rel, v))
            if len(buf) > self._max_pts:
                del buf[: len(buf) - self._max_pts]
        self._dirty = True

    def _tick_repaint(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        if self._data:
            t_latest = max((d[-1][0] for d in self._data.values() if d), default=0.0)
            t_min = t_latest - self.window_spin.value()
            for i, buf in self._data.items():
                if buf and buf[0][0] < t_min:
                    cut = next((j for j, (tj, _) in enumerate(buf) if tj >= t_min),
                               len(buf))
                    if cut > 0:
                        del buf[:cut]
        self._repaint()

    def _repaint(self) -> None:
        if not self.plot:
            return
        while len(self._curves) < len(self._data):
            color = pg.intColor(len(self._curves), hues=8)
            self._curves.append(self.plot.plot(pen=pg.mkPen(color, width=2),
                                               name=f"ch{len(self._curves)}"))
        t_latest = 0.0
        y_min, y_max = float("inf"), float("-inf")
        for i, data in self._data.items():
            xs = [p[0] for p in data]
            ys = [p[1] for p in data]
            self._curves[i].setData(xs, ys)
            if xs:
                t_latest = max(t_latest, xs[-1])
            if ys:
                y_min = min(y_min, min(ys))
                y_max = max(y_max, max(ys))
        if t_latest > 0.0:
            win = self.window_spin.value()
            self.plot.setXRange(max(0.0, t_latest - win), t_latest, padding=0)
        if self.auto_y_check.isChecked() and y_max >= y_min:
            if y_max == y_min:
                pad = 1.0 if y_max == 0.0 else abs(y_max) * 0.1 + 1e-6
                self.plot.setYRange(y_min - pad, y_max + pad, padding=0)
            else:
                span = y_max - y_min
                self.plot.setYRange(y_min - span * 0.08, y_max + span * 0.08, padding=0)


class SimScopeTab(QWidget):
    """Scope tab for host-side simulation results."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.sim_btn = QPushButton("Simulate Model")
        self.sim_btn.setToolTip("Run the block model in Python and plot the result.")
        bar.addWidget(self.sim_btn)
        bar.addWidget(QLabel("Duration (s):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setDecimals(1)
        self.duration_spin.setRange(0.5, 600.0)
        self.duration_spin.setSingleStep(1.0)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setToolTip("How many seconds to simulate.")
        bar.addWidget(self.duration_spin)
        self.auto_y_check = QCheckBox("Auto-Y")
        self.auto_y_check.setChecked(True)
        bar.addWidget(self.auto_y_check)
        bar.addStretch(1)
        lay.addLayout(bar)

        self.plot = _make_plot_widget()
        if self.plot:
            lay.addWidget(self.plot, 1)
            self._curves: List = []
        else:
            lay.addWidget(QLabel("pyqtgraph is not installed."))

    def show_simulation(self, times: np.ndarray, signals: Dict[str, np.ndarray]) -> None:
        if not self.plot:
            return
        self.plot.clear()
        self._curves = []
        colors = [pg.mkPen(pg.intColor(i, hues=8), width=2) for i in range(len(signals))]
        for (name, ys), pen in zip(signals.items(), colors):
            self._curves.append(self.plot.plot(times, ys, pen=pen, name=name))
        if self.auto_y_check.isChecked() and self._curves:
            all_y = np.concatenate(list(signals.values()))
            y_min, y_max = float(all_y.min()), float(all_y.max())
            if y_min == y_max:
                pad = 1.0 if y_max == 0.0 else abs(y_max) * 0.1 + 1e-6
                self.plot.setYRange(y_min - pad, y_max + pad, padding=0)
            else:
                span = y_max - y_min
                self.plot.setYRange(y_min - span * 0.08, y_max + span * 0.08, padding=0)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

# STM32 pin pattern: P followed by a letter A-H and 1–2 digits (e.g. PA5, PC13)
_PIN_RE = re.compile(r"^P[A-H]\d{1,2}$", re.IGNORECASE)


def _try_eval_param(s: str, workspace=None) -> Optional[float]:
    """Try to resolve parameter string *s* to a float.

    Accepts numeric literals and workspace variable references.
    Returns None if resolution fails.
    """
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    if workspace is not None:
        try:
            v = workspace.eval_param(str(s))
            return float(v)
        except Exception:
            pass
    return None


def _is_valid_stm32_pin(pin: str) -> bool:
    return bool(_PIN_RE.match(pin.strip()))


def _validate_block(b: dict, workspace=None) -> List[ValidationError]:
    """Return a list of ValidationErrors for a single block dict."""
    errors: List[ValidationError] = []
    btype = b["type"]
    bid   = b["id"]
    p     = b.get("params", {})

    def _num(param: str, default: str = "0.0") -> Optional[float]:
        return _try_eval_param(p.get(param, default), workspace)

    def _bad(param: str, code: str, msg: str) -> None:
        errors.append(ValidationError(bid, btype, param, code, msg))

    # ---- SquareWave -------------------------------------------------------
    if btype == "SquareWave":
        f = _num("frequency_hz", "1.0")
        if f is None:          _bad("frequency_hz", "E001", "Must be a valid number or workspace expression")
        elif f <= 0:           _bad("frequency_hz", "E002", "Frequency must be > 0 Hz")
        if _num("amplitude", "1.0") is None:
                               _bad("amplitude",    "E001", "Must be a valid number")
        d = _num("duty", "0.5")
        if d is None:          _bad("duty",         "E001", "Must be a valid number")
        elif not 0.0 <= d <= 1.0:
                               _bad("duty",         "E002", "Duty cycle must be in range [0, 1]")
        if _num("offset", "0.0") is None:
                               _bad("offset",       "E001", "Must be a valid number")

    # ---- GpioIn -----------------------------------------------------------
    elif btype == "GpioIn":
        pin = p.get("pin", "").strip()
        if not _is_valid_stm32_pin(pin):
            _bad("pin", "E004", f"'{pin}' is not a valid STM32 pin (e.g. PC13, PA5)")
        pull = p.get("pull", "none").strip().lower()
        if pull not in ("none", "up", "down"):
            _bad("pull", "E003", "Must be 'none', 'up', or 'down'")
        al = _num("active_low", "1")
        if al is None or int(al) not in (0, 1):
            _bad("active_low", "E002", "Must be 0 or 1")

    # ---- GpioOut ----------------------------------------------------------
    elif btype == "GpioOut":
        pin = p.get("pin", "").strip()
        if not _is_valid_stm32_pin(pin):
            _bad("pin", "E004", f"'{pin}' is not a valid STM32 pin (e.g. PA5)")
        if _num("threshold", "0.5") is None:
            _bad("threshold", "E001", "Must be a valid number")

    # ---- Scope ------------------------------------------------------------
    elif btype == "Scope":
        mp = _num("max_points", "500")
        if mp is None or mp < 1 or mp % 1 != 0:
            _bad("max_points", "E002", "Must be a positive integer")
        s = _num("stream", "1")
        if s is None or int(s) not in (0, 1):
            _bad("stream", "E002", "Must be 0 or 1")

    # ---- Ultrasonic -------------------------------------------------------
    elif btype == "Ultrasonic":
        trig = p.get("trig_pin", "").strip()
        echo = p.get("echo_pin", "").strip()
        if not _is_valid_stm32_pin(trig):
            _bad("trig_pin", "E004", f"'{trig}' is not a valid STM32 pin")
        if not _is_valid_stm32_pin(echo):
            _bad("echo_pin", "E004", f"'{echo}' is not a valid STM32 pin")
        if (_is_valid_stm32_pin(trig) and _is_valid_stm32_pin(echo)
                and trig.upper() == echo.upper()):
            _bad("echo_pin", "E003", "TRIG and ECHO pins must be different")
        pm = _num("period_ms", "60")
        if pm is None:     _bad("period_ms",  "E001", "Must be a valid number")
        elif pm < 50:      _bad("period_ms",  "E002", "Must be >= 50 ms (HC-SR04 minimum)")
        tu = _num("timeout_us", "30000")
        if tu is None:     _bad("timeout_us", "E001", "Must be a valid number")
        elif tu <= 0:      _bad("timeout_us", "E002", "Must be > 0 us")

    # ---- Constant ---------------------------------------------------------
    elif btype == "Constant":
        if _num("value", "1.0") is None:
            _bad("value", "E001", "Must be a valid number or workspace expression")

    # ---- Sum / Product — no params to validate ----------------------------
    elif btype in ("Sum", "Product"):
        pass

    # ---- Step -------------------------------------------------------------
    elif btype == "Step":
        st = _num("step_time", "1.0")
        if st is None:   _bad("step_time",     "E001", "Must be a valid number")
        elif st < 0:     _bad("step_time",     "E002", "Step time must be >= 0")
        if _num("initial_value", "0.0") is None:
                         _bad("initial_value", "E001", "Must be a valid number")
        if _num("final_value", "1.0") is None:
                         _bad("final_value",   "E001", "Must be a valid number")

    # ---- Integrator -------------------------------------------------------
    elif btype == "Integrator":
        if _num("initial_value", "0.0") is None:
            _bad("initial_value", "E001", "Must be a valid number")
        ul = _num("upper_limit",  "1e10")
        ll = _num("lower_limit", "-1e10")
        if ul is None:   _bad("upper_limit", "E001", "Must be a valid number")
        if ll is None:   _bad("lower_limit", "E001", "Must be a valid number")
        if ul is not None and ll is not None and ul <= ll:
            _bad("upper_limit", "E007", "upper_limit must be strictly greater than lower_limit")

    # ---- TransferFcn ------------------------------------------------------
    elif btype == "TransferFcn":
        num_str = p.get("numerator",   "1")
        den_str = p.get("denominator", "1 1")
        num_c: List[float] = []
        den_c: List[float] = []
        try:
            num_c = [float(x) for x in num_str.split()]
            if not num_c:
                _bad("numerator", "E001", "Must contain at least one coefficient")
        except ValueError:
            _bad("numerator", "E001",
                 "Coefficients must be space-separated numbers (e.g. '1 2 1')")
        try:
            den_c = [float(x) for x in den_str.split()]
            if not den_c:
                _bad("denominator", "E001", "Must contain at least one coefficient")
            elif den_c[0] == 0.0:
                _bad("denominator", "E001",
                     "Leading denominator coefficient must be non-zero")
        except ValueError:
            _bad("denominator", "E001",
                 "Coefficients must be space-separated numbers (e.g. '1 2 1')")
        if num_c and den_c and len(num_c) > len(den_c):
            _bad("numerator", "E006",
                 f"Improper transfer function: numerator order {len(num_c)-1} "
                 f"> denominator order {len(den_c)-1}")

    # ---- PID --------------------------------------------------------------
    elif btype == "PID":
        for pname in ("Kp", "Ki", "Kd"):
            if _num(pname, "0.0") is None:
                _bad(pname, "E001", f"{pname} must be a valid number")
        n = _num("N", "100.0")
        if n is None:    _bad("N", "E001", "Must be a valid number")
        elif n <= 0:     _bad("N", "E002", "Derivative filter bandwidth N must be > 0")
        ul = _num("upper_limit",  "1e10")
        ll = _num("lower_limit", "-1e10")
        if ul is None:   _bad("upper_limit", "E001", "Must be a valid number")
        if ll is None:   _bad("lower_limit", "E001", "Must be a valid number")
        if ul is not None and ll is not None and ul <= ll:
            _bad("upper_limit", "E007",
                 "upper_limit must be strictly greater than lower_limit")

    # ---- ToWorkspace ------------------------------------------------------
    elif btype == "ToWorkspace":
        vn = p.get("variable_name", "yout").strip()
        if not vn:
            _bad("variable_name", "E003", "Variable name cannot be empty")
        elif not vn.isidentifier():
            _bad("variable_name", "E005",
                 f"'{vn}' is not a valid Python identifier")
        mp = _num("max_points", "10000")
        if mp is None or mp < 1:
            _bad("max_points", "E002", "Must be a positive integer >= 1")
        dc = _num("decimation", "1")
        if dc is None or dc < 1:
            _bad("decimation", "E002", "Must be a positive integer >= 1")
        st = _num("save_time", "1")
        if st is None or int(st) not in (0, 1):
            _bad("save_time", "E002", "Must be 0 or 1")

    # ---- SineWave ---------------------------------------------------------
    elif btype == "SineWave":
        f = _num("frequency_hz", "1.0")
        if f is None:    _bad("frequency_hz", "E001", "Must be a valid number or workspace expression")
        elif f <= 0:     _bad("frequency_hz", "E002", "Frequency must be > 0 Hz")
        if _num("amplitude", "1.0") is None:
                         _bad("amplitude",    "E001", "Must be a valid number")
        if _num("phase_deg", "0.0") is None:
                         _bad("phase_deg",    "E001", "Must be a valid number")
        if _num("offset", "0.0") is None:
                         _bad("offset",       "E001", "Must be a valid number")

    # ---- Ramp -------------------------------------------------------------
    elif btype == "Ramp":
        if _num("slope", "1.0") is None:
            _bad("slope",          "E001", "Must be a valid number")
        st = _num("start_time", "0.0")
        if st is None:   _bad("start_time",     "E001", "Must be a valid number")
        elif st < 0:     _bad("start_time",     "E002", "start_time must be >= 0")
        if _num("initial_output", "0.0") is None:
            _bad("initial_output", "E001", "Must be a valid number")

    # ---- Clock ------------------------------------------------------------
    elif btype == "Clock":
        pass  # no params to validate

    # ---- PulseGenerator ---------------------------------------------------
    elif btype == "PulseGenerator":
        if _num("amplitude", "1.0") is None:
            _bad("amplitude",   "E001", "Must be a valid number")
        per = _num("period", "1.0")
        if per is None:  _bad("period",      "E001", "Must be a valid number")
        elif per <= 0:   _bad("period",      "E002", "Period must be > 0")
        pw = _num("pulse_width", "50")
        if pw is None:   _bad("pulse_width", "E001", "Must be a valid number")
        elif not (0.0 <= pw <= 100.0):
                         _bad("pulse_width", "E002", "pulse_width must be in [0, 100]")
        if _num("phase_delay", "0.0") is None:
            _bad("phase_delay", "E001", "Must be a valid number")

    # ---- Gain -------------------------------------------------------------
    elif btype == "Gain":
        if _num("gain", "1.0") is None:
            _bad("gain", "E001", "Must be a valid number")

    # ---- Abs --------------------------------------------------------------
    elif btype == "Abs":
        pass  # no params

    # ---- Sign -------------------------------------------------------------
    elif btype == "Sign":
        pass  # no params

    # ---- Sqrt -------------------------------------------------------------
    elif btype == "Sqrt":
        mode = p.get("mode", "sqrt").strip().lower()
        if mode not in ("sqrt", "signed_sqrt"):
            _bad("mode", "E003", "mode must be 'sqrt' or 'signed_sqrt'")

    # ---- Saturation -------------------------------------------------------
    elif btype == "Saturation":
        ul = _num("upper_limit",  "1.0")
        ll = _num("lower_limit", "-1.0")
        if ul is None:   _bad("upper_limit", "E001", "Must be a valid number")
        if ll is None:   _bad("lower_limit", "E001", "Must be a valid number")
        if ul is not None and ll is not None and ul <= ll:
            _bad("upper_limit", "E007", "upper_limit must be strictly greater than lower_limit")

    # ---- DeadZone ---------------------------------------------------------
    elif btype == "DeadZone":
        uv = _num("upper_value",  "0.5")
        lv = _num("lower_value", "-0.5")
        if uv is None:   _bad("upper_value", "E001", "Must be a valid number")
        if lv is None:   _bad("lower_value", "E001", "Must be a valid number")
        if uv is not None and lv is not None and uv < lv:
            _bad("upper_value", "E007", "upper_value must be >= lower_value")

    # ---- MinMax -----------------------------------------------------------
    elif btype == "MinMax":
        fn = p.get("function", "min").strip().lower()
        if fn not in ("min", "max"):
            _bad("function", "E003", "function must be 'min' or 'max'")

    # ---- RelationalOperator -----------------------------------------------
    elif btype == "RelationalOperator":
        op = p.get("operator", ">").strip()
        if op not in (">", "<", ">=", "<=", "==", "!="):
            _bad("operator", "E003", "operator must be one of: >, <, >=, <=, ==, !=")

    # ---- LogicalOperator --------------------------------------------------
    elif btype == "LogicalOperator":
        op = p.get("operator", "AND").strip().upper()
        if op not in ("AND", "OR", "NOT", "NAND", "NOR", "XOR"):
            _bad("operator", "E003", "operator must be one of: AND, OR, NOT, NAND, NOR, XOR")

    # ---- Switch -----------------------------------------------------------
    elif btype == "Switch":
        if _num("threshold", "0.5") is None:
            _bad("threshold", "E001", "Must be a valid number")
        crit = p.get("criteria", ">=").strip()
        if crit not in (">", ">=", "=="):
            _bad("criteria", "E003", "criteria must be one of: >, >=, ==")

    # ---- UnitDelay --------------------------------------------------------
    elif btype == "UnitDelay":
        if _num("initial_condition", "0.0") is None:
            _bad("initial_condition", "E001", "Must be a valid number")

    # ---- DiscreteIntegrator -----------------------------------------------
    elif btype == "DiscreteIntegrator":
        if _num("gain_value", "1.0") is None:
            _bad("gain_value",        "E001", "Must be a valid number")
        if _num("initial_condition", "0.0") is None:
            _bad("initial_condition", "E001", "Must be a valid number")
        ul = _num("upper_limit",  "1e10")
        ll = _num("lower_limit", "-1e10")
        if ul is None:   _bad("upper_limit", "E001", "Must be a valid number")
        if ll is None:   _bad("lower_limit", "E001", "Must be a valid number")
        if ul is not None and ll is not None and ul <= ll:
            _bad("upper_limit", "E007", "upper_limit must be strictly greater than lower_limit")
        meth = p.get("method", "Forward Euler").strip()
        valid_methods = ("Forward Euler", "Backward Euler", "Trapezoidal")
        if meth not in valid_methods:
            _bad("method", "E003", f"method must be one of: {', '.join(valid_methods)}")

    # ---- ZeroOrderHold ----------------------------------------------------
    elif btype == "ZeroOrderHold":
        st = _num("sample_time", "0.01")
        if st is None:   _bad("sample_time", "E001", "Must be a valid number")
        elif st <= 0:    _bad("sample_time", "E002", "sample_time must be > 0")

    # ---- Derivative -------------------------------------------------------
    elif btype == "Derivative":
        ic = _num("initial_condition", "0.0")
        if ic is None or (isinstance(ic, float) and (math.isnan(ic) or math.isinf(ic))):
            _bad("initial_condition", "E001", "Must be a valid number")

    # ---- Lookup1D ---------------------------------------------------------
    elif btype == "Lookup1D":
        bp_str  = p.get("breakpoints", "0 1")
        tbl_str = p.get("table_data",  "0 1")
        bp_vals: List[float] = []
        tbl_vals: List[float] = []
        try:
            bp_vals = [float(x) for x in bp_str.split()]
            if not bp_vals:
                _bad("breakpoints", "E001", "Must contain at least one value")
        except ValueError:
            _bad("breakpoints", "E001", "Must be space-separated numbers")
        try:
            tbl_vals = [float(x) for x in tbl_str.split()]
            if not tbl_vals:
                _bad("table_data", "E001", "Must contain at least one value")
        except ValueError:
            _bad("table_data", "E001", "Must be space-separated numbers")
        if bp_vals and tbl_vals and len(bp_vals) != len(tbl_vals):
            _bad("table_data", "E007",
                 f"breakpoints ({len(bp_vals)}) and table_data ({len(tbl_vals)}) must have same length")
        if len(bp_vals) >= 2:
            for i in range(len(bp_vals) - 1):
                if bp_vals[i] >= bp_vals[i+1]:
                    _bad("breakpoints", "E002", "breakpoints must be strictly increasing")
                    break
        extrap = p.get("extrapolation", "clip").strip().lower()
        if extrap not in ("clip", "linear"):
            _bad("extrapolation", "E003", "extrapolation must be 'clip' or 'linear'")

    # ---- ADC --------------------------------------------------------------
    elif btype == "ADC":
        ch = _num("channel", "1")
        if ch is None or not (1 <= int(ch) <= 16):
            _bad("channel", "E002", "channel must be an integer in [1, 16]")
        res = _num("resolution", "12")
        if res is None or int(res) not in (6, 8, 10, 12):
            _bad("resolution", "E002", "resolution must be 6, 8, 10, or 12")
        vr = _num("vref", "3.3")
        if vr is None:   _bad("vref",      "E001", "Must be a valid number")
        elif vr <= 0:    _bad("vref",      "E002", "vref must be > 0")
        if _num("sim_value", "0.0") is None:
            _bad("sim_value", "E001", "Must be a valid number")

    # ---- DAC --------------------------------------------------------------
    elif btype == "DAC":
        ch = _num("channel", "1")
        if ch is None or int(ch) not in (1, 2):
            _bad("channel", "E002", "channel must be 1 or 2")
        vr = _num("vref", "3.3")
        if vr is None:   _bad("vref", "E001", "Must be a valid number")
        elif vr <= 0:    _bad("vref", "E002", "vref must be > 0")

    # ---- PWMOut -----------------------------------------------------------
    elif btype == "PWMOut":
        timer = p.get("timer", "").strip()
        if not timer:
            _bad("timer", "E003", "timer name cannot be empty (e.g. TIM2)")
        ch = _num("channel", "1")
        if ch is None or int(ch) not in (1, 2, 3, 4):
            _bad("channel", "E002", "channel must be 1, 2, 3, or 4")
        fhz = _num("frequency_hz", "1000")
        if fhz is None:  _bad("frequency_hz", "E001", "Must be a valid number")
        elif fhz <= 0:   _bad("frequency_hz", "E002", "frequency_hz must be > 0")
        md = _num("max_duty", "100.0")
        if md is None:   _bad("max_duty", "E001", "Must be a valid number")
        elif md <= 0:    _bad("max_duty", "E002", "max_duty must be > 0")

    # ---- TimerTick --------------------------------------------------------
    elif btype == "TimerTick":
        sc = _num("scale", "0.001")
        if sc is None:   _bad("scale", "E001", "Must be a valid number")
        elif sc == 0:    _bad("scale", "E002", "scale must be != 0")

    return errors


def validate_model(model: dict, workspace=None) -> List[ValidationError]:
    """Validate all block parameters in *model*.

    Returns a (possibly empty) list of :class:`ValidationError` objects.
    An empty list means the model is ready to simulate / generate code.

    Error codes
    -----------
    E001  Invalid value — cannot be parsed as a number
    E002  Out of range — numeric value violates a constraint
    E003  Invalid option string — not one of the allowed choices
    E004  Invalid pin name — does not match STM32 Pxnn format
    E005  Invalid identifier — not a legal Python variable name
    E006  Improper transfer function — numerator degree > denominator degree
    E007  Limit conflict — upper_limit ≤ lower_limit
    """
    errors: List[ValidationError] = []
    for b in model.get("blocks", []):
        errors.extend(_validate_block(b, workspace))
    return errors


# ---------------------------------------------------------------------------
# Pure-Python simulator (for the "Simulate Model" button)
# ---------------------------------------------------------------------------

def simulate_model(model: dict, duration_s: float = 2.0, step_s: float = 0.001
                   ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run the block diagram numerically on the host, without hardware.

    Only the supported blocks (SquareWave, GpioIn, GpioOut, Scope) are
    simulated. GpioIn is treated as a constant 0 input since there is no
    real button; users can override by assigning ``gpioin_<id>`` in the
    workspace.
    """
    n = int(duration_s / step_s)
    t = np.arange(n) * step_s

    # Evaluate each block's output trace
    outs: Dict[Tuple[str, str], np.ndarray] = {}

    def pval(pstr: str) -> float:
        return float(WORKSPACE.eval_param(pstr))

    blocks_by_id = {b["id"]: b for b in model["blocks"]}

    # First pass: produce outputs for source blocks
    for b in model["blocks"]:
        if b["type"] == "SquareWave":
            f = pval(b["params"]["frequency_hz"])
            A = pval(b["params"]["amplitude"])
            off = pval(b["params"]["offset"])
            duty = pval(b["params"]["duty"])
            phase = (t * f) % 1.0
            y = np.where(phase < duty, A, off)
            outs[(b["id"], "y")] = y
        elif b["type"] == "Constant":
            val = pval(b["params"]["value"])
            outs[(b["id"], "y")] = np.full(n, val)
        elif b["type"] == "GpioIn":
            override = WORKSPACE.globals.get(f"gpioin_{b['id']}")
            if override is None:
                outs[(b["id"], "y")] = np.zeros(n)
            else:
                outs[(b["id"], "y")] = np.asarray(override)[:n]
        elif b["type"] == "Ultrasonic":
            override = WORKSPACE.globals.get(f"ultrasonic_{b['id']}")
            if override is None:
                outs[(b["id"], "d")] = np.zeros(n)
            else:
                arr = np.asarray(override, dtype=float)
                if arr.ndim == 0:
                    outs[(b["id"], "d")] = np.full(n, float(arr))
                else:
                    outs[(b["id"], "d")] = arr[:n]
        elif b["type"] == "Step":
            st  = pval(b["params"].get("step_time",     "1.0"))
            iv  = pval(b["params"].get("initial_value",  "0.0"))
            fv  = pval(b["params"].get("final_value",    "1.0"))
            outs[(b["id"], "y")] = np.where(t >= st, fv, iv)

        elif b["type"] == "SineWave":
            freq  = pval(b["params"].get("frequency_hz", "1.0"))
            amp   = pval(b["params"].get("amplitude",    "1.0"))
            phase = pval(b["params"].get("phase_deg",    "0.0")) * np.pi / 180.0
            off   = pval(b["params"].get("offset",       "0.0"))
            outs[(b["id"], "y")] = amp * np.sin(2 * np.pi * freq * t + phase) + off

        elif b["type"] == "Ramp":
            slope = pval(b["params"].get("slope",          "1.0"))
            start = pval(b["params"].get("start_time",     "0.0"))
            init  = pval(b["params"].get("initial_output", "0.0"))
            outs[(b["id"], "y")] = np.where(t >= start, init + slope * (t - start), init)

        elif b["type"] == "Clock":
            outs[(b["id"], "y")] = t.copy()

        elif b["type"] == "PulseGenerator":
            amp   = pval(b["params"].get("amplitude",   "1.0"))
            per   = max(1e-9, pval(b["params"].get("period",       "1.0")))
            pw    = pval(b["params"].get("pulse_width",  "50")) / 100.0
            delay = pval(b["params"].get("phase_delay",  "0.0"))
            t_rel = np.where(t >= delay, np.mod(t - delay, per), -1.0)
            outs[(b["id"], "y")] = np.where((t >= delay) & (t_rel < pw * per), amp, 0.0)

        elif b["type"] == "ADC":
            sim_val = pval(b["params"].get("sim_value", "0.0"))
            outs[(b["id"], "y")] = np.full(n, sim_val)

        elif b["type"] == "TimerTick":
            scale = pval(b["params"].get("scale", "0.001"))
            # t is in seconds; HAL_GetTick is ms, so sim output = t * 1000 * scale
            outs[(b["id"], "y")] = t * 1000.0 * scale

    # Build wire map once for all subsequent passes.
    wires: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for c in model["connections"]:
        wires[(c["dst_block"], c["dst_port"])] = (c["src_block"], c["src_port"])

    # Middle pass: all blocks with inputs, in topological order.
    for b in _ct_topo_order(model):
        bid = b["id"]

        def _input(port: str, default: float = 0.0) -> np.ndarray:
            src = wires.get((bid, port))
            if src is None:
                return np.full(n, default)
            return outs.get(src, np.full(n, default))

        if b["type"] == "Sum":
            outs[(bid, "y")] = _input("u0") + _input("u1")

        elif b["type"] == "Product":
            outs[(bid, "y")] = _input("u0", 1.0) * _input("u1", 1.0)

        elif b["type"] == "Integrator":
            u_arr = _input("u")
            ic    = pval(b["params"].get("initial_value",  "0.0"))
            upper = pval(b["params"].get("upper_limit",    "1e10"))
            lower = pval(b["params"].get("lower_limit",   "-1e10"))
            # Forward Euler: y[k] = ic + sum(u[0..k-1]) * dt
            y = ic + np.concatenate([[0.0], np.cumsum(u_arr[:-1])]) * step_s
            outs[(bid, "y")] = np.clip(y, lower, upper)

        elif b["type"] == "TransferFcn":
            u_arr   = _input("u")
            num_str = b["params"].get("numerator",   "1")
            den_str = b["params"].get("denominator", "1 1")
            try:
                bz, az = _bilinear_tf(num_str, den_str, 1.0 / step_s)
                # Use scipy if available; fall back to a pure-numpy IIR
                try:
                    from scipy.signal import lfilter
                    y = lfilter(bz, az, u_arr)
                except ImportError:
                    y = np.zeros(n)
                    order = len(az) - 1
                    s = np.zeros(order)          # Direct Form II Transposed state
                    for k in range(n):
                        yk = bz[0] * u_arr[k] + (s[0] if order else 0.0)
                        for j in range(order - 1):
                            s[j] = bz[j+1]*u_arr[k] - az[j+1]*yk + s[j+1]
                        if order:
                            s[order-1] = bz[order]*u_arr[k] - az[order]*yk
                        y[k] = yk
            except Exception:
                y = np.zeros(n)
            outs[(bid, "y")] = y

        elif b["type"] == "PID":
            u_arr  = _input("u")
            Kp     = pval(b["params"].get("Kp",          "1.0"))
            Ki     = pval(b["params"].get("Ki",          "0.0"))
            Kd     = pval(b["params"].get("Kd",          "0.0"))
            N_filt = pval(b["params"].get("N",          "100.0"))
            upper  = pval(b["params"].get("upper_limit", "1e10"))
            lower  = pval(b["params"].get("lower_limit","-1e10"))
            KdN    = Kd * N_filt
            N_dt   = min(N_filt * step_s, 1.99)   # clamp for forward-Euler stability
            y      = np.empty(n)
            integ  = 0.0
            d_st   = 0.0
            for k in range(n):
                e      = u_arr[k]
                p_out  = Kp * e
                integ += Ki * e * step_s
                d_out  = KdN * (e - d_st)
                d_st  += N_dt * (e - d_st)
                raw    = p_out + integ + d_out
                y[k]   = max(lower, min(upper, raw))
            outs[(bid, "y")] = y

        elif b["type"] == "Gain":
            k = pval(b["params"].get("gain", "1.0"))
            outs[(bid, "y")] = k * _input("u")

        elif b["type"] == "Abs":
            outs[(bid, "y")] = np.abs(_input("u"))

        elif b["type"] == "Sign":
            outs[(bid, "y")] = np.sign(_input("u")).astype(float)

        elif b["type"] == "Sqrt":
            u = _input("u")
            if b["params"].get("mode", "sqrt").strip().lower() == "signed_sqrt":
                outs[(bid, "y")] = np.sign(u) * np.sqrt(np.abs(u))
            else:
                outs[(bid, "y")] = np.sqrt(np.abs(u))

        elif b["type"] == "Saturation":
            upper = pval(b["params"].get("upper_limit",  "1.0"))
            lower = pval(b["params"].get("lower_limit", "-1.0"))
            outs[(bid, "y")] = np.clip(_input("u"), lower, upper)

        elif b["type"] == "DeadZone":
            u = _input("u")
            upper = pval(b["params"].get("upper_value",  "0.5"))
            lower = pval(b["params"].get("lower_value", "-0.5"))
            outs[(bid, "y")] = np.where(u > upper, u - upper,
                               np.where(u < lower, u - lower, 0.0))

        elif b["type"] == "MinMax":
            fn = b["params"].get("function", "min").strip().lower()
            u0, u1 = _input("u0"), _input("u1")
            outs[(bid, "y")] = np.minimum(u0, u1) if fn == "min" else np.maximum(u0, u1)

        elif b["type"] == "RelationalOperator":
            op = b["params"].get("operator", ">").strip()
            u0, u1 = _input("u0"), _input("u1")
            result = {">" : u0 > u1, "<" : u0 < u1, ">=": u0 >= u1,
                      "<=": u0 <= u1, "==": u0 == u1, "!=": u0 != u1}.get(op, u0 > u1)
            outs[(bid, "y")] = result.astype(float)

        elif b["type"] == "LogicalOperator":
            op = b["params"].get("operator", "AND").strip().upper()
            a = _input("u0") != 0
            b_ = _input("u1") != 0
            ops = {"AND": a & b_, "OR": a | b_, "NOT": ~a,
                   "NAND": ~(a & b_), "NOR": ~(a | b_), "XOR": a ^ b_}
            outs[(bid, "y")] = ops.get(op, a & b_).astype(float)

        elif b["type"] == "Switch":
            thr  = pval(b["params"].get("threshold", "0.5"))
            crit = b["params"].get("criteria", ">=").strip()
            u0, u1, u2 = _input("u0"), _input("u1"), _input("u2")
            cond = (u1 > thr) if crit == ">" else (u1 == thr) if crit == "==" else (u1 >= thr)
            outs[(bid, "y")] = np.where(cond, u0, u2)

        elif b["type"] == "UnitDelay":
            ic = pval(b["params"].get("initial_condition", "0.0"))
            u  = _input("u")
            outs[(bid, "y")] = np.concatenate([[ic], u[:-1]])

        elif b["type"] == "DiscreteIntegrator":
            K     = pval(b["params"].get("gain_value",        "1.0"))
            ic    = pval(b["params"].get("initial_condition", "0.0"))
            upper = pval(b["params"].get("upper_limit",  "1e10"))
            lower = pval(b["params"].get("lower_limit", "-1e10"))
            meth  = b["params"].get("method", "Forward Euler").strip()
            u = _input("u")
            y = np.empty(n)
            state = ic
            for k in range(n):
                y[k] = max(lower, min(upper, state))
                if meth == "Backward Euler":
                    state = max(lower, min(upper, state + K * u[k] * step_s))
                elif meth.lower().startswith("trap"):
                    prev = u[k-1] if k > 0 else u[0]
                    state = max(lower, min(upper, state + K * 0.5 * (u[k] + prev) * step_s))
                else:
                    state = max(lower, min(upper, state + K * u[k] * step_s))
            outs[(bid, "y")] = y

        elif b["type"] == "ZeroOrderHold":
            sample_t = max(step_s, pval(b["params"].get("sample_time", "0.01")))
            u  = _input("u")
            y  = np.empty(n)
            held      = u[0]
            next_samp = 0.0
            for k in range(n):
                if t[k] >= next_samp:
                    held = u[k]
                    next_samp += sample_t
                y[k] = held
            outs[(bid, "y")] = y

        elif b["type"] == "Derivative":
            ic = pval(b["params"].get("initial_condition", "0.0"))
            u  = _input("u")
            u_prev = np.concatenate([[ic], u[:-1]])
            outs[(bid, "y")] = (u - u_prev) / step_s

        elif b["type"] == "Lookup1D":
            u    = _input("u")
            extrap = b["params"].get("extrapolation", "clip").strip().lower()
            try:
                bp  = np.array([float(x) for x in b["params"].get("breakpoints","0 1").split()])
                tbl = np.array([float(x) for x in b["params"].get("table_data", "0 1").split()])
                if extrap == "linear" and len(bp) >= 2:
                    sl = (tbl[1]-tbl[0])/(bp[1]-bp[0]) if bp[1] != bp[0] else 0.0
                    sr = (tbl[-1]-tbl[-2])/(bp[-1]-bp[-2]) if bp[-1] != bp[-2] else 0.0
                    y = np.where(u < bp[0],  tbl[0]  + sl*(u-bp[0]),
                        np.where(u > bp[-1], tbl[-1] + sr*(u-bp[-1]),
                        np.interp(u, bp, tbl)))
                else:
                    y = np.interp(u, bp, tbl)
            except Exception:
                y = np.zeros(n)
            outs[(bid, "y")] = y

    # ToWorkspace: capture signal into the Python workspace and expose it to
    # the Simulate Scope tab so the user can see what was saved.
    for b in model["blocks"]:
        if b["type"] == "ToWorkspace":
            var_name = b["params"].get("variable_name", "yout").strip() or "yout"
            try:
                max_pts = max(1, int(float(b["params"].get("max_points", "10000"))))
            except Exception:
                max_pts = 10000
            try:
                decim = max(1, int(float(b["params"].get("decimation", "1"))))
            except Exception:
                decim = 1
            save_t = int(float(b["params"].get("save_time", "1"))) != 0

            src = wires.get((b["id"], "u"))
            if src is not None:
                sig = outs.get(src)
                if sig is not None:
                    sig_d   = sig[::decim][-max_pts:]
                    time_d  = t[::decim][-max_pts:]
                    WORKSPACE.globals[var_name] = sig_d
                    if save_t:
                        WORKSPACE.globals[f"{var_name}_t"] = time_d
                    # keep the trimmed slice so the display loop below can
                    # show it in the Simulate Scope tab at the correct time axis
                    outs[(b["id"], "_saved")] = (time_d, sig_d)

    display: Dict[str, np.ndarray] = {}
    for b in model["blocks"]:
        if b["type"] == "Scope":
            for port in ("u0", "u1", "u2"):
                key = (b["id"], port)
                if key in wires:
                    sig = outs.get(wires[key])
                    if sig is not None:
                        display[f"{b['id']}.{port}"] = sig
        elif b["type"] == "GpioOut":
            key = (b["id"], "u")
            if key in wires:
                sig = outs.get(wires[key])
                if sig is not None:
                    thr = pval(b["params"]["threshold"])
                    display[f"{b['id']}.pin"] = (sig > thr).astype(float)
        elif b["type"] in ("DAC", "PWMOut"):
            key = (b["id"], "u")
            if key in wires:
                sig = outs.get(wires[key])
                if sig is not None:
                    display[f"{b['id']}.u"] = sig
        elif b["type"] == "ToWorkspace":
            saved = outs.get((b["id"], "_saved"))
            if saved is not None:
                var_name = b["params"].get("variable_name", "yout").strip() or "yout"
                _tw_t, _tw_sig = saved
                # Pad/trim to full time axis length for the scope plot
                # (decimated data: just resample to full grid via interpolation
                # so the scope x-axis lines up with other channels).
                if len(_tw_t) == n:
                    display[f"{var_name} [{b['id']}]"] = _tw_sig
                else:
                    display[f"{var_name} [{b['id']}]"] = np.interp(t, _tw_t, _tw_sig)

    if not display:
        display = {k[0] + "." + k[1]: v for k, v in outs.items()}
    return t, display


# ---------------------------------------------------------------------------
# Build & flash worker
# ---------------------------------------------------------------------------

def find_cubef4() -> Optional[str]:
    """Look for an STM32CubeF4 install in the usual locations.

    Returns a forward-slashed path (make prefers those even on Windows)
    or None if nothing is found.
    """
    import glob
    candidates: List[str] = []
    home = Path.home()
    for root in (
        home / "STM32Cube" / "Repository",
        home / "STM32Cube",
        Path("C:/STM32Cube/Repository") if os.name == "nt" else Path("/opt/STM32Cube"),
    ):
        if root.exists():
            candidates.extend(glob.glob(str(root / "STM32Cube_FW_F4_*")))
    # Newest version first
    candidates.sort(reverse=True)
    if candidates:
        return candidates[0].replace("\\", "/")
    return None


class BuildFlashWorker(QThread):

    log = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, project_dir: Path, flash: bool):
        super().__init__()
        self.project_dir = project_dir
        self.flash = flash

    def _make_env(self) -> dict:
        """Environment for the subprocess, with CUBE_F4 auto-detected
        if the user hasn't already exported it."""
        env = os.environ.copy()
        if not env.get("CUBE_F4"):
            detected = find_cubef4()
            if detected:
                env["CUBE_F4"] = detected
                self.log.emit(f"# auto-detected CUBE_F4 = {detected}")
            else:
                self.log.emit(
                    "! CUBE_F4 is not set and no STM32CubeF4 install was "
                    "auto-detected. Install the STM32CubeF4 firmware "
                    "package, or set CUBE_F4 by hand before building."
                )
        else:
            self.log.emit(f"# using CUBE_F4 = {env['CUBE_F4']}")
        return env

    def _run(self, cmd: List[str], env: dict) -> int:
        self.log.emit("$ " + " ".join(cmd))
        try:
            p = subprocess.Popen(
                cmd, cwd=str(self.project_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            assert p.stdout is not None
            for line in p.stdout:
                self.log.emit(line.rstrip())
            p.wait()
            return p.returncode
        except FileNotFoundError as e:
            self.log.emit(f"! tool not found: {e}")
            return 127

    def run(self) -> None:
        env = self._make_env()
        rc = self._run(["make", "-j4"], env)
        if rc != 0:
            self.log.emit(f"build failed ({rc})")
            self.done.emit(False)
            return
        if self.flash:
            rc = self._run(["make", "flash"], env)
            if rc != 0:
                self.log.emit(f"flash failed ({rc})")
                self.done.emit(False)
                return
        self.log.emit("OK")
        self.done.emit(True)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.resize(1400, 900)

        self.scene = BlockScene()
        self.view = BlockView(self.scene)

        self.tabs = QTabWidget()
        # Block editor tab
        editor_widget = QWidget()
        editor_layout = QHBoxLayout(editor_widget)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        self.palette = BlockPalette()
        self.palette.setMaximumWidth(180)
        self.param_panel = ParamPanel()
        self.param_panel.setMinimumWidth(240)
        self.scene.block_selected.connect(self.param_panel.set_block)
        self.param_panel.error_block_selected.connect(self._select_block_by_id)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.palette)
        split.addWidget(self.view)
        split.addWidget(self.param_panel)
        split.setStretchFactor(1, 1)
        editor_layout.addWidget(split)

        self.tabs.addTab(editor_widget, "Block Diagram")
        self.python_tab = MatlabWorkspace(root_dir=Path.cwd())
        self.tabs.addTab(self.python_tab, "Python Workspace")
        self.live_scope_tab = LiveScopeTab()
        self.tabs.addTab(self.live_scope_tab, "Live Scope")
        self.sim_scope_tab = SimScopeTab()
        self.sim_scope_tab.sim_btn.clicked.connect(self.on_simulate)
        self.tabs.addTab(self.sim_scope_tab, "Simulate Scope")

        # Debounce timer: re-simulate 800 ms after the last diagram change
        # when the Simulate Scope tab is visible.
        self._sim_debounce = QTimer(self)
        self._sim_debounce.setSingleShot(True)
        self._sim_debounce.setInterval(800)
        self._sim_debounce.timeout.connect(self._auto_simulate)
        self.scene.changed.connect(self._on_scene_changed)
        self.build_log = QPlainTextEdit(readOnly=True)
        self.build_log.setStyleSheet(
            "background: #0d0d0d; color: #b8e0b8; font-family: Consolas, monospace;"
        )
        self.tabs.addTab(self.build_log, "Build Log")

        self.setCentralWidget(self.tabs)

        self._make_toolbar()
        self._make_statusbar()
        self._make_menu()

        # current board
        self.board = "NUCLEO-F446RE"
        self.step_ms = 1  # model step

        # worker
        self._worker: Optional[BuildFlashWorker] = None

        self._example_model()

    # --- UI ---------------------------------------------------------------

    def _make_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel(" Board: "))
        self.board_box = QComboBox()
        for b in BOARDS:
            self.board_box.addItem(b)
        self.board_box.currentTextChanged.connect(self._board_changed)
        tb.addWidget(self.board_box)

        tb.addSeparator()
        tb.addWidget(QLabel(" Step (ms): "))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1000); self.step_spin.setValue(1)
        self.step_spin.valueChanged.connect(lambda v: setattr(self, "step_ms", v))
        tb.addWidget(self.step_spin)

        tb.addSeparator()
        build_act = QAction("Build", self)
        build_act.triggered.connect(lambda: self._build_and_flash(flash=False))
        tb.addAction(build_act)

        self.play_act = QAction("▶ Run on Board", self)
        self.play_act.setToolTip("Generate C code, build with arm-none-eabi-gcc, "
                                 "and flash the binary to the connected STM32.")
        self.play_act.triggered.connect(lambda: self._build_and_flash(flash=True))
        tb.addAction(self.play_act)

        tb.addSeparator()
        sim_act = QAction("Simulate", self)
        sim_act.triggered.connect(self.on_simulate)
        tb.addAction(sim_act)

    def _make_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("Ready.")

    def _make_menu(self) -> None:
        m = self.menuBar()
        file_m = m.addMenu("&File")
        save_a = QAction("Save Model...", self); save_a.triggered.connect(self.save_model)
        open_a = QAction("Open Model...", self); open_a.triggered.connect(self.open_model)
        export_a = QAction("Export Generated C...", self); export_a.triggered.connect(self.export_c)
        file_m.addAction(open_a); file_m.addAction(save_a); file_m.addSeparator()
        file_m.addAction(export_a)
        file_m.addSeparator()
        quit_a = QAction("Quit", self); quit_a.triggered.connect(self.close)
        file_m.addAction(quit_a)

        help_m = m.addMenu("&Help")
        about_a = QAction("About", self)
        about_a.triggered.connect(lambda: QMessageBox.information(
            self, "About",
            f"{APP_NAME} {VERSION}\n\nA Simulink-style visual IDE for STM32.\n"
            "See the docs/ folder for getting started.",
        ))
        help_m.addAction(about_a)

    # --- actions ----------------------------------------------------------

    def _board_changed(self, name: str) -> None:
        self.board = name
        self.statusBar().showMessage(f"Board: {name}")

    def _example_model(self) -> None:
        """Drop a default SquareWave -> GpioOut example so the UI isn't empty."""
        b1 = self.scene.add_block_by_type("SquareWave", QPointF(-300, -50))
        b1.instance.params["frequency_hz"] = "2.0"
        b2 = self.scene.add_block_by_type("GpioOut", QPointF(-50, -50))
        b2.instance.params["pin"] = "PA5"  # green LED
        b3 = self.scene.add_block_by_type("Scope", QPointF(200, -50))
        self.scene.add_connection(b1.output_ports[0], b2.input_ports[0])
        self.scene.add_connection(b1.output_ports[0], b3.input_ports[0])

    def save_model(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "JSON (*.json)")
        if not path:
            return
        data = self.scene.to_model()
        data["board"] = self.board
        data["step_ms"] = self.step_ms
        Path(path).write_text(json.dumps(data, indent=2))
        self.statusBar().showMessage(f"Saved {path}")

    def open_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Model", "", "JSON (*.json)")
        if not path:
            return
        data = json.loads(Path(path).read_text())
        # Rebuild
        self.scene.clear()
        self.scene.__init__()
        self.view.setScene(self.scene)
        self.scene.block_selected.connect(self.param_panel.set_block)
        self._clear_validation_errors()
        positions = {}
        for b in data["blocks"]:
            item = self.scene.add_block_by_type(b["type"], QPointF(b["x"], b["y"]))
            # keep id consistent
            self.scene.blocks.pop(item.instance.block_id, None)
            item.instance.block_id = b["id"]
            item.instance.params.update(b.get("params", {}))
            self.scene.blocks[b["id"]] = item
            positions[b["id"]] = item
        for c in data["connections"]:
            src_item = positions[c["src_block"]]
            dst_item = positions[c["dst_block"]]
            src_port = next(p for p in src_item.output_ports if p.port_name == c["src_port"])
            dst_port = next(p for p in dst_item.input_ports if p.port_name == c["dst_port"])
            self.scene.add_connection(src_port, dst_port)
        self.board = data.get("board", "NUCLEO-F446RE")
        self.board_box.setCurrentText(self.board)
        self.step_ms = int(data.get("step_ms", 1))
        self.step_spin.setValue(self.step_ms)
        self.statusBar().showMessage(f"Loaded {path}")

    # --- simulate / export / build ---------------------------------------

    def _on_scene_changed(self, _=None) -> None:
        if self.tabs.currentWidget() is self.sim_scope_tab:
            self._sim_debounce.start()

    def _auto_simulate(self) -> None:
        self.on_simulate()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _select_block_by_id(self, block_id: str) -> None:
        """Select and centre-view a block by its ID (called from error panel)."""
        item = self.scene.blocks.get(block_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        self.view.centerOn(item)
        self.param_panel.set_block(item)

    def _clear_validation_errors(self) -> None:
        """Remove all error highlights and clear the error panel."""
        for item in self.scene.blocks.values():
            item.set_error_highlight(False)
        self.param_panel.clear_errors()

    def _show_validation_errors(self, errors: List[ValidationError]) -> None:
        """Highlight faulty blocks and populate the error panel."""
        bad_ids = {e.block_id for e in errors}
        for bid, item in self.scene.blocks.items():
            item.set_error_highlight(bid in bad_ids)
        self.param_panel.show_errors(errors)
        # Switch to the Block Diagram tab so the user sees the highlights
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            f"Simulation blocked — {len(errors)} parameter error"
            f"{'s' if len(errors) != 1 else ''} found."
        )

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------

    def on_simulate(self) -> None:
        model = self.scene.to_model()
        model["board"] = self.board
        model["step_ms"] = self.step_ms

        # Always clear previous error state first.
        self._clear_validation_errors()

        # Validate all block parameters before touching the simulator.
        errors = validate_model(model, WORKSPACE)
        if errors:
            self._show_validation_errors(errors)
            return

        duration = self.sim_scope_tab.duration_spin.value()
        try:
            t, sigs = simulate_model(model, duration_s=duration,
                                     step_s=self.step_ms / 1000.0)
        except Exception as e:
            QMessageBox.critical(self, "Simulation error", str(e))
            return
        self.sim_scope_tab.show_simulation(t, sigs)
        self.tabs.setCurrentWidget(self.sim_scope_tab)
        # Collect the names written by ToWorkspace blocks, refresh the
        # variable table, and echo the names to the command window so the
        # user knows what landed in the workspace.
        tw_vars: List[str] = []
        for b in model["blocks"]:
            if b["type"] == "ToWorkspace":
                vn = b["params"].get("variable_name", "yout").strip() or "yout"
                tw_vars.append(vn)
                if int(float(b["params"].get("save_time", "1"))):
                    tw_vars.append(f"{vn}_t")
        self.python_tab.refresh_workspace(tw_vars or None)

    def export_c(self) -> None:
        target = QFileDialog.getExistingDirectory(self, "Export project to folder")
        if not target:
            return
        self._generate_project(Path(target))
        self.statusBar().showMessage(f"Exported to {target}")

    def _generate_project(self, out_dir: Path) -> Path:
        model = self.scene.to_model()
        model["board"] = self.board
        model["step_ms"] = self.step_ms
        proj = generate_project(out_dir, model, WORKSPACE)
        return proj

    def _build_and_flash(self, flash: bool) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "Build already in progress.")
            return
        self.tabs.setCurrentWidget(self.build_log)
        self.build_log.clear()
        out_dir = Path.cwd() / "build_stm32_ide"
        out_dir.mkdir(exist_ok=True)
        try:
            proj = self._generate_project(out_dir)
        except Exception as e:
            self.build_log.appendPlainText(f"codegen error: {e}")
            return
        self.build_log.appendPlainText(f"Generated project: {proj}")
        self._worker = BuildFlashWorker(proj, flash=flash)
        self._worker.log.connect(self.build_log.appendPlainText)
        self._worker.done.connect(lambda ok: self.statusBar().showMessage(
            "Flash succeeded." if ok else "Build/flash failed."
        ))
        self._worker.start()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

