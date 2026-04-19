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


from code_templates import generate_project, BOARDS, _topo_order as _ct_topo_order
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
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setPos(instance.x, instance.y)
        self._make_ports()

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
        painter.drawText(
            self.rect().adjusted(5, 5, -5, -self.HEIGHT/2),
            Qt.AlignLeft | Qt.AlignTop, self.instance.spec.display_name,
        )
        f2 = QFont(); f2.setPointSize(8)
        painter.setFont(f2)
        painter.drawText(
            self.rect().adjusted(5, self.HEIGHT/2, -5, -5),
            Qt.AlignLeft | Qt.AlignBottom, f"#{self.instance.block_id}",
        )

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

    def __init__(self):
        super().__init__()
        self._block: Optional[BlockItem] = None
        self.layout_ = QVBoxLayout(self)
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
            # No real sensor on the host. Allow override via workspace variable
            # ``ultrasonic_<id>`` (scalar or array of meters) for what-if tests.
            override = WORKSPACE.globals.get(f"ultrasonic_{b['id']}")
            if override is None:
                outs[(b["id"], "d")] = np.zeros(n)
            else:
                arr = np.asarray(override, dtype=float)
                if arr.ndim == 0:
                    outs[(b["id"], "d")] = np.full(n, float(arr))
                else:
                    outs[(b["id"], "d")] = arr[:n]

    # Build wire map once for all subsequent passes.
    wires: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for c in model["connections"]:
        wires[(c["dst_block"], c["dst_port"])] = (c["src_block"], c["src_port"])

    # Middle pass: through blocks (Sum, Product) in topological order so that
    # chains (e.g. Sum → Product) are resolved correctly.
    for b in _ct_topo_order(model):
        if b["type"] == "Sum":
            src0 = wires.get((b["id"], "u0"))
            src1 = wires.get((b["id"], "u1"))
            arr0 = outs.get(src0, np.zeros(n)) if src0 else np.zeros(n)
            arr1 = outs.get(src1, np.zeros(n)) if src1 else np.zeros(n)
            outs[(b["id"], "y")] = arr0 + arr1
        elif b["type"] == "Product":
            src0 = wires.get((b["id"], "u0"))
            src1 = wires.get((b["id"], "u1"))
            arr0 = outs.get(src0, np.ones(n)) if src0 else np.ones(n)
            arr1 = outs.get(src1, np.ones(n)) if src1 else np.ones(n)
            outs[(b["id"], "y")] = arr0 * arr1

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

    def on_simulate(self) -> None:
        model = self.scene.to_model()
        model["board"] = self.board
        model["step_ms"] = self.step_ms
        duration = self.sim_scope_tab.duration_spin.value()
        try:
            t, sigs = simulate_model(model, duration_s=duration,
                                     step_s=self.step_ms / 1000.0)
        except Exception as e:
            QMessageBox.critical(self, "Simulation error", str(e))
            return
        self.sim_scope_tab.show_simulation(t, sigs)
        self.tabs.setCurrentWidget(self.sim_scope_tab)

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

