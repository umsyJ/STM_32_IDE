"""
MATLAB-style Python workspace widget.

Four panels laid out like MATLAB's default desktop:

    +------------+-------------------------------+------------+
    |            | Editor (tabs, line numbers,   |            |
    |   Current  |   Python syntax highlight)    | Workspace  |
    |   Folder   |-------------------------------|  variables |
    |  (files)   | Command window (REPL)         |   (table)  |
    +------------+-------------------------------+------------+

Everything shares the global :data:`workspace_shared.WORKSPACE`, so any
variable defined in the command window or by running a script in the
editor is instantly available to block parameters and to the host-side
simulator.

Keyboard shortcuts
    F5              Run the current editor buffer
    F9              Run the currently selected text
    Ctrl+N / O / S  New / Open / Save in the editor
    Up / Down       History in the command window
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QDir, QRegularExpression, QRect, QSize, Qt
from PyQt5.QtGui import (
    QColor, QFont, QKeySequence, QPainter, QSyntaxHighlighter,
    QTextCharFormat, QTextCursor,
)
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QFileDialog, QFileSystemModel, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox, QPlainTextEdit,
    QPushButton, QShortcut, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QToolBar, QTreeView, QVBoxLayout, QWidget,
)

import numpy as np

from workspace_shared import WORKSPACE


# ---------------------------------------------------------------------------
# Python syntax highlighter (simple but reasonable)
# ---------------------------------------------------------------------------

_PY_KEYWORDS = (
    "and as assert async await break class continue def del elif else "
    "except finally for from global if import in is lambda nonlocal not "
    "or pass raise return try while with yield True False None"
).split()


class PythonHighlighter(QSyntaxHighlighter):
    """Lightweight Python highlighter sufficient for an editor pane."""

    def __init__(self, document):
        super().__init__(document)
        kw_fmt = QTextCharFormat(); kw_fmt.setForeground(QColor("#569cd6")); kw_fmt.setFontWeight(QFont.Bold)
        builtin_fmt = QTextCharFormat(); builtin_fmt.setForeground(QColor("#4ec9b0"))
        str_fmt = QTextCharFormat(); str_fmt.setForeground(QColor("#ce9178"))
        num_fmt = QTextCharFormat(); num_fmt.setForeground(QColor("#b5cea8"))
        com_fmt = QTextCharFormat(); com_fmt.setForeground(QColor("#6a9955")); com_fmt.setFontItalic(True)
        dec_fmt = QTextCharFormat(); dec_fmt.setForeground(QColor("#dcdcaa"))

        self._rules: List[tuple] = []
        for kw in _PY_KEYWORDS:
            self._rules.append((QRegularExpression(r"\b" + kw + r"\b"), kw_fmt))
        for b in ("print", "len", "range", "list", "dict", "tuple", "set", "int",
                  "float", "str", "bool", "abs", "min", "max", "sum", "enumerate",
                  "zip", "map", "filter", "open", "type", "isinstance"):
            self._rules.append((QRegularExpression(r"\b" + b + r"\b"), builtin_fmt))
        self._rules.append((QRegularExpression(r"@\w+"), dec_fmt))
        self._rules.append((QRegularExpression(r"\b\d+(\.\d+)?([eE][+-]?\d+)?\b"), num_fmt))
        self._rules.append((QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), str_fmt))
        self._rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), str_fmt))
        self._rules.append((QRegularExpression(r"#[^\n]*"), com_fmt))

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)


# ---------------------------------------------------------------------------
# Line-numbered Python editor (standard Qt pattern)
# ---------------------------------------------------------------------------

class _LineNumberArea(QWidget):
    def __init__(self, editor: "PythonEditor"):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, ev) -> None:
        self.editor.line_number_area_paint(ev)


class PythonEditor(QPlainTextEdit):
    """Editable source pane with line numbers and Python highlighting."""

    def __init__(self, file_path: Optional[Path] = None):
        super().__init__()
        font = QFont("Consolas", 11)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.setStyleSheet(
            "QPlainTextEdit { background:#1e1e1e; color:#d4d4d4; "
            "selection-background-color:#264f78; }"
        )

        self._line_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_area_width)
        self.updateRequest.connect(self._update_line_area)
        self._update_line_area_width(0)

        self.highlighter = PythonHighlighter(self.document())
        self.file_path: Optional[Path] = file_path
        self._dirty = False
        self.textChanged.connect(self._on_changed)

        if file_path and Path(file_path).exists():
            self.setPlainText(Path(file_path).read_text(encoding="utf-8"))
            self._dirty = False

    # --- dirty tracking ---------------------------------------------------

    def _on_changed(self) -> None:
        self._dirty = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    def save_to(self, path: Path) -> None:
        path.write_text(self.toPlainText(), encoding="utf-8")
        self.file_path = path
        self._dirty = False

    # --- line numbers -----------------------------------------------------

    def line_number_area_width(self) -> int:
        digits = max(3, len(str(max(1, self.blockCount()))))
        return 8 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_area_width(self, _n: int) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_area(self, rect: QRect, dy: int) -> None:
        if dy:
            self._line_area.scroll(0, dy)
        else:
            self._line_area.update(0, rect.y(), self._line_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_area_width(0)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        cr = self.contentsRect()
        self._line_area.setGeometry(QRect(cr.left(), cr.top(),
                                          self.line_number_area_width(), cr.height()))

    def line_number_area_paint(self, ev) -> None:
        painter = QPainter(self._line_area)
        painter.fillRect(ev.rect(), QColor("#252526"))
        block = self.firstVisibleBlock()
        num = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        painter.setPen(QColor("#858585"))
        fm = self.fontMetrics()
        while block.isValid() and top <= ev.rect().bottom():
            if block.isVisible() and bottom >= ev.rect().top():
                painter.drawText(
                    0, top, self._line_area.width() - 4, fm.height(),
                    Qt.AlignRight, str(num + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            num += 1


# ---------------------------------------------------------------------------
# Command window (REPL) — same shared globals as the editor
# ---------------------------------------------------------------------------

class CommandWindow(QWidget):
    """A MATLAB-style command window backed by WORKSPACE.globals."""

    def __init__(self, on_changed=None):
        super().__init__()
        self._on_changed = on_changed or (lambda: None)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        header = QLabel("<b>Command Window</b>")
        header.setTextFormat(Qt.RichText)
        lay.addWidget(header)

        self.output = QPlainTextEdit(readOnly=True)
        self.output.setStyleSheet(
            "QPlainTextEdit { background:#1e1e1e; color:#d4d4d4; "
            "font-family: Consolas, monospace; }"
        )
        lay.addWidget(self.output, 1)

        self.input = QLineEdit()
        self.input.setPlaceholderText(">> enter a Python statement   (↑/↓ for history)")
        self.input.setStyleSheet(
            "QLineEdit { font-family: Consolas, monospace; background:#252526; color:#d4d4d4; }"
        )
        self.input.returnPressed.connect(self._on_submit)
        self.input.installEventFilter(self)
        lay.addWidget(self.input)

        self._history: List[str] = []
        self._hist_index = 0

    # --- history ----------------------------------------------------------

    def eventFilter(self, obj, ev):
        from PyQt5.QtCore import QEvent
        if obj is self.input and ev.type() == QEvent.KeyPress:
            if ev.key() == Qt.Key_Up and self._history:
                self._hist_index = max(0, self._hist_index - 1)
                self.input.setText(self._history[self._hist_index])
                return True
            if ev.key() == Qt.Key_Down and self._history:
                self._hist_index = min(len(self._history), self._hist_index + 1)
                if self._hist_index == len(self._history):
                    self.input.clear()
                else:
                    self.input.setText(self._history[self._hist_index])
                return True
        return super().eventFilter(obj, ev)

    # --- exec -------------------------------------------------------------

    def _on_submit(self) -> None:
        src = self.input.text().strip()
        if not src:
            return
        self.output.appendPlainText(f">> {src}")
        self.input.clear()
        self._history.append(src)
        self._hist_index = len(self._history)
        self.run_snippet(src, echo_header=False)

    def run_snippet(self, src: str, echo_header: bool = True,
                    label: Optional[str] = None) -> None:
        """Exec a chunk of source in the shared workspace.

        First tries eval() so simple expressions echo their value
        MATLAB-style; falls back to exec() for statements.
        """
        if echo_header:
            tag = f"% Running {label}" if label else "% (running snippet)"
            self.output.appendPlainText(tag)
        try:
            try:
                val = eval(src, WORKSPACE.globals)
                if val is not None:
                    self.output.appendPlainText(repr(val))
            except SyntaxError:
                exec(src, WORKSPACE.globals)
        except Exception:
            self.output.appendPlainText(traceback.format_exc().rstrip())
        self.output.moveCursor(QTextCursor.End)
        self._on_changed()


# ---------------------------------------------------------------------------
# Workspace variables table
# ---------------------------------------------------------------------------

# Names that exist in WORKSPACE.globals but shouldn't be shown to the user.
_HIDDEN = {"np", "math", "pi"}


def _describe(name: str, value) -> tuple:
    """Return (value_str, class_str, size_str) for display."""
    cls = type(value).__name__
    if isinstance(value, np.ndarray):
        size = "x".join(str(s) for s in value.shape) or "scalar"
        if value.size <= 8:
            v = np.array2string(value, precision=4, separator=",")
        else:
            v = f"<array {value.shape} {value.dtype}>"
        return v, f"ndarray[{value.dtype}]", size
    if isinstance(value, (list, tuple, set)):
        try:
            v = repr(value)
            if len(v) > 60:
                v = v[:57] + "..."
        except Exception:
            v = f"<{cls}>"
        return v, cls, str(len(value))
    if isinstance(value, dict):
        return f"<dict[{len(value)}]>", cls, str(len(value))
    try:
        v = repr(value)
        if len(v) > 60:
            v = v[:57] + "..."
    except Exception:
        v = "<unrepr>"
    return v, cls, "1"


class VariableTable(QTableWidget):
    """Live view of WORKSPACE.globals."""

    def __init__(self, command_window: CommandWindow):
        super().__init__()
        self.command_window = command_window
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Name", "Value", "Class", "Size"])
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.cellDoubleClicked.connect(self._dbl)

    def refresh(self) -> None:
        items = []
        for k, v in WORKSPACE.globals.items():
            if k.startswith("_") or k in _HIDDEN:
                continue
            if callable(v) and getattr(v, "__module__", "") in ("builtins", None):
                continue
            items.append((k, v))
        items.sort(key=lambda x: x[0].lower())
        self.setRowCount(len(items))
        for row, (name, value) in enumerate(items):
            val, cls, size = _describe(name, value)
            self.setItem(row, 0, QTableWidgetItem(name))
            self.setItem(row, 1, QTableWidgetItem(val))
            self.setItem(row, 2, QTableWidgetItem(cls))
            self.setItem(row, 3, QTableWidgetItem(size))

    def _dbl(self, row: int, _col: int) -> None:
        name = self.item(row, 0).text()
        self.command_window.input.setText(name)
        self.command_window.input.setFocus()

    def _menu(self, pos) -> None:
        m = QMenu(self)
        clear_one = m.addAction("Clear selected")
        clear_all = m.addAction("Clear all")
        plot = m.addAction("Plot (if array)")
        act = m.exec_(self.viewport().mapToGlobal(pos))
        if act is clear_one:
            for i in self.selectionModel().selectedRows():
                name = self.item(i.row(), 0).text()
                WORKSPACE.globals.pop(name, None)
            self.refresh()
        elif act is clear_all:
            keep = {k: WORKSPACE.globals[k] for k in _HIDDEN if k in WORKSPACE.globals}
            WORKSPACE.globals.clear()
            WORKSPACE.globals.update(keep)
            self.refresh()
        elif act is plot:
            rows = self.selectionModel().selectedRows()
            if not rows:
                return
            name = self.item(rows[0].row(), 0).text()
            self.command_window.run_snippet(
                f"import matplotlib.pyplot as __plt; __plt.figure(); "
                f"__plt.plot({name}); __plt.title('{name}'); __plt.show()",
                echo_header=False,
            )


# ---------------------------------------------------------------------------
# Full MATLAB-style widget
# ---------------------------------------------------------------------------

class MatlabWorkspace(QWidget):
    """Top-level replacement for the old Python tab."""

    def __init__(self, root_dir: Optional[Path] = None):
        super().__init__()
        self.root_dir = Path(root_dir or os.getcwd()).resolve()

        # --- Current Folder (left) --------------------------------------
        self.fs_model = QFileSystemModel()
        self.fs_model.setRootPath(str(self.root_dir))
        self.fs_model.setNameFilters(["*.py", "*.json", "*.md", "*.txt"])
        self.fs_model.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.fs_model)
        self.tree.setRootIndex(self.fs_model.index(str(self.root_dir)))
        for col in (1, 2, 3):
            self.tree.setColumnHidden(col, True)
        self.tree.setHeaderHidden(True)
        self.tree.doubleClicked.connect(self._on_tree_dbl)
        folder_label = QLabel(f"<b>Current Folder</b><br><small>{self.root_dir}</small>")
        folder_label.setTextFormat(Qt.RichText)
        folder_label.setWordWrap(True)
        folder_col = QWidget(); fl = QVBoxLayout(folder_col)
        fl.setContentsMargins(4, 4, 4, 4)
        fl.addWidget(folder_label)
        fl.addWidget(self.tree, 1)

        # --- Editor + command window (center) ---------------------------
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self._close_tab)

        tb = QToolBar("Editor")
        new_act = QAction("New", self); new_act.triggered.connect(self._new_file); tb.addAction(new_act)
        open_act = QAction("Open...", self); open_act.triggered.connect(self._open_file); tb.addAction(open_act)
        save_act = QAction("Save", self); save_act.triggered.connect(self._save_file); tb.addAction(save_act)
        saveas_act = QAction("Save As...", self); saveas_act.triggered.connect(self._save_as); tb.addAction(saveas_act)
        tb.addSeparator()
        run_act = QAction("Run (F5)", self); run_act.triggered.connect(self.run_current); tb.addAction(run_act)
        runsel_act = QAction("Run Selection (F9)", self); runsel_act.triggered.connect(self.run_selection); tb.addAction(runsel_act)

        editor_host = QWidget(); eh = QVBoxLayout(editor_host)
        eh.setContentsMargins(0, 0, 0, 0); eh.setSpacing(0)
        eh.addWidget(tb); eh.addWidget(self.editor_tabs, 1)

        self.command = CommandWindow(on_changed=self._refresh_vars)

        center_split = QSplitter(Qt.Vertical)
        center_split.addWidget(editor_host)
        center_split.addWidget(self.command)
        center_split.setStretchFactor(0, 2)
        center_split.setStretchFactor(1, 1)

        # --- Variables table (right) ------------------------------------
        self.var_table = VariableTable(self.command)
        var_host = QWidget(); vh = QVBoxLayout(var_host)
        vh.setContentsMargins(4, 4, 4, 4)
        vh.addWidget(QLabel("<b>Workspace</b>"))
        vh.addWidget(self.var_table, 1)

        # --- Outer horizontal splitter ----------------------------------
        outer = QSplitter(Qt.Horizontal)
        outer.addWidget(folder_col)
        outer.addWidget(center_split)
        outer.addWidget(var_host)
        outer.setSizes([220, 900, 300])

        main = QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(outer)

        # Keyboard shortcuts
        QShortcut(QKeySequence("F5"), self, activated=self.run_current)
        QShortcut(QKeySequence("F9"), self, activated=self.run_selection)
        QShortcut(QKeySequence.New, self, activated=self._new_file)
        QShortcut(QKeySequence.Open, self, activated=self._open_file)
        QShortcut(QKeySequence.Save, self, activated=self._save_file)

        # Start with one blank script so the area isn't empty
        self._new_file()
        self.var_table.refresh()

        self.command.output.appendPlainText(
            "% Python workspace ready. Variables defined here (or by the editor)\n"
            "% are visible to every block parameter in the diagram.\n"
            "% Try:  Ts = 0.001     then reference 'Ts' in any parameter field."
        )

    # --- callbacks --------------------------------------------------------

    def _refresh_vars(self) -> None:
        self.var_table.refresh()

    # --- editor management ------------------------------------------------

    def _current_editor(self) -> Optional[PythonEditor]:
        return self.editor_tabs.currentWidget()

    def _tab_title(self, editor: PythonEditor) -> str:
        if editor.file_path is None:
            return "untitled.py"
        return editor.file_path.name

    def _refresh_tab_titles(self) -> None:
        for i in range(self.editor_tabs.count()):
            ed = self.editor_tabs.widget(i)
            star = "*" if ed.dirty else ""
            self.editor_tabs.setTabText(i, star + self._tab_title(ed))

    def _new_file(self) -> None:
        ed = PythonEditor()
        idx = self.editor_tabs.addTab(ed, "untitled.py")
        self.editor_tabs.setCurrentIndex(idx)
        ed.textChanged.connect(self._refresh_tab_titles)

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Python File", str(self.root_dir),
            "Python (*.py);;All files (*)",
        )
        if not path:
            return
        self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        # If already open, focus that tab
        for i in range(self.editor_tabs.count()):
            ed = self.editor_tabs.widget(i)
            if ed.file_path and ed.file_path.resolve() == path.resolve():
                self.editor_tabs.setCurrentIndex(i)
                return
        ed = PythonEditor(file_path=path)
        idx = self.editor_tabs.addTab(ed, path.name)
        ed.textChanged.connect(self._refresh_tab_titles)
        self.editor_tabs.setCurrentIndex(idx)

    def _save_file(self) -> None:
        ed = self._current_editor()
        if ed is None:
            return
        if ed.file_path is None:
            self._save_as()
            return
        ed.save_to(ed.file_path)
        self._refresh_tab_titles()

    def _save_as(self) -> None:
        ed = self._current_editor()
        if ed is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Python File", str(self.root_dir),
            "Python (*.py);;All files (*)",
        )
        if not path:
            return
        ed.save_to(Path(path))
        self._refresh_tab_titles()

    def _close_tab(self, idx: int) -> None:
        ed = self.editor_tabs.widget(idx)
        if ed is None:
            return
        if ed.dirty:
            r = QMessageBox.question(
                self, "Unsaved changes",
                f"Save changes to {self._tab_title(ed)} before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if r == QMessageBox.Cancel:
                return
            if r == QMessageBox.Save:
                self._save_file()
        self.editor_tabs.removeTab(idx)

    def _on_tree_dbl(self, idx) -> None:
        path = Path(self.fs_model.filePath(idx))
        if path.is_file() and path.suffix == ".py":
            self.open_path(path)

    # --- run --------------------------------------------------------------

    def run_current(self) -> None:
        ed = self._current_editor()
        if ed is None:
            return
        src = ed.toPlainText()
        if not src.strip():
            return
        label = self._tab_title(ed)
        self.command.run_snippet(src, echo_header=True, label=label)

    def run_selection(self) -> None:
        ed = self._current_editor()
        if ed is None:
            return
        sel = ed.textCursor().selectedText()
        if not sel:
            # If no selection, run the current line instead
            cur = ed.textCursor()
            cur.select(QTextCursor.LineUnderCursor)
            sel = cur.selectedText()
        # Qt uses u2029 for line breaks inside a selection
        sel = sel.replace("\u2029", "\n")
        if sel.strip():
            self.command.run_snippet(sel, echo_header=True, label="selection")
