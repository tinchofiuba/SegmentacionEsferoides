"""View de la app: ventana principal PyQt5. No contiene lógica de modelo —
todo lo que sabe de segmentación lo pide a través de SegmentationModel
(app/model.py), así que esta ventana se podría reemplazar por otra View
(web, CLI) sin tocar esa capa.
"""

import json
import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QShortcut,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.editing import (
    CLASS_NAMES,
    UndoStack,
    fill_holes,
    paint_segment,
    replace_object,
    save_mask,
)
from app.image_canvas import (
    TOOL_BRUSH,
    TOOL_DELETE,
    TOOL_ERASER,
    TOOL_RECLASS,
    TOOL_ZONE,
    ImageCanvas,
)
from app.model import PredictionResult, SegmentationModel
from app.viewport import build_pyramid


class PredictWorker(QThread):
    finished_ok = pyqtSignal(object)  # PredictionResult
    failed = pyqtSignal(str)

    def __init__(self, model, image_rgb, magnification, bbox):
        super().__init__()
        self.model = model
        self.image_rgb = image_rgb
        self.magnification = magnification
        self.bbox = bbox

    def run(self):
        try:
            result: PredictionResult = self.model.predict(
                self.image_rgb, magnification=self.magnification, bbox=self.bbox
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario, no se traga
            self.failed.emit(str(exc))


MASKS_DIR = os.path.join("data", "insitu_3t3", "masks")

TOOL_LABELS = [
    (TOOL_ZONE, "Zona (Z)"),
    (TOOL_BRUSH, "Pincel (B)"),
    (TOOL_ERASER, "Borrador (E)"),
    (TOOL_DELETE, "Borrar objeto (D)"),
    (TOOL_RECLASS, "Cambiar clase de objeto (R)"),
]


class MainWindow(QMainWindow):
    def __init__(self, model: SegmentationModel):
        super().__init__()
        self.model = model
        self.full_res_image = None  # np.ndarray RGB, resolución nativa
        self.image_path = None
        self.mask = None            # np.ndarray uint8 clases 0-3, resolución nativa
        self.undo_stack = UndoStack()
        self.worker = None
        self.pending_bbox = None    # bbox (resolución nativa) de la predicción en curso
        self.origin = "manual"      # manual | modelo | modelo_corregido
        self.dirty = False

        self.setWindowTitle("Predicción y tagging de esferoides")
        self._build_ui()
        self._build_shortcuts()

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top = QHBoxLayout()
        layout.addLayout(top)
        open_button = QPushButton("Abrir imagen...")
        open_button.clicked.connect(self._on_open_image)
        top.addWidget(open_button)
        self.accept_button = QPushButton("Aceptar y guardar máscara")
        self.accept_button.clicked.connect(self._on_accept)
        self.accept_button.setEnabled(False)
        top.addWidget(self.accept_button)
        self.undo_button = QPushButton("Deshacer (Ctrl+Z)")
        self.undo_button.clicked.connect(self._undo)
        top.addWidget(self.undo_button)
        self.redo_button = QPushButton("Rehacer (Ctrl+Y)")
        self.redo_button.clicked.connect(self._redo)
        top.addWidget(self.redo_button)
        top.addStretch()

        self.canvas = ImageCanvas()
        self.canvas.stroke_started.connect(self._on_stroke_started)
        self.canvas.stroke_segment.connect(self._on_stroke_segment)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.stroke_finished.connect(self._on_stroke_finished)
        self.canvas.object_clicked.connect(self._on_object_clicked)
        layout.addWidget(self.canvas, stretch=1)

        tools = QHBoxLayout()
        layout.addLayout(tools)
        self.tool_buttons = {}
        tool_group = QButtonGroup(self)
        for tool, label in TOOL_LABELS:
            btn = QRadioButton(label)
            btn.toggled.connect(lambda checked, t=tool: checked and self._set_tool(t))
            tool_group.addButton(btn)
            tools.addWidget(btn)
            self.tool_buttons[tool] = btn
        self.tool_buttons[TOOL_ZONE].setChecked(True)

        brush = QHBoxLayout()
        layout.addLayout(brush)
        brush.addWidget(QLabel("Clase a pintar / asignar:"))
        self.combo_class = QComboBox()
        for cid, name in CLASS_NAMES.items():
            self.combo_class.addItem(f"{cid} - {name}", cid)
        self.combo_class.setCurrentIndex(1)  # esferoides
        brush.addWidget(self.combo_class)
        self.slider_brush, self.label_brush = self._add_size_slider(brush, "Pincel", 10, self._on_brush_size)
        self.slider_eraser, self.label_eraser = self._add_size_slider(brush, "Borrador", 3, self._on_eraser_size)
        brush.addWidget(QLabel("Transparencia máscara:"))
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(50)
        self.slider_alpha.valueChanged.connect(self._refresh_canvas)
        brush.addWidget(self.slider_alpha)
        self.check_show_mask = QCheckBox("Ver máscara (M)")
        self.check_show_mask.setChecked(True)
        self.check_show_mask.toggled.connect(self._refresh_canvas)
        brush.addWidget(self.check_show_mask)
        self.label_zoom = QLabel("Zoom: -")
        brush.addWidget(self.label_zoom)
        fit_button = QPushButton("Ajustar (F)")
        fit_button.clicked.connect(self.canvas.fit_view)
        brush.addWidget(fit_button)
        fill_button = QPushButton("Rellenar huecos (clase)")
        fill_button.clicked.connect(self._on_fill_holes)
        brush.addWidget(fill_button)

        predict = QHBoxLayout()
        layout.addLayout(predict)
        self.radio_full = QRadioButton("Predecir toda la imagen")
        self.radio_zone = QRadioButton("Predecir zona seleccionada")
        self.radio_full.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.radio_full)
        mode_group.addButton(self.radio_zone)
        predict.addWidget(self.radio_full)
        predict.addWidget(self.radio_zone)
        predict.addWidget(QLabel("Magnificación:"))
        self.combo_magnification = QComboBox()
        self.combo_magnification.addItems(["4x", "10x"])
        predict.addWidget(self.combo_magnification)
        self.predict_button = QPushButton("Predecir")
        self.predict_button.clicked.connect(self._on_predict_clicked)
        self.predict_button.setEnabled(False)
        predict.addWidget(self.predict_button)
        predict.addStretch()

        self.status_label = QLabel("Abrí una imagen para empezar.")
        layout.addWidget(self.status_label)

    def _build_shortcuts(self):
        def sc(key, fn):
            QShortcut(QKeySequence(key), self, activated=fn)

        sc("Ctrl+Z", self._undo)
        sc("Ctrl+Y", self._redo)
        sc("Ctrl+S", self._on_accept)
        sc("Z", lambda: self.tool_buttons[TOOL_ZONE].setChecked(True))
        sc("B", lambda: self.tool_buttons[TOOL_BRUSH].setChecked(True))
        sc("E", lambda: self.tool_buttons[TOOL_ERASER].setChecked(True))
        sc("D", lambda: self.tool_buttons[TOOL_DELETE].setChecked(True))
        sc("R", lambda: self.tool_buttons[TOOL_RECLASS].setChecked(True))
        sc("M", lambda: self.check_show_mask.toggle())
        sc("F", self.canvas.fit_view)
        sc("]", lambda: self._scale_active_size(1.25))
        sc("[", lambda: self._scale_active_size(0.8))
        for i in (1, 2, 3):
            sc(str(i), lambda i=i: self.combo_class.setCurrentIndex(i - 1))

    def _add_size_slider(self, layout, name, value, on_change):
        layout.addWidget(QLabel(f"{name}:"))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, 150)
        slider.setValue(value)
        label = QLabel(f"{value}px")
        layout.addWidget(slider)
        layout.addWidget(label)
        # El callback usa self.label_*, que recién existen cuando esto retorna:
        # se conecta a una función que actualiza la etiqueta local.
        def changed(v):
            label.setText(f"{v}px")
            on_change(v)

        slider.valueChanged.connect(changed)
        on_change(value)
        return slider, label

    def _on_brush_size(self, value):
        self.canvas.brush_radius = value
        self.canvas.update()

    def _on_eraser_size(self, value):
        self.canvas.eraser_radius = value
        self.canvas.update()

    def _scale_active_size(self, factor):
        slider = self.slider_eraser if self.canvas.tool == TOOL_ERASER else self.slider_brush
        v = slider.value()
        scaled = round(v * factor)
        if scaled == v:  # a tamaños chicos, el factor no mueve el entero
            scaled += 1 if factor > 1 else -1
        slider.setValue(max(1, scaled))

    def _on_zoom_changed(self, zoom):
        self.label_zoom.setText(f"Zoom: {zoom * 100:.0f}%")

    def _set_mask(self, mask):
        self.mask = mask
        self.canvas.mask = mask
        self.canvas.update()

    def _set_tool(self, tool):
        self.canvas.set_tool(tool)

    # ---------- Imagen / render ----------
    def _on_open_image(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", "", "Imágenes (*.png *.tif *.tiff *.jpg *.jpeg)"
        )
        if not path:
            return

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            QMessageBox.warning(self, "Error", f"No se pudo leer la imagen:\n{path}")
            return

        self.image_path = path
        self.full_res_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        full_h, full_w = self.full_res_image.shape[:2]
        self.mask = np.zeros((full_h, full_w), dtype=np.uint8)
        self.undo_stack.clear()
        self.origin = "manual"
        self.dirty = False

        self.canvas.set_image(build_pyramid(self.full_res_image), self.mask)

        self.predict_button.setEnabled(True)
        self.accept_button.setEnabled(True)
        self.status_label.setText(
            f"Imagen cargada: {full_w}x{full_h}px. Rueda = zoom, botón derecho = mover."
        )

    def _refresh_canvas(self, *_):
        self.canvas.alpha = self.slider_alpha.value() / 100
        self.canvas.show_mask = self.check_show_mask.isChecked()
        self.canvas.update()

    # ---------- Edición ----------
    def _selected_class(self):
        return self.combo_class.currentData()

    def _mark_edited(self):
        self.dirty = True
        if self.origin == "modelo":
            self.origin = "modelo_corregido"

    def _on_stroke_started(self):
        self.undo_stack.push(self.mask)

    def _on_stroke_segment(self, x0, y0, x1, y1):
        if self.mask is None:
            return
        if self.canvas.tool == TOOL_ERASER:
            class_id, radius = 0, self.canvas.eraser_radius
        else:
            class_id, radius = self._selected_class(), self.canvas.brush_radius
        paint_segment(self.mask, x0, y0, x1, y1, radius, class_id)
        self.canvas.update()

    def _on_stroke_finished(self):
        self._mark_edited()

    def _on_object_clicked(self, x, y):
        if self.mask is None:
            return
        new_class = 0 if self.canvas.tool == TOOL_DELETE else self._selected_class()
        self.undo_stack.push(self.mask)
        if replace_object(self.mask, x, y, new_class):
            self._mark_edited()
            self.canvas.update()
        else:
            self.undo_stack.discard_last()  # no cambió nada: descartar el snapshot
            self.status_label.setText("No hay ningún objeto bajo el cursor.")

    def _on_fill_holes(self):
        if self.mask is None:
            return
        self.undo_stack.push(self.mask)
        fill_holes(self.mask, self._selected_class())
        self._mark_edited()
        self.canvas.update()

    def _undo(self):
        self._restore(self.undo_stack.undo(self.mask) if self.mask is not None else None)

    def _redo(self):
        self._restore(self.undo_stack.redo(self.mask) if self.mask is not None else None)

    def _restore(self, mask):
        if mask is None:
            return
        self._set_mask(mask)
        self.dirty = True

    # ---------- Guardado ----------
    def _on_accept(self):
        if self.mask is None:
            return
        if not np.any(self.mask):
            reply = QMessageBox.question(
                self, "Máscara vacía", "La máscara no tiene ningún objeto. ¿Guardarla igual?"
            )
            if reply != QMessageBox.Yes:
                return
        stem = os.path.splitext(os.path.basename(self.image_path))[0]
        default_dir = MASKS_DIR if os.path.isdir(MASKS_DIR) else os.path.dirname(self.image_path)
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar máscara", os.path.join(default_dir, f"{stem}_Taggeada.png"), "PNG (*.png)"
        )
        if not path:
            return
        try:
            save_mask(path, self.mask)
            # Registro de origen: una máscara que salió del modelo no debería
            # usarse para evaluarlo sin saberlo.
            with open(os.path.splitext(path)[0] + ".origin.json", "w") as f:
                json.dump({"origin": self.origin, "image": os.path.basename(self.image_path),
                           "model_run": self.model.run_dir}, f, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Error al guardar", str(exc))
            return
        self.dirty = False
        self.status_label.setText(f"Máscara guardada en {path} (origen: {self.origin})")

    def _confirm_discard(self):
        if not self.dirty:
            return True
        reply = QMessageBox.question(
            self, "Cambios sin guardar", "Hay cambios sin guardar en la máscara. ¿Descartarlos?"
        )
        return reply == QMessageBox.Yes

    def closeEvent(self, event):
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()

    # ---------- Predicción ----------
    def _on_predict_clicked(self):
        if self.full_res_image is None:
            return

        bbox = None
        if self.radio_zone.isChecked():
            sel = self.canvas.selection
            if sel is None:
                QMessageBox.information(
                    self, "Zona no seleccionada",
                    "Elegí la herramienta Zona y dibujá un rectángulo sobre la imagen primero.",
                )
                return
            full_h, full_w = self.full_res_image.shape[:2]
            x, y = int(sel[0]), int(sel[1])
            bbox = (x, y, max(1, min(int(sel[2]), full_w - x)), max(1, min(int(sel[3]), full_h - y)))

        if self.dirty and np.any(self.mask):
            reply = QMessageBox.question(
                self, "Sobrescribir",
                "Predecir reemplaza la máscara " + ("de la zona" if bbox else "completa")
                + " (se puede deshacer). ¿Continuar?",
            )
            if reply != QMessageBox.Yes:
                return

        self.pending_bbox = bbox
        self.predict_button.setEnabled(False)
        self.status_label.setText("Procesando... (puede tardar varios segundos en CPU)")

        self.worker = PredictWorker(
            self.model, self.full_res_image, self.combo_magnification.currentText(), bbox
        )
        self.worker.finished_ok.connect(self._on_predict_finished)
        self.worker.failed.connect(self._on_predict_failed)
        self.worker.start()

    def _on_predict_finished(self, result: PredictionResult):
        self.predict_button.setEnabled(True)
        had_content = bool(np.any(self.mask))
        self.undo_stack.push(self.mask)

        if self.pending_bbox is None:
            self._set_mask(result.pred_mask.astype(np.uint8))
        else:
            x, y, w, h = self.pending_bbox
            self.mask[y : y + h, x : x + w] = result.pred_mask
        if not had_content:
            self.origin = "modelo"
        else:
            self.origin = "modelo_corregido"  # mezcla de tags previos y modelo
        self.dirty = True
        self.canvas.update()

        counts = result.class_pixel_counts
        self.status_label.setText(
            f"Listo. Corregí lo que haga falta y aceptá. Píxeles por clase — fondo: {counts[0]}, "
            f"células sueltas: {counts[1]}, esferoides: {counts[2]}, atípicos: {counts[3]}"
        )

    def _on_predict_failed(self, message):
        self.predict_button.setEnabled(True)
        self.status_label.setText("Error al predecir.")
        QMessageBox.critical(self, "Error al predecir", message)
