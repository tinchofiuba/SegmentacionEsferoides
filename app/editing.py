"""Operaciones puras de edición de máscaras (sin Qt ni torch), para usar la
app como herramienta de tagging: pintar, borrar/reasignar un objeto entero,
undo y guardado en el formato de color de ImgTaggeadas.
"""

import cv2
import numpy as np

from u_resnet_segesferoides.dataset import MASK_COLOR_TO_CLASS

# Colores RGB para mostrar en pantalla (el mapeo de archivo es BGR, ver save_mask).
DISPLAY_COLORS_RGB = {1: (0, 255, 0), 2: (255, 255, 0), 3: (0, 255, 255)}
CLASS_NAMES = {1: "células sueltas", 2: "esferoides", 3: "atípicos"}


def paint_circle(mask, x, y, radius, class_id):
    """Pinta un círculo de `class_id` (0 = borrar) en `mask`, in-place."""
    cv2.circle(mask, (int(x), int(y)), max(1, int(radius)), int(class_id), thickness=-1)


def paint_segment(mask, x0, y0, x1, y1, radius, class_id):
    """Trazo continuo (línea con extremos redondeados) de `class_id` (0 =
    borrar) entre dos puntos. Evita huecos cuando el mouse se mueve rápido,
    y con radio chico sirve para cortar entre objetos pegados."""
    p0 = (int(round(x0)), int(round(y0)))
    p1 = (int(round(x1)), int(round(y1)))
    r = max(1, int(radius))
    cv2.line(mask, p0, p1, int(class_id), thickness=2 * r)
    cv2.circle(mask, p0, r, int(class_id), thickness=-1)
    cv2.circle(mask, p1, r, int(class_id), thickness=-1)


def replace_object(mask, x, y, new_class):
    """Reasigna a `new_class` (0 = borrar) el objeto conexo bajo (x, y), es
    decir, la región 4-conexa con la misma clase. Devuelve True si cambió algo.
    Si (x, y) cae en fondo y new_class != 0, no hace nada (no hay objeto)."""
    h, w = mask.shape
    x, y = int(x), int(y)
    if not (0 <= x < w and 0 <= y < h):
        return False
    current = int(mask[y, x])
    if current == 0 or current == new_class:
        return False
    cv2.floodFill(mask, None, (x, y), int(new_class), loDiff=0, upDiff=0, flags=4)
    return True


def fill_holes(mask, class_id):
    """Rellena huecos internos de la clase `class_id`, in-place."""
    binary = (mask == class_id).astype(np.uint8)
    inv = 1 - binary
    h, w = inv.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    reachable = inv.copy()
    # Fondo conectado al borde: se marca con 2 desde cada píxel de borde que sea fondo
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if reachable[seed[1], seed[0]] == 1:
            cv2.floodFill(reachable, flood, seed, 2)
    holes = reachable == 1
    mask[holes & (mask == 0)] = class_id


def render_overlay(image_rgb, mask, alpha=0.5):
    """Imagen con la máscara superpuesta (fondo transparente)."""
    overlay = image_rgb.copy()
    color = np.zeros_like(image_rgb)
    for cid, rgb in DISPLAY_COLORS_RGB.items():
        color[mask == cid] = rgb
    blended = cv2.addWeighted(image_rgb, 1 - alpha, color, alpha, 0)
    has_class = mask > 0
    overlay[has_class] = blended[has_class]
    return overlay


def save_mask(path, mask):
    """Guarda `mask` (clases 0-3) con los colores que lee load_mask_labels."""
    color_bgr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for bgr, class_id in MASK_COLOR_TO_CLASS.items():
        color_bgr[mask == class_id] = bgr
    if not cv2.imwrite(path, color_bgr):
        raise OSError(f"No se pudo escribir la máscara en {path}")


class UndoStack:
    """Pila acotada de snapshots de máscara (cada una pesa HxW bytes)."""

    def __init__(self, max_size=8):
        self.max_size = max_size
        self._undo = []
        self._redo = []

    def push(self, mask):
        self._undo.append(mask.copy())
        if len(self._undo) > self.max_size:
            self._undo.pop(0)
        self._redo.clear()

    def discard_last(self):
        if self._undo:
            self._undo.pop()

    def undo(self, current):
        if not self._undo:
            return None
        self._redo.append(current.copy())
        return self._undo.pop()

    def redo(self, current):
        if not self._redo:
            return None
        self._undo.append(current.copy())
        return self._redo.pop()

    def clear(self):
        self._undo.clear()
        self._redo.clear()
