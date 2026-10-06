"""Matemática pura de mapeo de coordenadas entre el preview achicado (lo que
se muestra en pantalla) y la imagen a resolución nativa (lo que se le pasa al
modelo). No depende de Qt ni de torch — se puede testear con pytest normal.
"""


def compute_preview_scale(orig_w, orig_h, max_preview_dim=1000):
    """Factor de achicado para que la imagen entre en `max_preview_dim` px
    en su lado más largo. Nunca agranda (si ya es chica, scale=1.0).

    Devuelve (scale, preview_w, preview_h).
    """
    scale = min(1.0, max_preview_dim / max(orig_w, orig_h))
    preview_w = max(1, round(orig_w * scale))
    preview_h = max(1, round(orig_h * scale))
    return scale, preview_w, preview_h


def map_bbox_to_full_res(bbox_preview, scale, full_w, full_h):
    """Convierte un rectángulo (x, y, w, h) dibujado sobre el preview a
    coordenadas de la imagen a resolución nativa, recortado a sus bordes.
    """
    x, y, w, h = bbox_preview

    fx = round(x / scale)
    fy = round(y / scale)
    fw = round(w / scale)
    fh = round(h / scale)

    fx = max(0, min(fx, full_w - 1))
    fy = max(0, min(fy, full_h - 1))
    fw = max(1, min(fw, full_w - fx))
    fh = max(1, min(fh, full_h - fy))

    return fx, fy, fw, fh
