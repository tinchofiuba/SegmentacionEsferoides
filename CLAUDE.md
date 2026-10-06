# CLAUDE.md

Pipeline de segmentación de esferoides y células sueltas en imágenes de
microscopía de contraste de fase (4x/10x), con U-Net (ResNet34 o
EfficientNet-B0 como backbone) en PyTorch. Ver `docs/status.md` para el
estado actual del proyecto (baseline, experimentos en curso, pendientes).

**No tocar la lógica de modelo/entrenamiento/post-procesado sin que lo pida
explícitamente la tarea** (arquitectura, losses, augmentations, umbrales
empíricos de post-procesado, TTA). Son decisiones ya validadas con
resultados medidos; cualquier cambio ahí es un experimento, no un refactor.

## Stack

- Python 3.10, uv-managed (`pyproject.toml`, `src/` layout).
- Core: `torch`, `torchvision`, `segmentation-models-pytorch`. Datos/visión:
  `albumentations`, `opencv-python-headless`, `tifffile`, `pillow`.
- Tests/lint: `pytest`, `ruff`.
- **CPU-only en esta máquina** (`torch.cuda.is_available() == False`). Todo
  entrenamiento corre en CPU — tenerlo en cuenta al elegir backbone y al
  estimar tiempos (un `calculate_empirical_thresholds` sobre el dataset de
  `insitu_3t3` ya tarda varios minutos solo escaneando máscaras, porque las
  imágenes son de altísima resolución, ~6000x8000px).

## Comandos

- Setup: `uv sync --group dev`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Train: `uv run u-resnet-train --config configs/insitu_3t3.yaml`
- Inferencia: `uv run u-resnet-infer --config configs/insitu_3t3.yaml`

## Datos

Las imágenes y máscaras NO viven en el repo (son pesadas y privadas). Cada
dataset tiene su propia carpeta bajo `data/<dataset>/`, con `raw/` y `masks/`
como **symlinks** a la ubicación real — todo `data/` está gitignoreado.

Setup del dataset vigente (`insitu_3t3`, en `/media/martin/D`):

```bash
mkdir -p data/insitu_3t3
ln -s /media/martin/D/ImgCrudas data/insitu_3t3/raw
ln -s /media/martin/D/ImgTaggeadas data/insitu_3t3/masks
```

`configs/default.yaml` es un config legacy de la primera ronda de
experimentos cuyo dataset ya no existe en disco; se mantiene como referencia
histórica, no como algo a correr.

## Convenciones

- Los hiperparámetros (crop_size, batch_size, lr, splits, etc.) viven
  **únicamente** en `configs/*.yaml`. No hardcodear valores en el código.
- Splits reproducibles: si un config define `paths.splits_dir`, el split
  train/val/test se lee de `<splits_dir>/{train,val,test}.txt` (listas de
  basenames, versionadas en git) en vez de recalcularse con `seed` + shuffle
  en cada corrida — así no cambia si el dataset crece. Ver
  `splits/insitu_3t3/` para el ejemplo vigente. `train_split`/`val_split`/
  `test_split`/`seed` quedan como fallback para configs que no usan
  `splits_dir` (p. ej. `configs/default.yaml`, legacy).
- Mapeo de color de máscaras (fijo, no cambiar sin actualizar todo el
  pipeline): verde=1 (células sueltas), amarillo=2 (esferoides), cyan=3
  (atípicos) — ver `MASK_COLOR_TO_CLASS` en `dataset.py`.
- Cada corrida de entrenamiento escribe a su propia carpeta
  `outputs/models/<config>_<fecha>/` (con reintento `(1)`, `(2)`... si ya
  existe) — nunca pisa una corrida anterior.
- `outputs/` está gitignoreado **excepto `outputs/reports/`**, que sí se
  commitea: ahí van los reportes de experimentos (ver abajo).
- Un solo entrenamiento a la vez en esta máquina (CPU compartida): antes de
  lanzar uno, chequear `pgrep -f u_resnet_segesferoides.train` / `pgrep -f
  "train_model("` y no superponer corridas.

## Scripts

`scripts/` quedó reducido a lo vigente (se borraron 19 scripts de
debug/diagnóstico puntual que apuntaban a datasets/runs que ya no existen en
disco, o duplicaban lo que ya hace `u-resnet-train`/`u-resnet-infer` — ver
`docs/status.md`):
- `data_prep/prepare_insitu_dataset.py`: convierte `ImgCrudas`/`ImgTaggeadas`
  al formato que espera `SpheroidDataset`.
- `viz/generate_comparison_images.py`, `viz/regenerate_comparisons.py`:
  collages de comparación sobre un modelo ya entrenado, sin reentrenar.

## App de escritorio (predicción por zona)

`app/` (fuera de `src/`, no se empaqueta — es una app consumidora del
paquete, no parte del "cerebro") expone una ventana PyQt5 para importar una
imagen, verla achicada en pantalla, y predecir sobre toda la imagen o sobre
una zona marcada con un recuadro arrastrable. Separación Model/View:

- `app/model.py`: `SegmentationModel` — sin ningún import de PyQt5, envuelve
  un run entrenado (`config.yaml` + `best_model.pth`) y expone
  `predict(image_rgb, magnification, bbox=None)`. Reusa `pad_image`,
  `preprocess_input`, `tta_inference`, `postprocess_mask`, `colorize_mask`
  del paquete — cero lógica de modelo duplicada. **Si hay `bbox`, recorta
  ANTES de inferir** (preserva la escala en píxeles; nunca se reescala la
  imagen antes de pasarla al modelo).
- `app/editing.py`: edición de máscaras pura (pincel/borrador en trazos
  continuos, borrar/reasignar objeto por flood fill, rellenar huecos, undo,
  guardado en los colores de `ImgTaggeadas`). Tests en `tests/test_app_editing.py`.
- `app/viewport.py`: matemática y render de zoom/pan (recorta solo la región
  visible a resolución nativa; pirámide para zooms bajos). Tests en
  `tests/test_app_viewport.py`.
- `app/geometry.py`: mapeo preview↔nativa de la primera versión; ya no lo usa
  la ventana (todo se edita en coordenadas nativas), se conserva con su test.
- `app/image_canvas.py`, `app/main_window.py`, `app/main.py`: la View.
  Controles: rueda = zoom, botón derecho/central = mover, `[` `]` = tamaño de
  la herramienta, F = ajustar. "Aceptar" (Ctrl+S) guarda `<img>_Taggeada.png` +
  `.origin.json` (manual / modelo / modelo_corregido).

Instalar y correr (como módulo, con `-m` — correrlo como script suelto,
`python app/main.py`, falla con `ModuleNotFoundError: No module named 'app'`
porque Python agrega `app/` a `sys.path` en vez de la raíz del repo):
```bash
uv sync --extra app
uv run python -m app.main --output-dir outputs --model-name insitu_3t3
```
Carga el run más reciente que matchee `<model-name>_<fecha>` (vía
`find_latest_run_dir` en `utils.py`, reusada también por `u-resnet-infer`).

## Reportes de experimentos

Al terminar un experimento relevante (cambio de backbone, de augmentation,
de post-proceso, etc.), documentarlo en
`outputs/reports/<tema>-<fecha>.md`: qué se cambió, por qué, y cómo se
compara contra el run anterior (mismo seed/split cuando sea posible).

## Testing

`tests/` tiene tests reales (pytest, con fixtures sintéticas, sin depender
de `data/` ni de pesos entrenados) sobre las piezas puras de
`dataset.py`/`utils.py` (decodificación de máscaras, padding, post-proceso
por umbrales). No hay tests de integración del training completo (correr un
epoch real tarda minutos en CPU con estas imágenes).
