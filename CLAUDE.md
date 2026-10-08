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
  **El split vigente es idéntico al de `spheroid-seg`** (68/15/13, mismas
  imágenes en cada conjunto, para poder comparar modelos entre repos). Se
  importó con `scripts/data_prep/import_splits.py` (quita el sufijo
  `_4x`/`_10x` y chequea leaks); no editar a mano. El split anterior
  (seed=42, 58/28/10) quedó en `splits/insitu_3t3_seed42/` — los reportes y
  métricas previos a esta fecha (2026-10-06) se midieron con ese, no son
  comparables con corridas nuevas.
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
- `data_prep/import_splits.py`: importa un split de otro repo (spheroid-seg).
- `viz/generate_comparison_images.py`, `viz/regenerate_comparisons.py`:
  collages de comparación sobre un modelo ya entrenado, sin reentrenar.

## Backend para GUIs externas (`serve`)

La GUI de predicción/tagging **no vive en este repo**: está en
`/home/martin/repos/GUI_SegmentacionEsferoides` y no importa nada de este
paquete. Este repo tampoco sabe nada de la GUI: solo expone
`src/u_resnet_segesferoides/serve.py`, un servidor genérico por stdin/stdout
(una petición JSON por línea → una respuesta) que cualquier cliente puede
usar. Comandos: `info`, `list_models`, `load`, `predict`, `shutdown`; la
imagen y la máscara se pasan por ruta de archivo. El protocolo (v1) está
especificado en `docs/protocol.md` del repo de la GUI.

```bash
uv run python -m u_resnet_segesferoides.serve --models-dir outputs/models \
    [--model-name insitu_3t3 | --run-dir outputs/models/<run>]
```

- Reusa `pad_image`, `preprocess_input`, `tta_inference`, `postprocess_mask`:
  cero lógica de modelo duplicada. **Si hay `bbox`, recorta ANTES de inferir**
  (preserva la escala en píxeles; nunca se reescala la imagen).
- Lee `training.encoder_name` del `config.yaml` de cada run (default
  resnet34), así sirve también runs EfficientNet.
- stdout es solo protocolo (se redirige el resto a stderr): no hacer `print`
  a stdout en el camino de `serve`. Tests en `tests/test_serve.py`.
- Para usarlo desde la GUI: `config/backends.yaml` del repo de la GUI apunta a
  `.venv/bin/python -m u_resnet_segesferoides.serve ...` de este repo.
- `list_models` marca con una nota los runs sin `metrics.txt` (probablemente
  entrenándose: su `best_model.pth` se sobrescribe).

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
