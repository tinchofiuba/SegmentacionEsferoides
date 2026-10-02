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

`scripts/` está organizado por propósito:
- `data_prep/`: conversión y preparación de datasets.
- `analysis/`: diagnóstico puntual sobre máscaras/predicciones.
- `viz/`: collages y overlays de comparación.
- `eval/`: evaluación de checkpoints ya entrenados.
- `legacy_debug/`: scripts y "tests" viejos pre-refactor que ya no corren
  (dependen de un módulo `train_unet` que no existe, o de datasets que ya no
  están en disco). Se conservan como referencia, no se ejecutan ni se
  mantienen.

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
