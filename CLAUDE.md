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
- ONNX: al terminar `u-resnet-train` se guarda también `best_model.onnx` en la carpeta del run
  (exportado desde `best_model.pth`, no de la última época; si falla solo avisa). Para runs viejos:
  `uv run u-resnet-export-onnx --run-dir outputs/models/<run>`. Entrada: imagen ya preprocesada
  (ImageNet, múltiplo de 32), NCHW, H/W dinámicos; salida: logits. Sin argmax/TTA/post-proceso.
- Evaluación (análoga a `spheroid_seg.eval`): `uv run u-resnet-eval --config configs/insitu_3t3.yaml --split test [--run-dir outputs/models/<run>]`.
  Imagen completa (+TTA del run), confusión pooled, Dice/IoU/precisión/recall por clase,
  por magnificación, vista de 3 clases (2+3 fusionadas), métrica `object`, Brier/log loss
  sobre softmax, y las mismas métricas post-procesado. Escribe `outputs/evals/<run>_<split>/`.
  Código en `src/u_resnet_segesferoides/evaluate.py`; tests en `tests/test_evaluate.py`.
- Evaluación YOLOv8-seg (mismas métricas, mismo split, para comparar cabeza a cabeza):
  `uv run --with ultralytics u-resnet-eval-yolo --run-dir <dir del modelo YOLO>`. `ultralytics`
  no es dependencia del proyecto. Sin Brier/log loss ni variante post-procesada (YOLO no da
  probabilidades por píxel). Código en `src/u_resnet_segesferoides/evaluate_yolo.py`.

## Datos

Las imágenes y máscaras NO viven en el repo (son pesadas y privadas). Cada
dataset tiene su propia carpeta bajo `data/<dataset>/`, con `raw/` y `masks/`
como **symlinks** a la ubicación real — todo `data/` está gitignoreado.

Setup del dataset vigente (`insitu_3t3`; las rutas de origen dependen de cada máquina, ej. `/ruta/a`):

```bash
mkdir -p data/insitu_3t3
ln -s /ruta/a/ImgCrudas data/insitu_3t3/raw
ln -s /ruta/a/ImgTaggeadas data/insitu_3t3/masks
```

`configs/default.yaml` es un config legacy de la primera ronda de
experimentos cuyo dataset ya no existe en disco; se mantiene como referencia
histórica, no como algo a correr.

## Convenciones

- Los hiperparámetros (crop_size, batch_size, lr, num_workers, splits, etc.) viven
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
- `postprocess/recompute_thresholds.py`: recalcula `inference.empirical_thresholds`
  en el `config.yaml` de runs ya entrenados (el post-proceso no depende de los
  pesos, no hace falta reentrenar). Resguarda el original como
  `config.pre_thresholds.yaml`.
- `viz/generate_comparison_images.py`, `viz/regenerate_comparisons.py`:
  collages de comparación sobre un modelo ya entrenado, sin reentrenar.

## Integración con la GUI (ONNX + manifiesto)

La GUI (`GUI_SegmentacionEsferoides`, `tinchofiuba/GUI_SegmentacionEsferoides`) **no importa nada de
este paquete ni usa `serve`**: solo corre modelos `.onnx` con `onnxruntime`, y cada `.onnx` lleva al
lado un manifiesto `.yaml` (especificación en `docs/model_manifest.md` del repo de la GUI).
`export_onnx.py` escribe ese manifiesto (`best_model.yaml`) junto a `best_model.onnx` al exportar:
clases/colores (`MANIFEST_CLASSES`, mismo mapeo que `MASK_COLOR_TO_CLASS` pero en RGB), normalización
ImageNet, `pad_multiple: 32` y las 3 vistas de TTA. Si cambia el preprocesado, el TTA o las clases,
hay que actualizar `build_manifest` (y `tests/test_export_onnx.py`). La GUI se distribuye con los
modelos aparte: se pasan `best_model.onnx` + `best_model.yaml` y se instalan desde la GUI
("Instalar modelo...").

`src/u_resnet_segesferoides/serve.py` (protocolo stdin/stdout, `u-resnet-serve`) es el mecanismo
anterior de la GUI; ya no lo usa nadie. Se mantiene por ahora (tests en `tests/test_serve.py`).

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
