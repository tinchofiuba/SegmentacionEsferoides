# Estado del proyecto

Snapshot vivo del estado del pipeline. Las convenciones de trabajo están en
`CLAUDE.md`; los reportes detallados de cada experimento, en
`outputs/reports/<tema>-<fecha>.md`.

Última actualización: 2026-10-02 (reestructuración del repo: paquete uv,
convenciones documentadas, tests reales, configs portables — sin tocar
lógica de modelo/entrenamiento).

## Módulos

| Módulo | Estado | Notas |
|---|---|---|
| Pipeline de datos | OK | `SpheroidDataset` con crops de 512px centrados en objetos (80% sobre objeto / 20% random); normalización ImageNet |
| Modelo | OK | U-Net (segmentation-models-pytorch) con backbone ResNet34 (baseline); experimento con EfficientNet-B0 iniciado en rama `eff_backbone` (ver abajo) |
| Loss | OK | Focal (γ=2.0) + Dice, combinadas 50/50 |
| Entrenamiento | OK | AdamW + ReduceLROnPlateau (monitorea val_iou), checkpoint por mejor val_loss |
| Post-procesado empírico | OK, bajo revisión | Filtra blobs por percentil 5/95 de diámetro (por clase, por aumento 4x/10x) y muta C1→C2 si es demasiado grande; ver pendiente #4 abajo |
| Inferencia | OK | TTA opcional (4 vistas, flips), stitching NO implementado (una sola pasada sobre la imagen completa, con padding a múltiplo de 32) |
| Collages de comparación | OK | Original / Ground Truth / Predicción, 3 paneles |
| Tests | Reescritos | `tests/` ahora tiene tests reales con fixtures sintéticas (antes: 0 asserts reales, scripts de debug con paths hardcodeados — movidos a `scripts/legacy_debug/`) |
| CI | Pendiente | `.github/workflows/ci.yml` con lint + pytest |
| Empaquetado | OK | `src/u_resnet_segesferoides/` instalable vía uv, entry points `u-resnet-train` / `u-resnet-infer` |

## Baseline conocido

Run `insitu_3t3_2026-09-28` (ResNet34, `configs/insitu_3t3.yaml`, seed=42,
split 60/30/10 sobre 96 imágenes): **Val Loss 0.1555, Val IoU 0.8482**.

Normalización ImageNet aplicada (antes el preprocesamiento solo escalaba a
[0,1] sin restar media/dividir std de ImageNet, pese a usar
`encoder_weights="imagenet"`) — ver `outputs/reports/2026-09-23-mejoras-resnet34.md`.

## Caveat de escala de las imágenes

Las imágenes de `insitu_3t3` son de altísima resolución (~6000×8000px). El
modelo se entrena con crops de 512px a resolución nativa, sin ningún
augmentation de escala/zoom. Esto significa que **no es seguro reescalar una
imagen de entrada a una resolución mucho menor antes de inferir**: los
objetos aparecerían con un tamaño en píxeles distinto al que vio el modelo
en entrenamiento, y los umbrales empíricos de post-procesado (en píxeles,
calibrados por 4x/10x) dejarían de ser válidos. Si hace falta inferir rápido
sobre imágenes grandes, la alternativa correcta es tiling con solapamiento y
stitching de logits (no implementado todavía), no un resize global.

## Experimento en curso: EfficientNet-B0

En la rama `eff_backbone` se probó reemplazar el backbone ResNet34
(~24.4M parámetros) por EfficientNet-B0 (~6.3M), con el mismo split de test
(mismo seed) para comparar cabeza a cabeza, más `ElasticTransform` activado
en las augmentations. Motivación: dataset chico (58 imágenes de train),
menor capacidad relativa reduce riesgo de overfitting y entrena más rápido
en CPU. Al momento de esta reestructuración esos cambios quedaron sin
mergear a esta rama (`RES-NET`); si se retoman, hay que volver a aplicar
`encoder_name` configurable en `train.py`/`inference.py` (hoy sigue
hardcodeado a `"resnet34"`) y el `configs/insitu_3t3_effnet.yaml` correspondiente.

## Pendientes (de `outputs/reports/2026-09-23-mejoras-resnet34.md`)

1. ~~Normalización ImageNet~~ — hecho.
2. Activar `ElasticTransform`/`GaussianBlur` en `get_augmentation()` (hoy
   comentadas en `dataset.py`).
3. Guardar el mejor checkpoint por `val_iou` en vez de `val_loss`.
4. Auditar si `postprocess_mask` introduce o tapa errores (comparar collages
   con/sin post-proceso).
5. Clasificar errores recurrentes usando los collages ya generados antes de
   seguir tuneando a ciegas (objetos fundidos entre sí, objetos chicos
   perdidos, bordes imprecisos).
6. Backbone más grande / arquitectura distinta, `ReduceLROnPlateau` con más
   paciencia, más épocas — si el resto no alcanza.

## Deuda de ingeniería conocida (no bloqueante)

- `configs/default.yaml` apunta a un dataset que ya no existe en disco
  (legacy, documentado en el propio archivo).
- `encoder_name` sigue hardcodeado a `"resnet34"` en `train.py` e
  `inference.py` (el experimento EfficientNet-B0 que lo parametrizaba no
  llegó a mergearse acá).
- `scripts/legacy_debug/` tiene 6 scripts que no corren tal cual (importan
  un módulo `train_unet` que ya no existe, o hardcodean datasets borrados).
  Se conservan como referencia pero no están cubiertos por tests ni CI.
- Excepciones de ruff documentadas en `pyproject.toml`
  (`[tool.ruff.lint] ignore`) sobre código heredado: líneas largas,
  `if x: continue` en una línea, un `except:` desnudo y un `zip()` sin
  `strict=` — no se tocaron para no mezclar limpieza de estilo con el
  refactor de estructura.
