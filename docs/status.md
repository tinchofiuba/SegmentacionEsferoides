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
| Pipeline de datos | OK | `SpheroidDataset` con crops de 512px centrados en objetos (80% sobre objeto / 20% random); normalización ImageNet; split train/val/test materializado en `splits/insitu_3t3/*.txt` (antes: recalculado con seed+shuffle en cada corrida) |
| Modelo | OK | U-Net (segmentation-models-pytorch) con backbone ResNet34 (baseline); experimento con EfficientNet-B0 iniciado en rama `eff_backbone` (ver abajo) |
| Loss | OK | Focal (γ=2.0) + Dice, combinadas 50/50 |
| Entrenamiento | OK | AdamW + ReduceLROnPlateau (monitorea val_iou), checkpoint por mejor val_loss |
| Post-procesado empírico | OK, bajo revisión | Filtra blobs por percentil 5/95 de diámetro (por clase, por aumento 4x/10x) y muta C1→C2 si es demasiado grande; ver pendiente #4 abajo |
| Inferencia | OK | TTA opcional (4 vistas, flips), stitching NO implementado (una sola pasada sobre la imagen completa, con padding a múltiplo de 32) |
| Collages de comparación | OK | Original / Ground Truth / Predicción, 3 paneles |
| Tests | Reescritos | `tests/` ahora tiene tests reales con fixtures sintéticas (antes: 0 asserts reales) |
| Scripts | Podados | Se borraron 19 scripts de debug/diagnóstico que apuntaban a datasets/runs ya inexistentes (ver "Limpieza de scripts" abajo); quedan solo `data_prep/prepare_insitu_dataset.py` y `viz/{generate_comparison_images,regenerate_comparisons}.py` |
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

## Pendiente de decisión: formato de máscara vs. spheroid-seg

Las máscaras de este repo son PNG a color (verde/amarillo/cyan sobre negro,
decodificadas por `MASK_COLOR_TO_CLASS` en `dataset.py`). `spheroid-seg` usa
máscaras de 1 canal en escala de grises con el valor de píxel = id de clase
(0-3), formato más estándar y liviano. El mapeo de clases coincide 1 a 1
entre ambos repos, así que es una diferencia solo de codificación, no
semántica. Diferido a propósito: decidir si (a) se convierten las 96
máscaras físicas en `/media/martin/D/ImgTaggeadas` a grayscale, o (b)
`load_mask_labels` pasa a soportar ambos formatos sin tocar los archivos
originales.

## Deuda de ingeniería conocida (no bloqueante)

- `configs/default.yaml` apunta a un dataset que ya no existe en disco
  (legacy, documentado en el propio archivo).
- `encoder_name` sigue hardcodeado a `"resnet34"` en `train.py` e
  `inference.py` (el experimento EfficientNet-B0 que lo parametrizaba no
  llegó a mergearse acá).
- Excepciones de ruff documentadas en `pyproject.toml`
  (`[tool.ruff.lint] ignore`) sobre código heredado en `dataset.py`: líneas
  largas y `if x: continue` en una línea — no se tocaron para no mezclar
  limpieza de estilo con el refactor de estructura.

## Limpieza de scripts (2026-10-02)

Se revisaron a fondo los 20 scripts que había en `scripts/` (más los 6 que
vivían mal ubicados en `tests/`/raíz) y se borraron 19 por estar
confirmadamente muertos — ninguno quedó con lógica rescatable que no esté ya
en el paquete:

- Apuntaban a `inputsConvertidas/0_1_2_3`/`ConColores` o a
  `/home/martin/Descargas/drive-download-*`, directorios que ya no existen
  en disco (`analyze_areas.py`, `check_ground_truth.py`, `find_3_classes.py`,
  `convert_masks.py`, `visualize_mask.py`, `convert_jpg_to_tiff.py`,
  `extract_3t3_masks.py`, `diagnose.py`).
- Apuntaban a predicciones/runs de la convención vieja
  (`pred_3classes_*`, `outputs/predictions/default/`, run
  `default_2026-08-20`), ya inexistentes (`analyze_prediction.py`,
  `check_diameters.py`, `investigate_holes.py`, `test_inference_single.py` —
  este último además duplicaba `u-resnet-infer`).
- `make_collage.py`: sus 3 inputs estaban muertos y el output apuntaba a una
  carpeta de otra herramienta de IA (`~/.gemini/antigravity/...`), ajena al
  repo. La función que cumplía (`generate_comparison_collage`) ya corre
  automáticamente en cada training.
- `eval_huge_images.py`: archivo vacío (0 bytes).
- Los 6 de `tests/` que importaban `train_unet` (módulo de una arquitectura
  pre-refactor que ya no existe) o duplicaban `pad_image`/`colorize_mask` en
  vez de importarlos de `utils.py`.

Quedan solo `scripts/data_prep/prepare_insitu_dataset.py` y
`scripts/viz/{generate_comparison_images,regenerate_comparisons}.py`, los
tres parametrizados por `--config` y vigentes contra el dataset activo.
