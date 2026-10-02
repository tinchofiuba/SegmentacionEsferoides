# Mejoras propuestas para el entrenamiento (U-Net ResNet34)

Contexto: train IoU 83% / val IoU 88%, quedan errores puntuales en las predicciones.
Fecha: 2026-09-23

## [HECHO] 1. Normalización ImageNet en vez de /255.0

El encoder resnet34 se carga con `encoder_weights="imagenet"`, pero las imágenes
solo se escalaban a [0,1] sin restar media/dividir std de ImageNet. Los pesos
preentrenados esperan esa normalización específica, así que el backbone no
estaba aprovechando bien el transfer learning.

Aplicado en `dataset.py` (`preprocess_input` de smp), y propagado a
`train.py` (`run_inference`) e `inference.py` para que train e inferencia
usen exactamente el mismo preprocesamiento.

**IMPORTANTE:** hay que reentrenar desde cero: un checkpoint viejo fue entrenado
con la normalización anterior y ya no es compatible con este preprocesamiento.

## Pendientes (a evaluar según impacto de los errores que se vean en los collages)

### 2. Activar augmentations comentadas en `get_augmentation()` (dataset.py)
- `ElasticTransform` y `GaussianBlur` están comentados. Para esferoides (formas
  orgánicas, bordes fuera de foco) suelen aportar más que flips/rotaciones.
- Probar con probabilidad baja (0.2-0.3) primero para no distorsionar de más.

### 3. Guardar el mejor checkpoint por val_iou en vez de val_loss
- Actualmente `train.py` guarda el modelo cuando `avg_val_loss` es mínimo.
- Con loss combinada (Focal+Dice) el mínimo de loss no siempre coincide con
  el máximo de IoU. Cambiar el criterio a `avg_val_iou > best_val_iou`.

### 4. Revisar si el post-procesamiento empírico (`postprocess_mask` en utils.py) está introduciendo o tapando errores
- Descarta blobs por debajo del percentil 5 de diámetro y muta C1->C2 si es
  muy grande, en base a umbrales calculados sobre el propio dataset (chico
  => percentiles poco robustos).
- Comparar collages con y sin este post-proceso para saber si los errores
  visibles vienen del modelo o de estas reglas duras.

### 5. Clasificar los errores recurrentes usando los collages ya generados (`outputs/comparisons/<model_name>/`) antes de seguir tuneando a ciegas
- Objetos que se funden entre sí (esferoides tocándose) -> problema de
  instancia, no de semántica; considerar watershed post-hoc o loss con
  peso extra en los bordes.
- Objetos chicos perdidos -> subir `patches_per_image`, aumentar `crop_size`,
  o agregar class weights en la loss.
- Bordes imprecisos -> ElasticTransform (punto 2) + mayor resolución de crop.

### 6. Otros ajustes a probar según capacidad de cómputo
- Encoder más grande (resnet50) o arquitectura distinta (Unet++, DeepLabV3+)
  si el cuello de botella parece ser capacidad del modelo.
- `ReduceLROnPlateau` monitoreando val_iou en vez de (o además de) Cosine.
- Más épocas con early stopping por paciencia, si el Cosine no llegó a
  converger en las 40 épocas actuales.

**Prioridad sugerida:** 1 (hecho) -> 5 (diagnóstico con collages, gratis) -> 2 y 3
(baratos de probar) -> 4 (auditoría del post-proceso) -> 6 (más costoso).
