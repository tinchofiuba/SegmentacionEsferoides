# SegmentacionEsferoides

Segmentación de esferoides y células sueltas en microscopía de contraste de fase (4x/10x) con
U-Net (ResNet34 / EfficientNet-B0) en PyTorch.

**Para solo usar un modelo no hace falta este repo**: la GUI
([GUI_SegmentacionEsferoides](https://github.com/tinchofiuba/GUI_SegmentacionEsferoides)) corre
modelos `.onnx` con su manifiesto `.yaml`, sin torch. Este repo es para entrenar, evaluar y exportar.

## Instalación (Windows 10/11, Linux, macOS)

1. Instalar [uv](https://docs.astral.sh/uv/getting-started/installation/)
   (Windows, PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`).
2. `uv sync --group dev`  (en Windows instala torch para CPU; para GPU NVIDIA ver
   [la guía de uv para PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)).
3. `uv run pytest`

## Datos

Las imágenes y máscaras no están en el repo. El config espera `data/insitu_3t3/raw` y
`data/insitu_3t3/masks`:

- Linux/macOS: `ln -s /ruta/ImgCrudas data/insitu_3t3/raw` (idem `masks`).
- Windows (cmd, sin permisos de administrador): `mklink /J data\insitu_3t3\raw D:\ImgCrudas` y
  `mklink /J data\insitu_3t3\masks D:\ImgTaggeadas`. O bien copiar las carpetas, o cambiar
  `paths.images_dir` / `paths.masks_dir` en el config.

En Windows conviene poner `training.num_workers: 0` en el config si el DataLoader da problemas.

## Comandos

```
uv run u-resnet-train --config configs/insitu_3t3.yaml    # entrena; al final exporta best_model.onnx + .yaml
uv run u-resnet-export-onnx --run-dir outputs/models/<run> # exporta un run ya entrenado
uv run u-resnet-eval  --config configs/insitu_3t3.yaml --split test
uv run u-resnet-infer --config configs/insitu_3t3.yaml
```

## Pasar un modelo a otras personas

Cada run exportado tiene en su carpeta `best_model.onnx` y `best_model.yaml` (manifiesto: clases,
colores, normalización, TTA). Pasar esos dos archivos; quien tenga la GUI usa
**Instalar modelo...** y elige el `.onnx`.

## Publicar un modelo nuevo (actualización desde la GUI)

```
uv run u-resnet-export-onnx --run-dir outputs/models/<run>     # si el run no lo exportó solo
uv run u-resnet-make-index --run-dir outputs/models/<run> --out release/ --notes "qué mejora"
```

Subir `release/*` (`models.json`, `<run>.onnx`, `<run>.yaml`) a un release de GitHub. La GUI, con
`models_index_url: https://github.com/<usuario>/SegmentacionEsferoides/releases/latest/download/models.json`
en su `config/backends.yaml`, muestra **Buscar modelos nuevos**: compara el sha256 y baja lo que falte.
El repo tiene que ser público (o el índice estar en otro host): la GUI no manda credenciales.
