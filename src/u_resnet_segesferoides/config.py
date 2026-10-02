import os
import sys

import yaml


def load_config(config_path="configs/default.yaml"):
    """Carga la configuración desde un archivo YAML."""
    if not os.path.exists(config_path):
        print(f"\n\033[91m[ERROR] No se encontró el archivo de configuración: {config_path}\033[0m")
        print("Por favor verifica que la ruta sea correcta (ej: --config configs/default.yaml)\n")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config
