"""Punto de entrada de la app de escritorio. Se corre con:

    uv run python app/main.py [--output-dir outputs] [--model-name insitu_3t3]
"""

import argparse
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox

from app.main_window import MainWindow
from app.model import SegmentationModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", default="insitu_3t3")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    try:
        model = SegmentationModel.from_latest(args.output_dir, args.model_name)
    except FileNotFoundError as exc:
        QMessageBox.critical(None, "No se encontró un modelo entrenado", str(exc))
        sys.exit(1)

    window = MainWindow(model)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
