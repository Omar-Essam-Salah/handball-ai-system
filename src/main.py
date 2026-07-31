"""Handball Tagger — Phase 1 MVP entry point."""
import sys

from PyQt6.QtWidgets import QApplication

from player import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Handball Tagger")
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
