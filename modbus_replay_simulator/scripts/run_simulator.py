"""Entrypoint: launches the Modbus Replay Simulator GUI.

Usable two ways:

* As a module (preferred, no sys.path hack):
      python -m modbus_replay
* As a stand-alone script (when you don't want to ``cd`` into the package):
      python scripts/run_simulator.py
  → this path adds the package root to ``sys.path`` so the absolute
    import of ``modbus_replay.gui.main_window`` resolves regardless of CWD.
"""
import sys
from pathlib import Path

# When invoked as ``python scripts/run_simulator.py`` the package root
# (the directory that contains ``modbus_replay/``) is NOT on sys.path.
# Add it so the absolute import below resolves. Harmless when run as
# ``python -m modbus_replay`` because the package root is already on path.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from modbus_replay.gui.main_window import main

if __name__ == "__main__":
    sys.exit(main())