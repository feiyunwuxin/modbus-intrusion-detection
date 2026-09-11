"""Entrypoint: launches the Modbus Replay Simulator GUI."""
import sys
from modbus_replay.gui.main_window import main

if __name__ == "__main__":
    sys.exit(main())