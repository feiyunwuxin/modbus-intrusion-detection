"""Allow ``python -m modbus_replay`` to launch the GUI."""
from modbus_replay.gui.main_window import main

if __name__ == "__main__":
    raise SystemExit(main())
