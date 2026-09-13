# Modbus Replay Simulator

Desktop tool that reads `IanArffDataset.csv` and replays its rows as
32-byte binary frames over a serial port, preserving the dataset's
`time`-column intervals. Designed for bench-testing an STM32H743
(or STM32F407) running the TCN+SE intrusion-detection model.

## Quick start

```bash
git clone <this repo>
cd modbus_replay_simulator
pip install -e ".[dev]"
python -m modbus_replay
```

The GUI launches; pick the COM port your USB-UART bridge exposes,
select `IanArffDataset.csv`, click **▶ 开始**.

## Bench wiring

```
┌──────────────┐  USB-UART  ┌──────────┐   RS-485   ┌────────┐
│  PC (this    │──────────────│ MCU     │ │           │ Slaves │
│   simulator) │   115200 8N1│ H743    │ │  (off-     │        │
└──────────────┘             └──────────┘ │   bench)  └────────┘
                                          └────────────┘
```

If the MCU's USART is configured at 115200 8N1 (the default in
`TestH743/H743test/Core/Src/usart.c`), match the GUI dropdown.
For real Modbus RTU deployments change MCU to 9600 and update the
GUI accordingly.

## Frame format

32 bytes per packet, little-endian, no CRC. See
`modbus_replay/frame_format.py:FRAME_LAYOUT` for the full layout.
A reference Python decoder is in `frame_format.decode()`.

## CSV columns

The replay engine consumes the same 17-feature schema the TCN+SE
model expects (see `IanArffDataset.csv`):

| idx | column | type | notes |
|----:|--------|------|-------|
| 0 | time | float | seconds, monotonic within the file |
| 1 | address | int | Modbus slave address (1-247) |
| 2 | function | int | Modbus function code (1/2/3/4/5/6/15/16) |
| 3 | length | int | payload length, bytes |
| 4 | setpoint | int16 | scaled value (×100) |
| 5 | gain | int16 | scaled value (×100) |
| 6 | reset | int16 | scaled value (×100) |
| 7 | deadband | int16 | scaled value (×100) |
| 8 | cycle | int16 | cycle index |
| 9 | rate | float32 | setpoint change rate |
| 10 | system_mode | int16 | operating mode |
| 11 | control_scheme | int16 | control scheme |
| 12 | pump | int16 | pump state |
| 13 | solenoid | int16 | solenoid state |
| 14 | pressure | float32 | measured pressure (PSI) |
| 15 | response | int16 | response flag |
| 16 | crc_rate | int16 | scaled CRC rate (×100) |

Missing values (`?` in the ARFF or empty CSV cells) are encoded as
`int16 = 0` for integer fields and `float32 = 0x7FC00000` (quiet NaN)
for floating-point fields. See `frame_format._is_missing` for the
detection rules.

## Controls

| Control | Function |
|---------|----------|
| **COM** dropdown | Select the serial port; **🔄 刷新** rescans |
| **Baud / Data / Parity / Stop** | Five serial parameters (default 115200 8N1) |
| **📂 选择 CSV…** | Pick the source CSV (defaults to `IanArffDataset.csv`) |
| **▶ 开始** | Build a SerialWorker thread and start pacing |
| **⏸ 暂停 / ▶ 继续** | Toggle worker pause/resume (label flips) |
| **⏹ 停止** | Set `_stopping`; the loop exits at the next checkpoint |
| **循环播放** | Replay the CSV from the top after the last row |

The progress bar shows row position; the elapsed/ETA label ticks every
500 ms once the first frame has been sent.

## Tests

```bash
pytest -v
```

40 unit tests cover the seven modules (csv_loader, frame_format,
replay_engine, serial_worker, gui.widgets, gui.main_window,
integration_loopback). End-to-end hardware validation is the manual
checklist in `docs/MANUAL_VALIDATION.md` (Task 10).

## License

Research use, see repo root LICENSE.
