# Modbus Replay Simulator — Design Spec

**Date**: 2026-09-11
**Author**: Claude (brainstorming session)
**Status**: Awaiting user review
**Target MCU**: STM32H743 (also applicable to STM32F407)

---

## 1. Purpose

Build a desktop Modbus bus-replay simulator that reads the
**Mississippi State SCADA Gas Pipeline Dataset**
(`IanArffDataset.csv`) and sends its rows as fixed-length binary
frames over a serial port, preserving the original `time`-column
intervals. The simulator feeds an MCU that runs the TCN+SE
intrusion-detection model so the model can be validated against
realistic traffic on the bench (without the original SCADA testbed).

**Out of scope**: implementation of the MCU-side parser (this spec
covers only the PC simulator); the full Modbus RTU protocol; CRC
validation; replay of attacks beyond what the dataset already
contains.

---

## 2. Background

- **Dataset**: 274,628 rows, 20 columns, contains both master
  commands (`command response == 0`) and slave responses
  (`command response == 1`). Fields are *parsed*, not raw Modbus
  bytes; `crc rate` is a CRC-error count, not the CRC bytes
  themselves.
- **MCU side**: STM32H743 with USART1 configured at **115200 8N1**;
  the deployed TCN+SE model consumes 19 row-level features + 8
  window-level aggregates, window=16 timesteps, 23 dims after
  feature selection.
- **Windowing**: The model expects each 16-row window to contain a
  realistic mix of commands and responses (typical mix
  ≈ 25% commands / 75% responses). Therefore the simulator must
  replay *both* directions — not just commands.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│     ModbusReplaySimulator (PyQt5 Desktop App)                │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌────────────────┐    │
│  │  GUI Layer  │   │ReplayEngine │   │ SerialWorker   │    │
│  │ (MainWindow)│◄──┤ (timing)    │◄──┤ (QThread)      │    │
│  │             │   │             │   │                │    │
│  │ • COM       │   │ • read CSV  │   │ • pyserial     │    │
│  │ • baud rate │   │ • encode    │   │ • 32 B frames  │    │
│  │ • progress  │   │ • compute Δt│   │ • back-pressure│    │
│  └─────────────┘   └─────────────┘   └────────────────┘    │
│           ▲                 ▲                              │
│           └── Qt signals/slots (thread-safe) ──┘           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  StatusBus: progress / state / errors                  │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                          ▼ RS-232 / USB-UART
                   ┌──────────────┐
                   │  MCU (H743)  │
                   │  RX ← 32B/包 │
                   └──────────────┘
```

**Principles**:

1. **GUI / Engine / Serial three-layer split.** Serial I/O is
   blocking; it must run on a `QThread` so the GUI stays
   responsive.
2. **Qt Signals/Slots for cross-thread communication.**
   Progress, state, and errors propagate as signals.
3. **`ReplayEngine` is pure and unit-testable.** It converts
   `(row, base_time)` → `(bytes_frame, delta_ms)` with no side
   effects; the `SerialWorker` owns the wall-clock and the
   serial port.
4. **Single-writer discipline.** Only `SerialWorker` writes to
   the port; no byte interleaving.

---

## 4. GUI Layout

```
┌─ Modbus Replay Simulator ───────────────────────────────┐
│                                                         │
│ 串口设置                                                  │
│  [COM 口 ▼]  [115200 ▼]  [8 ▼]  [N ▼]  [1 ▼]  [🔄 刷新] │
│                                                         │
│ 数据源                                                    │
│  [📂 选择 CSV...] D:/workspace/.../IanArffDataset.csv   │
│                                                         │
│ 回放控制                                                  │
│  [▶ 开始] [⏸ 暂停] [⏹ 停止]  ☐ 循环播放                │
│                                                         │
│ 进度                                                       │
│  [████████████░░░░░░░░░░░░░░░░] 42.3% (115,832 / 274,628)│
│  已用时间: 00:18:42   预计剩余: 00:25:31                  │
│  当前包: row=115832 addr=4 fc=3 resp=1                    │
│                                                         │
│ 日志                                                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │ [12:00:01] COM3 opened @ 115200 8N1                │ │
│  │ [12:00:01] CSV loaded: 274628 rows                 │ │
│  │ [12:00:18] Replaying: row 115832 / 274628          │ │
│  │ [12:00:42] Buffer overrun warning @ row 145002     │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 4.1 Control inventory

| Widget            | Type          | Source                                             |
|-------------------|---------------|----------------------------------------------------|
| COM port          | `QComboBox`   | `serial.tools.list_ports.comports()`               |
| Baud rate         | `QComboBox`   | `[9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]` |
| Data bits         | `QComboBox`   | `[7, 8]`                                           |
| Parity            | `QComboBox`   | `[None, Even, Odd]`                                |
| Stop bits         | `QComboBox`   | `[1, 2]`                                           |
| Refresh ports     | `QPushButton` | re-enumerates COM ports                            |
| CSV path          | `QLineEdit` + `QPushButton` (file dialog) | user-selected `.csv`               |
| Start / Pause / Stop | 3 × `QPushButton` | drive `SerialWorker` state machine          |
| Loop checkbox     | `QCheckBox`   | `loop_mode` flag                                   |
| Progress bar      | `QProgressBar` | bound to `progress` signal                        |
| Elapsed / ETA     | 2 × `QLabel`  | updated from elapsed seconds + remaining rows      |
| Current-row info  | `QLabel`      | row index, address, FC, direction from latest frame |
| Log panel         | `QTextEdit` (read-only) | accumulates all log signals               |

---

## 5. Binary Frame Format (32 bytes/packet)

Little-endian, matching x86 / ARM-LE native byte order so the MCU
endianness matches.

| Offset | Field           | Type     | Source CSV column   | Notes                                     |
|--------|-----------------|----------|---------------------|-------------------------------------------|
| 0x00   | `slave_addr`    | `uint8`  | `address`           | Modbus slave address                      |
| 0x01   | `function_code` | `uint8`  | `function`          | Modbus function code                      |
| 0x02   | `length`        | `uint8`  | `length`            | original byte count (informational)       |
| 0x03   | `direction`     | `uint8`  | `command response`  | 0 = master command, 1 = slave response    |
| 0x04   | `setpoint`      | `int16`  | `setpoint`          | ×100 (preserves 2 decimals)               |
| 0x06   | `gain`          | `int16`  | `gain`              | ×100                                      |
| 0x08   | `reset_rate`    | `int16`  | `reset rate`        | ×100                                      |
| 0x0A   | `deadband`      | `int16`  | `deadband`          | ×100                                      |
| 0x0C   | `cycle_time`    | `int16`  | `cycle time`        | raw integer                               |
| 0x0E   | `rate`          | `int16`  | `rate`              | ×100                                      |
| 0x10   | `system_mode`   | `uint8`  | `system mode`       | 0–255                                     |
| 0x11   | `control_scheme`| `uint8`  | `control scheme`    | 0–255                                     |
| 0x12   | `pump`          | `uint8`  | `pump`              | 0 = off, 1 = on                           |
| 0x13   | `solenoid`      | `uint8`  | `solenoid`          | 0 = off, 1 = on                           |
| 0x14   | `pressure`      | `float32`| `pressure measurement`| 0–100 PSI (already clipped during preprocessing) |
| 0x18   | `crc_rate`      | `uint32` | `crc rate`          | CRC-error count                           |
| 0x1C   | `time_offset_ms`| `uint32` | derived             | ms since the first packet of the run      |
| **Total** |                |          |                     | **32 bytes**                              |

### 5.1 Encoding conventions

- **NaN handling**: CSV `?` (missing) is mapped to the field's
  zero-equivalent (`int16` → 0, `uint8` → 0, `float32` → quiet
  NaN bit pattern `0x7FC00000`).
- **Scaled-ints (×100)**: preserve 2 decimal places of the
  source float within the `int16` range (32767 / 100 = 327.67,
  more than enough for SCADA control parameters).
- **`time_offset_ms`**: computed by `ReplayEngine` as
  `(row.time − first_row.time) × 1000`. The MCU uses this to
  reconstruct the `time_diff` feature.
- **First-packet direction tag**: the first frame's `direction`
  byte is forced to 1 to mark the run-start reference. (Pure
  cosmetic; can be removed if it interferes with downstream
  parsing.)

---

## 6. Data Flow

### 6.1 Startup

```
User clicks [▶ 开始]
   │
   ▼
MainWindow.on_start_clicked()
   │  1. validate COM + CSV + params → disable GUI controls
   │  2. serial_worker.open(port, baud, ...)
   │  3. replay_engine.load(csv_path)  → rows[]
   │  4. serial_worker.start()        ← starts QThread
   ▼
ReplayEngine.run() executes inside SerialWorker thread:
   t0 = QDateTime.currentMSecsSinceEpoch()
   for row in rows:
       delay = (row.time − first.time)*1000 − (now − t0)
       if delay > 0: QThread.msleep(delay)
       frame = encode(row, delta_ms)             # 32 bytes
       self.serial.write(frame)                  # blocking
       emit progress(row_index, frame_metadata)
   if self.loop_mode:
       current_row = 0
       t0 = now()
       goto for
   else:
       emit finished()
```

### 6.2 Pause / Resume

Pause freezes the wall-clock, not the per-frame work: a paused
frame finishes its current write, then the loop waits in a
100 ms-tick sleep until `_paused` clears. On resume, `_t0` is
*not* reset — the relative offset is preserved, so the schedule
catches up smoothly instead of replaying missed frames all at
once.

### 6.3 Loop

On natural end-of-file: if `loop_mode` is true, reset
`current_row = 0`, reset `_t0 = now()`, and `continue`. Otherwise
emit `finished()` and exit cleanly.

### 6.4 Stop

Stop is checked at the top of every loop iteration. When set,
the loop exits *between frames* (the in-flight 32-byte write
completes), `serial.close()` is called, and `stopped()` is
emitted. The MCU receives one final frame at most; no
half-frames are sent.

---

## 7. Error Handling

| Scenario                       | Detection                                | Handling                                               |
|--------------------------------|------------------------------------------|--------------------------------------------------------|
| COM port missing or busy       | `serial.SerialException`                 | error dialog, do not start                             |
| CSV file missing / malformed   | `pandas.read_csv` exception              | error dialog, log                                      |
| Dataset empty                  | `len(rows) == 0` after load              | error dialog "no valid records"                        |
| Serial-buffer overrun          | `SerialTimeoutException` from `write`    | pause 100 ms, retry up to 3 times; on 3rd failure stop |
| Cable unplugged mid-replay     | `OSError` / `SerialException`            | auto-stop + dialog "serial disconnected"               |
| GUI window closed mid-replay    | `closeEvent`                             | graceful stop, wait ≤ 2 s for thread join             |
| CPU scheduling drift           | `actual_delay − planned_delay`           | log warning, do **not** compensate (preserve pacing)  |

---

## 8. Testing

### 8.1 Unit tests

| Module                              | Test                                          | Verifies                          |
|-------------------------------------|-----------------------------------------------|-----------------------------------|
| `frame_encoder.encode(row)`         | feed synthetic row, assert 32-byte layout      | field offsets, byte order, NaN    |
| `frame_encoder.decode(bytes)`       | round-trip via known row                      | round-trip equality               |
| `timing_calc.compute_delays(rows)`  | feed timestamps, assert `Δt` array            | monotonic scheduling              |
| `csv_loader.load(path)`             | feed truncated / malformed CSV                | raises typed error                |

### 8.2 Integration test

```
1. spawn ReplayEngine writing to a virtual COM (loopback or
   `serial.tools.list_ports`-mocked adapter)
2. open the other end with `pyserial`, capture the byte stream
3. assert:
   • total packets == 274,628 (each packet exactly 32 bytes)
   • `time_offset_ms` sequence matches CSV ± 5 ms
   • decode round-trips for every packet (encode → bytes →
     decode ≡ original row)
```

### 8.3 End-to-end (manual)

```
1. connect MCU (H743) via USB-UART
2. launch simulator, pick the USB-UART COM port
3. MCU receives 274,628 frames, parses, feeds TCN+SE
4. compare MCU detection counts vs offline-PC reference F1m
   (delta < 0.01 acceptable)
```

---

## 9. File / Module Layout

```
modbus_replay_simulator/
├── pyproject.toml                # pyserial, PyQt5, pandas, pytest
├── README.md
├── modbus_replay/
│   ├── __init__.py
│   ├── frame_format.py           # encode/decode + layout constants
│   ├── csv_loader.py             # load + validate IanArffDataset.csv
│   ├── replay_engine.py          # pure timing + encoding
│   ├── serial_worker.py          # QThread: owns pyserial port
│   └── gui/
│       ├── __init__.py
│       ├── main_window.py        # QMainWindow + widget layout
│       └── widgets.py            # reusable ComboBox / ProgressBar wrappers
├── scripts/
│   └── run_simulator.py          # entrypoint: python -m modbus_replay
└── tests/
    ├── test_frame_format.py
    ├── test_csv_loader.py
    ├── test_replay_engine.py
    └── test_integration_loopback.py
```

---

## 10. Open Questions / Future Work

- **Baud rate auto-detection**: when paired with a real Modbus
  RTU slave, baud is usually 9600. The 115200 default here is
  only correct for the bench setup. Add a "match MCU" preset
  that defaults to 115200.
- **MCU-side parser**: out of scope here; will be spec'd
  separately once this simulator lands.
- **Attack injection**: currently we replay the dataset as-is.
  Future enhancement: an "inject attack" toggle that flips
  selected rows to attack variants (e.g., NaN pressure,
  abnormal FC).

---

## 11. Acceptance Criteria

A reviewer should be able to:

1. Run `python scripts/run_simulator.py`, see the GUI.
2. Pick a COM port, baud 115200, choose `IanArffDataset.csv`.
3. Click Start, observe progress bar advance at ≈ real-time
   pace.
4. Connect an MCU running a stub parser; verify it receives
   exactly **274,628 packets × 32 bytes = 8,388,096 bytes** over
   the run.
5. Run `pytest tests/` and see all unit + integration tests
   pass.