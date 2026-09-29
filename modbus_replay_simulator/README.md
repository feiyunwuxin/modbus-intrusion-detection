# Modbus Replay Simulator

Desktop tool that reads `IanArffDataset.csv` and replays its rows as
32-byte binary frames over a serial port, preserving the dataset's
`time`-column intervals. Designed for bench-testing an STM32H743
(or STM32F407) running the TCN+SE intrusion-detection model.

## Quick start

```bash
git clone <this repo>
cd modbus_replay_simulator
pip install -e ".[dev]"           # core GUI only
pip install -e ".[ids,dev]"        # adds PC-side IDS monitor (torch etc.)
python -m modbus_replay
```

The GUI launches; pick the COM port your USB-UART bridge exposes,
select `IanArffDataset.csv`, click **▶ 开始**.

The `[ids]` extra pulls in `torch`, `numpy`, `scikit-learn`, and
`joblib` (~hundreds of MB). Without it the GUI still launches and
the IDS panel stays greyed out.

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
| **📡 打开串口** | Probe-open the selected port with a dedicated `pyserial.Serial` handle (independent from the replay worker) AND start the background RX reader that feeds the receive panel. Logs `opened COMx @ baud D P S (probe)` on success. |
| **🔌 关闭串口** | Stop the RX reader and release the probe handle. No-op (with INFO log) if the port is not currently open. |
| **▶ 开始** | Build a SerialWorker thread and start pacing |
| **⏸ 暂停 / ▶ 继续** | Toggle worker pause/resume (label flips) |
| **⏹ 停止** | Set `_stopping`; the loop exits at the next checkpoint |
| **循环播放** | Replay the CSV from the top after the last row |
| **📋 详细日志** | When checked, the worker emits one `TX` log line per frame sent, e.g. `[TX] row=42 t=2100ms addr=4 fc=3 resp hex=01 03 08 00 00 00 00 00`. Off by default — 274k lines per run would flood the panel. Enable for hardware debugging only. |

The probe handle (`_probe_serial`) is released automatically when the
window closes, even if the user opened it and forgot to click
**🔌 关闭串口**. The probe is **not** shared with the replay worker —
each replay run opens its own port via `SerialWorker`.

The progress bar shows row position; the elapsed/ETA label ticks every
500 ms once the first frame has been sent.

## Receive window

Once **📡 打开串口** opens the probe handle, a background
`_SerialReader(QThread)` polls `serial.in_waiting` every 100 ms and
forwards every chunk of received bytes to the bottom panel as
`hex+ASCII` blocks (toggle to `hex` or `ASCII` via the dropdown).
Timestamps are millisecond-precision; the byte counter accumulates
across the lifetime of the probe handle and resets on **🗑 清空**.
The reader stops automatically when **🔌 关闭串口** is clicked, when
the window closes, and before the probe handle is released.

## PC-side IDS monitoring (optional)

The GUI can mirror the H743 firmware's TCN+SE inference on the PC side
so you can compare per-frame probabilities against the JSON the MCU
uploads. Requires `pip install -e ".[ids]"`.

Steps:

1. Tick **🛡 启用 IDS** in the IDS panel (initially greyed out).
2. Pick three files each session — paths are not persisted:
   * **Model** — `*.pt` checkpoint (e.g. `model_v4_se_23dim_b64_ch32_do01_window16_s42.pt`)
   * **Scaler** — `*.joblib` (e.g. `scaler_binary_v2_scada.joblib`)
   * **Threshold** — `*.json` containing `best_threshold` (pre-fills the spinbox)
3. Optionally tick **记录每帧 prob 到 CSV** and pick an output path;
   every scored frame appends one row
   `(row_idx, time_ms, addr, fc, direction, prob, is_alarm)`.
   Re-runs append without a duplicate header.
4. Click **▶ 开始**. The status label flips from `window: 0/16` to
   `scoring` after 16 frames; then **已评分 / 报警数 / 报警率** start
   advancing. The alarm-log table keeps the most recent 200 alarms;
   per-frame data lives in the CSV.

Architecture used: 23-dim TCN+SE (`in_ch=23, n_blocks=3, channels=32,
kernel_size=3, dilations=(1,2,4), dropout=0.1, se_reduction=8`) — see
`modbus_replay/ids_worker_torch.py`.

Cross-check vs H743: feed the same CSV and compare
`pandas.read_csv('ids_log.csv')['is_alarm']` against the MCU's JSON
report for the same frame indices.

## Tests

```bash
pytest -v
```

92 unit tests cover the nine modules (csv_loader, frame_format,
replay_engine, serial_worker, gui.widgets, gui.main_window,
gui.rx_panel, integration_loopback, ids_pipeline+panel+worker).
End-to-end hardware validation is the manual checklist in
`docs/MANUAL_VALIDATION.md` (Task 10).

## License

Research use, see repo root LICENSE.
