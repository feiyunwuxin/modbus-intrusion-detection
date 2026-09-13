# Modbus Replay Simulator — Manual End-to-End Validation

This is the final gate before declaring the simulator usable on the
bench. It is **manual** because the target is a real STM32H743 (or
STM32F407) wired to a USB-UART bridge — there is no CI rig that
exercises real serial hardware. Run this checklist once per release
candidate (or after any change to `frame_format.py`,
`replay_engine.py`, `serial_worker.py`, or `gui/main_window.py`).

Automated coverage already in place (40 tests):

* 6 `csv_loader` — preserves `?`, missing-column guard, etc.
* 8 `frame_format` — round-trip, NaN encoding, size = 32, etc.
* 7 `replay_engine` — pacing, encoding, first-packet direction.
* 5 `serial_worker` — write progress, stop, error, paused-state, etc.
* 5 `gui/widgets` — PortSelector defaults, ProgressPanel, LogPanel.
* 4 `gui/main_window` — composition + missing-CSV / missing-port guards.
* 5 `integration_loopback` — 50-row round-trip, NaN round-trip,
  first-packet direction, real-dataset size guard (274,628 rows),
  real-dataset first-50-frame round-trip.

If any automated test fails, **stop** and fix before proceeding.

## Step 1 — Bench wiring

1. Connect the STM32H743 (or STM32F407) board to the PC with a
   USB-UART bridge. Confirm the bridge enumerates as a known
   COM port (e.g. `COM5`) in Device Manager.
2. Verify the MCU firmware is flashed and the TCN+SE intrusion
   detection loop is running (it should be logging via SWO / UART).
3. Open the GUI on the PC:

   ```bash
   cd modbus_replay_simulator
   python -m modbus_replay
   ```

4. In the GUI:
   - Pick the COM port exposed by the bridge from the **COM** dropdown.
   - Set **Baud** = `115200`, **Data** = `8`, **Parity** = `N`,
     **Stop** = `1` (the H743 default).
   - Click **🔄 刷新** if the port list is stale or empty.
5. Confirm the **CSV** field shows
   `C:\work\Claude\Issue\IanArffDataset.csv` (the default). If the
   dataset lives elsewhere, click **📂 选择 CSV…** and pick it.

Expected: GUI is responsive; no tracebacks in the log panel.

## Step 2 — CSV load

1. Click **▶ 开始**. The Start button disables; Pause and Stop
   enable; the port selector + CSV controls disable.
2. The progress bar's maximum snaps to **274,628** after the worker
   loads rows.
3. The log panel shows (in order):
   - `CSV loaded: 274628 rows`
   - `opened COMx @ 115200 8N1`
   - `state → running`

If `opened COMx …` does not appear within 1 s, the port is busy —
close any other process (PuTTY, Arduino IDE Serial Monitor) and try
again.

## Step 3 — Run and observe

1. Watch the progress bar advance. It should move at ≈ real-time
   pace — for the 274k-row dataset at 1 row/second the full run is
   ≈ 76 hours; sub-second sampling collapses this dramatically in
   practice.
2. The current-row label updates roughly once per frame:
   `row=N addr=A fc=F resp` (or `cmd` for master-initiated frames).
3. The elapsed/ETA label ticks every 500 ms once the first frame
   has been sent.

Expected: no ERROR / WARN lines in the log. A single WARN
(`write error at row N`) followed by automatic retry is acceptable;
two consecutive ERRORs indicate a real fault and you should stop.

## Step 4 — Pause / Resume

1. Click **⏸ 暂停**. Log shows `paused`; the button label flips to
   **▶ 继续**.
2. After ~5 s, click **▶ 继续**. Log shows `resumed`; the progress
   bar resumes from where it left off (no replay burst on resume).
3. The button label flips back to **⏸ 暂停**.

Expected: progress never jumps backward on resume; the elapsed
timer continues to advance through the pause window.

## Step 5 — Stop

1. Click **⏹ 停止**. Log shows `stop requested` then `run stopped`;
   the Start button re-enables; Pause and Stop disable; the port
   selector + CSV controls re-enable.
2. The serial port is closed cleanly (no orphaned handle — re-run
   Step 2 should succeed without rebooting).

## Step 6 — MCU-side verification

On the MCU, confirm via SWO / UART log:

1. `HAL_UART_Receive_DMA` reports the expected number of complete
   32-byte DMA transfers (matches `progress` log line count on the
   PC side, modulo the cut-off at Stop).
2. Each 32-byte buffer parses without residue — the MCU's frame
   decoder should produce no error count increment.
3. The TCN+SE inference count matches the dataset's positive
   samples ± 1 (offline reference: ≈ 24 % of windows are attack).

Expected: the MCU's intrusion-detection log mirrors the PC-side
progress without drops, overruns, or framing errors.

## Step 7 — Release tag (optional)

If this validation passed and you want to mark the release:

```bash
cd modbus_replay_simulator
git tag -a v0.1.0 -m "Modbus Replay Simulator v0.1.0 — initial bench release"
git push origin v0.1.0
```

(Only tag after Step 6 passes on real hardware. Tagging is a release
action — confirm with the maintainer before pushing.)

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ERROR open failed: PermissionError` | Another process holds the COM port | Close PuTTY / Arduino IDE / another simulator instance |
| `ERROR open failed: FileNotFoundError` | Wrong COM port selected | Click **🔄 刷新** and re-pick |
| Progress bar stuck at 0 | MCU is not actually reading; pyserial write buffer full | Check MCU UART config; reduce baud; check ground connection |
| WARN `write error at row N` followed by retry | Transient USB hiccup | Benign once; repeated → replace cable |
| First packet decoded as `direction=0` on MCU | MCU forgot to honor the `is_first_packet` flag | Re-flash firmware with the documented first-packet convention |

## Pass / fail criteria

The release is **passing** when Steps 1–6 all succeed with no
unrecovered errors. Step 7 (tagging) is a release action; the
release is "ready to tag" once Steps 1–6 are green.
