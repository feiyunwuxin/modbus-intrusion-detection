@echo off
REM ============================================================
REM  serial_tool_gui.bat
REM  一键启动 SCADA Modbus 串口工具 GUI
REM
REM  依赖: Python 3.10+ (含 tkinter) + pyserial
REM  安装依赖: pip install pyserial
REM ============================================================

chcp 65001 > nul
title SCADA Modbus 串口工具

REM 切换到 bat 所在目录
cd /d "%~dp0"

echo.
echo ============================================================
echo   SCADA Modbus 串口工具 v1.0
echo ============================================================
echo.

REM 检查 Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python,请先安装 Python 3.10+
    echo 下载: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM 检查 tkinter
python -c "import tkinter" > nul 2>&1
if errorlevel 1 (
    echo [错误] tkinter 不可用
    echo 解决: 重新安装 Python 时勾选 "tcl/tk and IDLE"
    echo.
    pause
    exit /b 1
)

REM 检查 pyserial
python -c "import serial" > nul 2>&1
if errorlevel 1 (
    echo [提示] pyserial 未安装,正在安装...
    pip install pyserial
    if errorlevel 1 (
        echo [错误] pyserial 安装失败,请手动运行: pip install pyserial
        pause
        exit /b 1
    )
)

REM 检查 matplotlib (曲线图功能依赖)
python -c "import matplotlib" > nul 2>&1
if errorlevel 1 (
    echo [提示] matplotlib 未安装,正在安装...
    pip install matplotlib
    if errorlevel 1 (
        echo [错误] matplotlib 安装失败,曲线图功能不可用
        echo 手动安装: pip install matplotlib
        pause
        exit /b 1
    )
)

REM 启动 GUI
echo 启动 GUI...
echo.
python serial_tool_gui.py

if errorlevel 1 (
    echo.
    echo [错误] GUI 异常退出
    pause
)
