@echo off
REM ============================================================
REM  start_client.bat
REM  启动 Modbus 客户端测试 (连接 + 读 + 可选写)
REM
REM  用法:
REM    start_client.bat                          默认 localhost:5020, 读 5 次
REM    start_client.bat 127.0.0.1 5020           指定主机端口
REM    start_client.bat 127.0.0.1 5020 10        读 10 次
REM    start_client.bat 127.0.0.1 5020 5 0.5     读 5 次, 间隔 0.5s
REM    start_client.bat 127.0.0.1 5020 - 25      写 setpoint=25
REM ============================================================

chcp 65001 >nul
setlocal

cd /d "%~dp0"

REM 默认值
set "HOST=127.0.0.1"
set "PORT=5020"
set "READ=5"
set "INTERVAL=1.0"

REM 解析参数
if not "%~1"=="" set "HOST=%~1"
if not "%~2"=="" set "PORT=%~2"

REM 模式 1: 写 setpoint
if /I "%~3"=="-" (
    if not "%~4"=="" (
        echo [INFO] 写 setpoint = %~4
        python modbus_scada_server.py client --host %HOST% --port %PORT% --write-sp %~4
        goto :end
    )
)

REM 模式 2: 循环读
if not "%~3"=="" set "READ=%~3"
if not "%~4"=="" set "INTERVAL=%~4"

echo.
echo ============================================================
echo  Modbus SCADA Client
echo ============================================================
echo   主机     : %HOST%
echo   端口     : %PORT%
echo   读次数   : %READ%
echo   间隔     : %INTERVAL%s
echo ============================================================
echo.

python modbus_scada_server.py client --host %HOST% --port %PORT% --read %READ% --interval %INTERVAL%

:end
echo.
pause
endlocal
