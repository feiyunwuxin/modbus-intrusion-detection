@echo off
REM ============================================================
REM  start_server.bat
REM  启动 Modbus SCADA 仿真服务器 (前台运行,Ctrl+C 停止)
REM
REM  用法:
REM    start_server.bat              默认端口 5020
REM    start_server.bat 5020         指定端口
REM    start_server.bat 5020 4       指定端口 + unit_id
REM ============================================================

chcp 65001 >nul
setlocal

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 默认值
set "PORT=5020"
set "UNIT_ID=4"

REM 解析参数
if not "%~1"=="" set "PORT=%~1"
if not "%~2"=="" set "UNIT_ID=%~2"

echo.
echo ============================================================
echo  Modbus SCADA Server
echo ============================================================
echo   端口     : %PORT%
echo   Unit ID  : %UNIT_ID%
echo   工作目录 : %CD%
echo   Python   :
where python >nul 2>nul && (python --version) || (echo   Python 未安装!)
echo ============================================================
echo.

REM 检查端口是否被占用
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul 2>nul
if %ERRORLEVEL%==0 (
    echo [WARN] 端口 %PORT% 已被占用! 请用其他端口:
    echo         start_server.bat 5021
    echo.
    pause
    exit /b 1
)

REM 启动服务器
echo [INFO] 启动服务器... (Ctrl+C 停止)
echo.
python modbus_scada_server.py start --port %PORT% --unit-id %UNIT_ID%

REM 退出后保持窗口
echo.
echo [INFO] 服务器已停止
pause
endlocal
