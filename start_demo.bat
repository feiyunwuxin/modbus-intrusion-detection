@echo off
REM ============================================================
REM  start_demo.bat
REM  一键演示: 自动启动服务器 + 客户端 + 注入 3 类攻击
REM
REM  用法:
REM    start_demo.bat           默认端口 5020
REM    start_demo.bat 5021      指定端口
REM ============================================================

chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "PORT=5020"
if not "%~1"=="" set "PORT=%~1"

echo.
echo ============================================================
echo  Modbus SCADA 一键演示
echo ============================================================
echo   端口 : %PORT%
echo   内容 : 启动服务器 + 客户端 + 注入 MPCI/MSCI/NMRI
echo   退出 : Ctrl+C 中断,或等待自动结束
echo ============================================================
echo.

python modbus_scada_server.py demo --port %PORT%

echo.
echo [INFO] 演示完成
pause
endlocal
