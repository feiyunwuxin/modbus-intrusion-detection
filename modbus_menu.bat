@echo off
REM ============================================================
REM  modbus_menu.bat
REM  Modbus SCADA 工具箱 — 菜单式启动器
REM
REM  用法: 双击运行,或 modbus_menu.bat
REM ============================================================

chcp 65001 >nul
setlocal
cd /d "%~dp0"

:MENU
cls
echo.
echo ============================================================
echo   Modbus SCADA 工具箱 v1.0
echo   PC 端气管道 SCADA Modbus TCP 仿真
echo ============================================================
echo.
echo   [1] 启动服务器 (前台,默认 5020)
echo   [2] 启动服务器 (后台新窗口,默认 5020)
echo   [3] 客户端测试 (读 5 次,默认 localhost:5020)
echo   [4] 客户端写 setpoint=25
echo   [5] 一键演示 (服务器 + 客户端 + 攻击)
echo.
echo   [6] 生成攻击数据集 (attack_injector generate)
echo   [7] 生成攻击数据 (2000 normal + 800 attack)
echo.
echo   [8] 验证 Python 环境
echo   [9] 查看 5020 端口占用
echo   [0] 退出
echo.
echo ============================================================
set /p "CHOICE=请选择 [0-9]: "

if "%CHOICE%"=="1" goto START_FOREGROUND
if "%CHOICE%"=="2" goto START_BACKGROUND
if "%CHOICE%"=="3" goto CLIENT_READ
if "%CHOICE%"=="4" goto CLIENT_WRITE
if "%CHOICE%"=="5" goto DEMO
if "%CHOICE%"=="6" goto ATTACK_SMALL
if "%CHOICE%"=="7" goto ATTACK_BIG
if "%CHOICE%"=="8" goto CHECK_ENV
if "%CHOICE%"=="9" goto CHECK_PORT
if "%CHOICE%"=="0" goto END
echo [WARN] 无效选择
timeout /t 2 >nul
goto MENU

:START_FOREGROUND
echo.
echo [INFO] 启动服务器 (前台,Ctrl+C 停止)
call start_server.bat
goto MENU

:START_BACKGROUND
echo.
echo [INFO] 启动服务器 (后台新窗口)
set "PORT=5020"
set /p "PORT=输入端口 [默认 5020]: "
if "%PORT%"=="" set "PORT=5020"
start "Modbus SCADA Server" cmd /k "cd /d %CD% && python modbus_scada_server.py start --port %PORT% && echo. && echo [INFO] 服务器已停止 && pause"
echo [INFO] 新窗口已打开
timeout /t 3 >nul
goto MENU

:CLIENT_READ
echo.
call start_client.bat 127.0.0.1 5020 5 1.0
goto MENU

:CLIENT_WRITE
echo.
echo [INFO] 写 setpoint=25 (写后服务器压力会上升)
call start_client.bat 127.0.0.1 5020 - 25
goto MENU

:DEMO
echo.
call start_demo.bat
goto MENU

:ATTACK_SMALL
echo.
echo [INFO] 生成小型攻击数据集 (500 normal + 200 attack)
python attack_injector.py generate --n-normal 500 --n-attack 200 --out-dir ./attack_test
echo.
echo [INFO] 输出目录: %CD%\attack_test\
echo   - raw_attack_dataset.csv
echo   - mcu_attack_log.jsonl
echo   - report.json
echo.
pause
goto MENU

:ATTACK_BIG
echo.
echo [INFO] 生成大型攻击数据集 (2000 normal + 800 attack)
python attack_injector.py generate --n-normal 2000 --n-attack 800 --out-dir ./attack_test
echo.
echo [INFO] 输出目录: %CD%\attack_test\
echo.
pause
goto MENU

:CHECK_ENV
echo.
echo ============================================================
echo  环境检查
echo ============================================================
echo.
echo [1] Python 版本:
where python >nul 2>nul && python --version || echo     [ERROR] Python 未找到!
echo.
echo [2] pymodbus (可选,本程序不依赖):
python -c "import pymodbus; print('     pymodbus', pymodbus.__version__)" 2>nul || echo     [SKIP] pymodbus 未安装 (OK,不需要)
echo.
echo [3] numpy (可选,用于 NPY 输出):
python -c "import numpy; print('     numpy', numpy.__version__)" 2>nul || echo     [SKIP] numpy 未安装 (NPY 输出不可用,其他正常)
echo.
echo [4] 必需文件检查:
if exist "modbus_scada_server.py" (echo     [OK]   modbus_scada_server.py) else (echo     [MISS] modbus_scada_server.py 缺失!)
if exist "attack_injector.py" (echo     [OK]   attack_injector.py) else (echo     [MISS] attack_injector.py 缺失!)
echo.
echo [5] 工作目录:
echo     %CD%
echo.
pause
goto MENU

:CHECK_PORT
echo.
echo ============================================================
echo  端口 5020 占用情况
echo ============================================================
echo.
netstat -ano | findstr ":5020 " | findstr "LISTENING" >nul 2>nul
if %ERRORLEVEL%==0 (
    echo   [OCCUPIED] 5020 已被占用:
    netstat -ano | findstr ":5020 " | findstr "LISTENING"
    echo.
    echo  解决方法:
    echo    1. 用其他端口: start_server.bat 5021
    echo    2. 找到占用进程: tasklist /fi "PID eq ^<上面最后一列^>"
    echo    3. 结束进程:     taskkill /PID ^<PID^> /F
) else (
    echo   [FREE] 5020 端口空闲
)
echo.
netstat -ano | findstr ":5021 " | findstr "LISTENING" >nul 2>nul && (
    echo  5021 端口: [OCCUPIED]
) || (
    echo  5021 端口: [FREE]
)
echo.
pause
goto MENU

:END
echo.
echo [INFO] 退出
timeout /t 1 >nul
endlocal
exit /b 0
