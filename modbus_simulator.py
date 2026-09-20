"""Modbus 模拟器 - 主程序入口。

CLI 模式:
    python modbus_simulator.py [data_file]   # 启动 GUI
    python modbus_simulator.py --self-test   # 自检模式
"""
import sys


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        print("Self-test mode not yet implemented")
        return 1
    print("GUI mode not yet implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())