"""
test_plot_headless.py — matplotlib 曲线功能 headless 验证

不弹窗实例化 SerialToolApp, 调用 4 个图表方法:
  1. _init_plot_lines
  2. _on_plot_update (3 次, 检查缓存递增)
  3. _clear_plot (检查清空)
  4. _on_plot_update (检查 _plot_x_counter 是否从 0 重启)

用法: python test_plot_headless.py
退出码 0 = 通过, 1 = 失败
"""

import sys
import tkinter as tk


def main() -> int:
    try:
        # 推迟 import, 触发 matplotlib 后端设置
        from serial_tool_gui import SerialToolApp
    except Exception as e:
        print(f"[FAIL] 无法导入 SerialToolApp: {e}")
        return 1

    root = tk.Tk()
    root.withdraw()  # 隐藏窗口 (headless)

    try:
        app = SerialToolApp(root)
        root.update()  # 让 after/事件跑一次

        # matplotlib 可用性
        if app.plot_fig is None:
            print("[FAIL] matplotlib 不可用, plot_fig 为 None")
            return 1
        print(f"[OK] plot_fig 创建: {type(app.plot_fig).__name__}, "
              f"通道数: {len(app.plot_channels)}, "
              f"缓存上限: {app.plot_max_points}")

        # 1. 初始化 Line2D (实际已在 __init__ 完成, 此处只验证)
        # 注意: 不要重复调用 _init_plot_lines, 会叠加 Line2D!
        assert len(app.plot_lines) == len(app.plot_channels), \
            f"Line2D 数量 {len(app.plot_lines)} != 通道数 {len(app.plot_channels)}"
        print(f"[OK] __init__ 已初始化: {len(app.plot_lines)} 条 Line2D")

        # 2. 推送 3 次数据 (模拟 3 次 FC=3 响应)
        sample_regs = [100, 50, 800, 0, 10, 200, 30, 0, 1, 100]  # 10 个寄存器
        for i in range(3):
            # 微调数据, 让曲线有变化
            test_data = [(v + i * 7) & 0xFFFF for v in sample_regs]
            app._on_plot_update(test_data)
            root.update()
        max_pts = max((len(d) for d in app.plot_data.values()), default=0)
        assert max_pts == 3, f"推送 3 次后期望 max_pts=3, 实际 {max_pts}"
        assert app._plot_x_counter == 3, \
            f"推送 3 次后期望 _plot_x_counter=3, 实际 {app._plot_x_counter}"
        print(f"[OK] _on_plot_update x3: max_pts={max_pts}, "
              f"_plot_x_counter={app._plot_x_counter}")

        # 3. 测试暂停检查
        app.plot_paused_var.set(True)
        before_x = app._plot_x_counter
        before_pts = max(len(d) for d in app.plot_data.values())
        app._on_plot_update(sample_regs)  # 暂停时应被忽略
        root.update()
        after_x = app._plot_x_counter
        after_pts = max(len(d) for d in app.plot_data.values())
        assert after_x == before_x, "暂停时 _plot_x_counter 不应增长"
        assert after_pts == before_pts, "暂停时数据点不应增加"
        print(f"[OK] 暂停检查: x 不变 ({before_x}->{after_x}), "
              f"点数不变 ({before_pts}->{after_pts})")
        app.plot_paused_var.set(False)

        # 4. 测试缓存上限 (push max_points + 5)
        for i in range(app.plot_max_points + 5):
            app._on_plot_update([(100 + i) & 0xFFFF] * 10)
        root.update()
        max_pts = max(len(d) for d in app.plot_data.values())
        assert max_pts == app.plot_max_points, \
            f"缓存应限制在 {app.plot_max_points}, 实际 {max_pts}"
        print(f"[OK] 缓存 FIFO: max_pts={max_pts} == 上限 "
              f"{app.plot_max_points}")

        # 5. 清空
        app._clear_plot()
        root.update()
        max_pts_after_clear = max(len(d) for d in app.plot_data.values())
        assert max_pts_after_clear == 0, \
            f"清空后期望 0 点, 实际 {max_pts_after_clear}"
        assert app._plot_x_counter == 0, \
            f"清空后期望 _plot_x_counter=0, 实际 {app._plot_x_counter}"
        print(f"[OK] _clear_plot: max_pts=0, _plot_x_counter=0")

        # 6. 清空后再推送, _plot_x_counter 从 0 重新计
        app._on_plot_update(sample_regs)
        root.update()
        assert app._plot_x_counter == 1, \
            f"清空后推送 1 次后期望 _plot_x_counter=1, 实际 {app._plot_x_counter}"
        print(f"[OK] 清空后 _plot_x_counter 重启: {app._plot_x_counter}")

        # 7. 复选框可见性切换
        # 默认 setpoint(0) + pressure×100(20) 勾选
        for idx, (name, var) in app.plot_channels.items():
            var.set(True)
        app._update_plot_visibility()
        root.update()
        all_visible = all(line.get_visible() for line in app.plot_lines.values())
        assert all_visible, "勾选所有通道后 Line2D 应可见"
        print(f"[OK] _update_plot_visibility: 全部可见")

        # 保存一个截图文件 (用 matplotlib 后端, 不依赖 GUI)
        import os
        screenshot_path = os.path.abspath("test_plot_screenshot.png")
        try:
            app.plot_fig.savefig(screenshot_path, dpi=100,
                                 facecolor="#2d2d2d")
            print(f"[OK] 截图保存: {screenshot_path}")
        except Exception as e:
            print(f"[WARN] 截图失败: {e}")

        print()
        print("=" * 50)
        print("  ALL 7 CHECKS PASSED — 曲线功能正常")
        print("=" * 50)
        rc = 0
    except AssertionError as e:
        print(f"[FAIL] 断言失败: {e}")
        rc = 1
    except Exception as e:
        print(f"[FAIL] 异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())