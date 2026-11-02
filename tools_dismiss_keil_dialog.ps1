# tools_dismiss_keil_dialog.ps1
# uVision 批量构建失败后会弹一个模态框 ("No Browse Information available ..."),
# 它会卡住命令行构建. 这个脚本把 uVision 进程里的模态框关掉.
param(
    [string]$TitleLike = "",
    [int]$TimeoutSec = 300
)

Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class UvDlg {
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc cb, IntPtr l);
    [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr p, EnumWindowsProc cb, IntPtr l);
    [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, uint msg, IntPtr wp, IntPtr lp);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint msg, IntPtr wp, IntPtr lp);
    public delegate bool EnumWindowsProc(IntPtr h, IntPtr l);

    // 找到 uVision 里的模态对话框, 点它的按钮 (IDOK=1)
    public static int Dismiss() {
        int hit = 0;
        EnumWindows((h, l) => {
            uint pid; GetWindowThreadProcessId(h, out pid);
            var c = new StringBuilder(256); GetClassName(h, c, 256);
            if (c.ToString() != "#32770" || !IsWindowVisible(h)) return true;
            // 只处理 uVision 的对话框: 标题里有 Vision
            var t = new StringBuilder(512); GetWindowText(h, t, 512);
            if (!t.ToString().Contains("Vision")) return true;
            EnumChildWindows(h, (ch, l2) => {
                var cc = new StringBuilder(256); GetClassName(ch, cc, 256);
                if (cc.ToString() == "Button") {
                    var bt = new StringBuilder(128); GetWindowText(ch, bt, 128);
                    int id = GetDlgCtrlID(ch);
                    if (id == 1 || id == 2) {   // IDOK / IDCANCEL
                        PostMessage(ch, 0x00F5, IntPtr.Zero, IntPtr.Zero); // BM_CLICK
                        hit++;
                    }
                }
                return true;
            }, IntPtr.Zero);
            return true;
        }, IntPtr.Zero);
        return hit;
    }

    [DllImport("user32.dll")] public static extern int GetDlgCtrlID(IntPtr h);
}
'@

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    $n = [UvDlg]::Dismiss()
    if ($n -gt 0) { Write-Output "dismissed $n dialog button(s)" }
    Start-Sleep -Milliseconds 700
}
