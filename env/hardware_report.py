"""Emit a plain-text description of the machine this ran on.

Written to env/hardware_report.txt by setup.sh. Stdlib only, so it runs before any
third-party package is installed. GPU/VRAM detection is best-effort via nvidia-smi;
if that isn't present we say so rather than guessing.
"""

from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
from datetime import UTC, datetime


def total_ram_gb() -> str:
    """Total physical RAM in GB, or 'unknown' if the platform path isn't covered."""
    system = platform.system()
    try:
        if system == "Windows":
            # GlobalMemoryStatusEx via ctypes avoids a psutil dependency.
            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return f"{stat.ullTotalPhys / 1024**3:.1f}"
        if system in ("Linux", "Darwin"):
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return f"{kb / 1024**2:.1f}"
    except Exception:  # noqa: BLE001 - reporting must never crash setup
        return "unknown"
    return "unknown"


def gpu_report() -> str:
    """Query nvidia-smi if available; otherwise report that no NVIDIA GPU was detected."""
    if shutil.which("nvidia-smi") is None:
        return "No nvidia-smi found (no NVIDIA GPU detected, or driver not installed)."
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return out.stdout.strip() or "nvidia-smi returned no rows."
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi call failed: {exc}"


def main() -> None:
    lines = [
        f"Generated (UTC): {datetime.now(UTC).isoformat()}",
        f"OS: {platform.platform()}",
        f"Machine: {platform.machine()}",
        f"Processor: {platform.processor() or 'unknown'}",
        f"Logical CPUs: {shutil.os.cpu_count()}",
        f"Total RAM (GB): {total_ram_gb()}",
        f"Python: {platform.python_version()}",
        "GPU:",
        f"  {gpu_report()}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
