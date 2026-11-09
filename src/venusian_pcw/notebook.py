"""Launch the detector from notebook cells without displaying local absolute paths."""

from pathlib import Path
import shlex
import subprocess
import sys

from venus_express.repro.settings import ROOT, display_path, redact_output


def run_cli(*args):
    command = [sys.executable, "-u", "-m", "venusian_pcw", *map(str, args)]
    shown = shlex.join(
        [
            "venusian-pcw",
            *(display_path(arg) if isinstance(arg, Path) else str(arg) for arg in args),
        ]
    )
    print(shown, flush=True)
    with subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        try:
            for line in process.stdout:
                print(redact_output(line, *args), end="", flush=True)
        except KeyboardInterrupt:
            process.terminate()
            raise
        status = process.wait()
    if status:
        raise RuntimeError(f"{shown} 执行失败（退出码 {status}）")


def day_flags(days):
    if days == []:
        raise ValueError("请指定至少一天，或明确设为 None 以运行全量")
    return [value for day in (days or []) for value in ("--day", day)]
