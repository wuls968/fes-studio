from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .paths import APP_PATH, PROJECT_ROOT, RUNTIME_DIR, SRC_ROOT

PRECHECK_MODULES = ["numpy", "pandas", "matplotlib", "scipy", "streamlit", "plotly", "openpyxl"]
PORT_RANGE = range(8501, 8536)


def build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(SRC_ROOT)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not current else src_path + os.pathsep + current
    return env


def import_failures() -> list[str]:
    failures: list[str] = []
    for module_name in PRECHECK_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - used as runtime guard
            failures.append(f"{module_name}: {exc}")
    return failures


def _venv_root() -> Path:
    return PROJECT_ROOT / ".venv"


def _macos_binary_targets(venv_root: Path) -> list[Path]:
    return [*venv_root.rglob("*.so"), *venv_root.rglob("*.dylib")]


def repair_macos_environment() -> list[str]:
    messages: list[str] = []
    if sys.platform != "darwin":
        messages.append("Platform-specific repair is only required on macOS.")
        return messages

    venv_root = _venv_root()
    if not venv_root.exists():
        messages.append(f"No local virtual environment was found at {venv_root}.")
        return messages

    if shutil.which("xattr") is None or shutil.which("codesign") is None:
        messages.append("Skipping macOS repair because xattr or codesign is unavailable.")
        return messages

    for attribute in ("com.apple.provenance", "com.apple.quarantine"):
        subprocess.run(["xattr", "-dr", attribute, str(venv_root)], check=False, capture_output=True)
    targets = _macos_binary_targets(venv_root)
    for file_path in targets:
        subprocess.run(["codesign", "--force", "--sign", "-", str(file_path)], check=False, capture_output=True)
    messages.append(f"Processed {len(targets)} binary extension files under {venv_root}.")
    return messages


def ensure_environment_ready() -> None:
    failures = import_failures()
    if not failures:
        return
    print("Detected a broken Python environment. Attempting automatic repair...", flush=True)
    for message in repair_macos_environment():
        print(message, flush=True)
    failures = import_failures()
    if not failures:
        return
    print("Environment preflight failed:", flush=True)
    for item in failures:
        print(f"  - {item}", flush=True)
    raise SystemExit(1)


def _status_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _http_ready(port: int, timeout: float = 0.75) -> bool:
    request = Request(_status_url(port), headers={"User-Agent": "FES-Studio-Launcher"})
    try:
        with contextlib.closing(urlopen(request, timeout=timeout)) as response:
            if response.status != 200:
                return False
            body = response.read(4096).decode("utf-8", errors="ignore").lower()
            return "streamlit" in body or "fes studio" in body
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def _read_runtime_integer(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def find_recorded_running_port() -> int | None:
    port = _read_runtime_integer(RUNTIME_DIR / "streamlit.port")
    if port and _http_ready(port):
        return port
    return None


def find_free_port() -> int:
    for port in PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"No free port found in {PORT_RANGE.start}-{PORT_RANGE.stop - 1}.")


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def open_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url, new=1, autoraise=True))
    except Exception:
        return False


def _spawn_browser_watcher(port: int, timeout_seconds: float) -> None:
    def worker() -> None:
        deadline = time.time() + timeout_seconds
        url = _status_url(port)
        while time.time() < deadline:
            if _http_ready(port, timeout=0.4):
                if not open_browser(url):
                    print(f"Browser auto-open did not complete. Please open {url} manually.", flush=True)
                return
            time.sleep(0.5)
        print(f"Browser auto-open timed out. Please open {url} manually.", flush=True)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def write_runtime_state(port: int, pid: int | None = None) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    (RUNTIME_DIR / "streamlit.port").write_text(str(port), encoding="utf-8")
    if pid is not None:
        (RUNTIME_DIR / "streamlit.pid").write_text(str(pid), encoding="utf-8")


def clear_runtime_state() -> None:
    for file_path in (RUNTIME_DIR / "streamlit.pid", RUNTIME_DIR / "streamlit.port"):
        with contextlib.suppress(FileNotFoundError):
            file_path.unlink()


def launch(args: argparse.Namespace) -> int:
    ensure_environment_ready()

    running_port = find_recorded_running_port()
    if running_port:
        url = _status_url(running_port)
        print(f"FES Studio is already running on {url}", flush=True)
        if not args.no_browser:
            if not open_browser(url):
                print(f"Please open {url} manually.", flush=True)
        return 0

    if args.port is not None:
        if not is_port_available(args.port):
            raise SystemExit(f"Requested port {args.port} is already in use.")
        port = args.port
    else:
        port = find_free_port()
    url = _status_url(port)
    print(f"Starting FES Studio on {url}", flush=True)
    if args.no_browser:
        print("Browser auto-open is disabled for this launch.", flush=True)
    else:
        print("The browser should open automatically in a few seconds. Keep this window open while using FES Studio.", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=build_runtime_env())
    write_runtime_state(port, process.pid)
    if not args.no_browser:
        _spawn_browser_watcher(port=port, timeout_seconds=args.browser_timeout)

    try:
        return process.wait()
    except KeyboardInterrupt:
        with contextlib.suppress(OSError):
            process.terminate()
        with contextlib.suppress(Exception):
            process.wait(timeout=5)
        print("Stopping FES Studio...", flush=True)
        return 130
    finally:
        clear_runtime_state()


def preflight(_: argparse.Namespace) -> int:
    failures = import_failures()
    if failures:
        print("Environment preflight failed:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("Environment preflight passed.")
    for module_name in PRECHECK_MODULES:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{module_name} {version}")
    print(f"project_root {PROJECT_ROOT}")
    return 0


def repair(_: argparse.Namespace) -> int:
    for message in repair_macos_environment():
        print(message)
    return preflight(argparse.Namespace())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-platform launcher utilities for FES Studio.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch", help="Start the Streamlit interface.")
    launch_parser.add_argument("--port", type=int, default=None, help="Preferred local port. Defaults to the first free port in 8501-8535.")
    launch_parser.add_argument("--no-browser", action="store_true", help="Do not try to open a browser automatically.")
    launch_parser.add_argument("--browser-timeout", type=float, default=30.0, help="Seconds to wait before giving up on browser auto-open.")
    launch_parser.set_defaults(handler=launch)

    preflight_parser = subparsers.add_parser("preflight", help="Check the local Python environment.")
    preflight_parser.set_defaults(handler=preflight)

    repair_parser = subparsers.add_parser("repair", help="Run platform-specific environment repair, then re-check imports.")
    repair_parser.set_defaults(handler=repair)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def launch_entrypoint() -> None:
    raise SystemExit(main(["launch"]))


def repair_entrypoint() -> None:
    raise SystemExit(main(["repair"]))


if __name__ == "__main__":
    raise SystemExit(main())
