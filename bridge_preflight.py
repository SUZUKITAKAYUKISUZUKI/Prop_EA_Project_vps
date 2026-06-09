"""
bridge_preflight.py — MT5 Bridge 起動前チェック

- 既に /health が応答 → 二重起動を回避
- ポート占有のみ（ゾンビ）→ PID 表示 / 任意で解放
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def bridge_health_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}/health"


def fetch_health(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(bridge_health_url(host, port), timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None


def is_bridge_healthy(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 2.0) -> bool:
    return fetch_health(host, port, timeout) is not None


def is_integrated_runtime(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 2.0) -> bool:
    """統合ランタイム（calendar + llm_auditor）が有効な /health か。"""
    data = fetch_health(host, port, timeout)
    if not data:
        return False
    return "calendar" in data and "llm_auditor" in data


def find_listening_pid(port: int = DEFAULT_PORT) -> int | None:
    """Windows netstat から LISTENING PID を取得。"""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
    except OSError:
        return None

    needle = f":{port}"
    for line in result.stdout.splitlines():
        if "LISTENING" not in line.upper():
            continue
        if needle not in line:
            continue
        parts = line.split()
        if parts:
            try:
                return int(parts[-1])
            except ValueError:
                continue
    return None


def kill_pid(pid: int) -> bool:
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def preflight(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, kill_stale: bool = False) -> int:
    """
    Returns exit code:
      0 = OK to start server OR already running (healthy)
      1 = blocked / failed
    """
    if is_bridge_healthy(host, port):
        if is_integrated_runtime(host, port):
            print(f"[INFO] MT5 Bridge は既に起動中です: {bridge_health_url(host, port)}")
            print("[INFO] カレンダー連携 + LLM監査 統合ランタイム稼働中")
            print("[INFO] 新しいサーバーは起動しません（二重起動を回避）")
            return 0

        print(f"[WARN] 旧版 MT5 Bridge が稼働中です: {bridge_health_url(host, port)}")
        print("[WARN] カレンダー連携 / LLM監査 統合には再起動が必要です")
        pid = find_listening_pid(port)
        if kill_stale and pid is not None:
            if kill_pid(pid):
                print(f"[OK] PID {pid} を終了しました。統合ランタイムで再起動します。")
                return 0
            print(f"[ERROR] PID {pid} の終了に失敗しました。")
            return 1
        print("  対処: start_mt5_bridge.bat を再実行（--kill-stale で自動再起動）")
        if pid is not None:
            print(f"        または taskkill /PID {pid} /F")
        return 1

    pid = find_listening_pid(port)
    if pid is not None:
        print(f"[WARN] ポート {port} は PID {pid} により使用中ですが /health は応答しません。")
        if kill_stale:
            if kill_pid(pid):
                print(f"[OK] PID {pid} を終了しました。サーバーを起動します。")
                return 0
            print(f"[ERROR] PID {pid} の終了に失敗しました。")
            return 1
        print("  対処: 以前のサーバーウィンドウを閉じるか、次を実行:")
        print(f"        taskkill /PID {pid} /F")
        print("  または: python bridge_preflight.py --kill-stale")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="MT5 Bridge preflight check")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--kill-stale", action="store_true", help="Free port if unhealthy listener")
    args = parser.parse_args()
    raise SystemExit(preflight(port=args.port, kill_stale=args.kill_stale))


if __name__ == "__main__":
    main()
