"""Windows Task Scheduler integration for PortfolioOS import daemon."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.sax.saxutils as saxutils
from pathlib import Path

TASK_NAME = "PortfolioOS Import Daemon"
STARTUP_BAT_NAME = "PortfolioOS Import Daemon.bat"
ACCESS_DENIED_HINT = (
    "Access denied while creating the scheduled task.\n"
    "Try one of:\n"
    "  1) python src\\windows\\install_task.py --method startup   (no admin)\n"
    "  2) python src\\windows\\install_task.py --boot               (admin PowerShell)\n"
    "  3) Right-click PowerShell → Run as administrator, then re-run install_task.py"
)


def _python_executable() -> str:
    return sys.executable


def _runner_script(project_root: Path) -> Path:
    return project_root / "src" / "daemon" / "daemon_runner.py"


def resolve_windows_username(username: str | None = None) -> str:
    if username:
        if "\\" in username or "@" in username:
            return username
        domain = os.environ.get("USERDOMAIN", "").strip()
        return f"{domain}\\{username}" if domain else username
    domain = os.environ.get("USERDOMAIN", "").strip()
    user = os.environ.get("USERNAME", "").strip()
    if domain and user:
        return f"{domain}\\{user}"
    return user or "SYSTEM"


def build_task_command(project_root: Path) -> str:
    python_exe = _python_executable()
    runner = _runner_script(project_root)
    work_dir = str(project_root)
    return f"\"{python_exe}\" \"{runner}\""


def build_task_xml(
    *,
    project_root: Path,
    task_name: str = TASK_NAME,
    username: str | None = None,
    logon_type: str = "InteractiveToken",
    include_boot: bool = False,
) -> str:
    python_exe = _python_executable()
    runner = _runner_script(project_root)
    work_dir = str(project_root)
    command = saxutils.escape(python_exe)
    arguments = saxutils.escape(f"\"{runner}\"")
    working = saxutils.escape(work_dir)
    user = saxutils.escape(resolve_windows_username(username))

    boot_trigger = ""
    if include_boot:
        boot_trigger = """
    <BootTrigger>
      <Enabled>true</Enabled>
      <Delay>PT2M</Delay>
    </BootTrigger>"""

    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>PortfolioOS Dropbox import daemon — auto-ingest VPS trade logs into SQLite</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <Delay>PT1M</Delay>
    </LogonTrigger>{boot_trigger}
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>{logon_type}</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{command}</Command>
      <Arguments>{arguments}</Arguments>
      <WorkingDirectory>{working}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def startup_bat_path() -> Path:
    appdata = os.environ.get("APPDATA", "")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / STARTUP_BAT_NAME


def startup_installed() -> bool:
    path = startup_bat_path()
    return path.exists() and path.is_file()


def install_startup_entry(project_root: Path) -> tuple[bool, str]:
    bat_path = startup_bat_path()
    bat_path.parent.mkdir(parents=True, exist_ok=True)
    python_exe = _python_executable()
    runner = _runner_script(project_root)
    content = (
        "@echo off\r\n"
        f"cd /d \"{project_root}\"\r\n"
        f"set DROPBOX_DATA_FLOW_ROLE=consumer\r\n"
        f"start \"PortfolioOS Import Daemon\" /MIN \"{python_exe}\" \"{runner}\"\r\n"
    )
    bat_path.write_text(content, encoding="utf-8")
    return True, f"Installed Startup entry: {bat_path}"


def uninstall_startup_entry() -> tuple[bool, str]:
    bat_path = startup_bat_path()
    if not bat_path.exists():
        return True, f"Startup entry not present: {bat_path}"
    bat_path.unlink(missing_ok=True)
    return True, f"Removed Startup entry: {bat_path}"


def task_exists(task_name: str = TASK_NAME) -> bool:
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", task_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def daemon_autostart_installed(task_name: str = TASK_NAME) -> bool:
    return task_exists(task_name) or startup_installed()


def _is_access_denied(message: str) -> bool:
    lower = message.lower()
    return "access is denied" in lower or "アクセスが拒否" in message


def install_task_cli(
    project_root: Path,
    *,
    task_name: str = TASK_NAME,
) -> tuple[bool, str]:
    command = build_task_command(project_root)
    result = subprocess.run(
        [
            "schtasks",
            "/Create",
            "/TN",
            task_name,
            "/TR",
            command,
            "/SC",
            "ONLOGON",
            "/RL",
            "LIMITED",
            "/F",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "schtasks failed").strip()
        if _is_access_denied(msg):
            return False, ACCESS_DENIED_HINT
        return False, msg
    return True, f"Installed scheduled task (logon): {task_name}"


def install_task_xml(
    project_root: Path,
    *,
    task_name: str = TASK_NAME,
    username: str | None = None,
    include_boot: bool = False,
) -> tuple[bool, str]:
    xml_content = build_task_xml(
        project_root=project_root,
        task_name=task_name,
        username=username,
        include_boot=include_boot,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-16") as tmp:
        tmp.write(xml_content)
        xml_path = tmp.name

    try:
        result = subprocess.run(
            ["schtasks", "/Create", "/TN", task_name, "/XML", xml_path, "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "schtasks failed").strip()
            if _is_access_denied(msg):
                hint = ACCESS_DENIED_HINT
                if include_boot:
                    hint += "\nBoot trigger requires administrator rights. Retry without --boot."
                return False, hint
            return False, msg
        mode = "logon+boot" if include_boot else "logon"
        return True, f"Installed scheduled task ({mode}): {task_name}"
    finally:
        Path(xml_path).unlink(missing_ok=True)


def install_task(
    project_root: Path,
    *,
    task_name: str = TASK_NAME,
    username: str | None = None,
    force: bool = True,
    include_boot: bool = False,
    method: str = "auto",
) -> tuple[bool, str]:
    if task_exists(task_name):
        if not force:
            return False, f"Task already exists: {task_name}"
        uninstall_task(task_name)

    if method == "startup":
        return install_startup_entry(project_root)

    if method == "cli":
        return install_task_cli(project_root, task_name=task_name)

    if method == "xml":
        return install_task_xml(
            project_root,
            task_name=task_name,
            username=username,
            include_boot=include_boot,
        )

    ok, message = install_task_xml(
        project_root,
        task_name=task_name,
        username=username,
        include_boot=include_boot,
    )
    if ok:
        return ok, message

    ok, message = install_task_cli(project_root, task_name=task_name)
    if ok:
        return ok, message

    if _is_access_denied(message):
        ok, startup_msg = install_startup_entry(project_root)
        if ok:
            return True, startup_msg + "\n(Task Scheduler unavailable — using Startup folder instead.)"
    return False, message


def uninstall_task(task_name: str = TASK_NAME) -> tuple[bool, str]:
    if not task_exists(task_name):
        return True, f"Task not present: {task_name}"
    result = subprocess.run(
        ["schtasks", "/Delete", "/TN", task_name, "/F"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "schtasks delete failed").strip()
        return False, msg
    return True, f"Removed scheduled task: {task_name}"


def uninstall_all(task_name: str = TASK_NAME) -> tuple[bool, str]:
    messages: list[str] = []
    ok = True
    if task_exists(task_name):
        task_ok, task_msg = uninstall_task(task_name)
        messages.append(task_msg)
        ok = ok and task_ok
    if startup_installed():
        startup_ok, startup_msg = uninstall_startup_entry()
        messages.append(startup_msg)
        ok = ok and startup_ok
    if not messages:
        messages.append("No autostart entry found")
    return ok, "\n".join(messages)


def run_task_now(task_name: str = TASK_NAME) -> tuple[bool, str]:
    result = subprocess.run(
        ["schtasks", "/Run", "/TN", task_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "run failed").strip()
    return True, f"Started task: {task_name}"
