"""Windows integration helpers."""
from src.windows.task_scheduler import TASK_NAME, install_task, task_exists, uninstall_task

__all__ = ["TASK_NAME", "install_task", "task_exists", "uninstall_task"]
