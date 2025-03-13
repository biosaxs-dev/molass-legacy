"""
    EnvMessages.py

    Copyright (c) 2019-2022, SAXS Team, KEK-PF
"""
from .EnvInfo import get_global_env_info

class LogPicker:
    def __init__(self):
        self.log_info = []

    def info(self, m):
        self.log_info.append([0, m])

    def warning(self, m):
        self.log_info.append([1, m])

def get_env_messages():
    messages = []

    picker = LogPicker()
    envinfo = get_global_env_info(gpu_info=True)
    envinfo.show_and_log_if_unavailable(None, picker)
    for level, message in picker.log_info:
        messages.append(message + "\n")

    messages.append("Reason:\n")

    texts = envinfo.get_gpu_reason_texts()
    for text in texts:
        messages.append("    " + text + "\n")

    return messages
