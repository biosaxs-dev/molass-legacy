"""
    Selective.ModelFrameStatus.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

class ModelFrameStatus:
    def __init__(self):
        self.has_info = False

    def save_status(self, advanced_frame):
        self.has_info = True

    def restore_status(self, advanced_frame):
        if not self.has_info:
            return
