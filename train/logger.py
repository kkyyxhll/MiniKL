import os

import shutil

class Logger:
    def __init__(self, task_name:str, out_dir=None):
        self.out_dir = os.path.join(os.path.dirname(__file__), f"logger_{task_name}") if out_dir is None else out_dir
        self.out_name = self._init_out_name()

    def _init_out_name(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        count = 0
        for root, dirs, files in os.walk(self.out_dir):
            count = len(files)
        return str(count)+".txt"

    def write(self, log):
        out_path = os.path.join(self.out_dir, self.out_name)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(log)

