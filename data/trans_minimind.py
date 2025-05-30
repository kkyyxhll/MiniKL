import json

import os
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimind_path", default="/home/kkyyxhll/Projects/PythonProjects/MiniKL/data/pretrain_hq.jsonl", type=str)
    parser.add_argument("--output_path", default="out.jsonl", type=str)
    args = parser.parse_args()


