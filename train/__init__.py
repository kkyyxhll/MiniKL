from MiniKL.models import MiniKLModel

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MiniKL Pretraining')
