import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(os.path.dirname(os.path.dirname(__file__)))
import warnings
import shutil
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import argparse

from logger import Logger
from models import MiniKLModel, MiniKLConfig
from dataset import PretrainDataset
from tokenizer import BaseTokenizer, TokenizerConfig
from tqdm import tqdm

import math

import matplotlib.pyplot as plt

import torch.distributed as dist
import torch.utils.data.distributed

import random
import wandb

def get_now_lr(total_steps, now_step, lr):
    return lr / 10.0 + 0.5 * lr * (1 + math.cos(math.pi * now_step / total_steps))

def exp_moving_average(data, alpha=0.9):
    if not data:
        return []
    output = [data[0]]
    for value in data[1:]:
        avg = alpha*value + (1-alpha)*output[-1]
        output.append(avg)
    return output

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser("MiniKL Pretrain Args")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--vocab_dict_path", type=str, default=r"/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json")
    parser.add_argument("--data_jsonl_path", type=str, default=r"/home/kkyyxhll/Projects/PythonProjects/MiniKL/data/test_pretrain.jsonl")
    parser.add_argument("--model_save_dir", default="saved_pretrain_model", type=str)
    parser.add_argument("--load_model_path", default="pretrain_model.pth", type=str)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="loukang")
    parser.add_argument("--wandb_project", type=str, default="test")
    args = parser.parse_args()

    logger = Logger(task_name="pretrain", )

    torch.manual_seed(42)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(42)

    tokenizer_config = TokenizerConfig(mode="test", vocab_dict_path=args.vocab_dict_path, max_seq_len=args.max_seq_len)
    tokenizer = BaseTokenizer(tokenizer_config)

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    args.device = f"cuda:{local_rank}"
    if os.path.exists(args.model_save_dir):
        shutil.rmtree(args.model_save_dir)
    os.mkdir(args.model_save_dir)
    print(args)
    if local_rank == 0 and args.wandb:
        run = wandb.init(entity=args.wandb_entity,
                         # Set the wandb project where this run will be logged.
                         project=args.wandb_project,
                         # Track hyperparameters and run metadata.
                         config={
                             "learning_rate": args.lr,
                             "architecture": "MiniKL Decoder",
                             "dataset": "books",
                             "epochs": args.epochs,
                             "batch_size": args.batch_size
                         }, )

    vocab_size = tokenizer.get_vocab_size()

    pretrain_dataset = PretrainDataset(tokenizer, args.data_jsonl_path)
    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_dataset)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.batch_size, sampler=pretrain_sampler)

    model_config = MiniKLConfig(vocab_size=vocab_size,)
    model = MiniKLModel(model_config).to(args.device)
    if os.path.exists(args.load_model_path):
        print(f"载入预训练模型:{args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.AdamW(model.parameters(),lr=args.lr)

    criterion = nn.CrossEntropyLoss(reduction="none")
    scaler = torch.amp.GradScaler(device=args.device.split(":")[0], )


    per_epoch_steps = len(pretrain_dataloader)
    all_steps = args.epochs * per_epoch_steps

    all_losses = []
    for e in range(args.epochs):

        with tqdm(pretrain_dataloader, unit="train") as pbar:
            for i, (x, y, padding_masks) in enumerate(pbar):
                x = x.to(args.device)
                y = y.to(args.device)
                padding_masks = padding_masks.to(args.device)

                lr = get_now_lr(all_steps, e*per_epoch_steps + i, args.lr)
                optimizer.param_groups[0]["lr"] = lr

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.float16):
                    pred_y = model(x)
                    pred_y = pred_y.transpose(-1, -2)
                    loss_masked = criterion(pred_y, y) * padding_masks
                    loss = torch.mean(loss_masked)

                scaler.scale(loss).backward()  # 缩放梯度
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                all_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.set_description(f"device:{args.device} epoch:[{e+1}|{args.epochs}] step:[{i+1}|{per_epoch_steps}] lr:[{lr:.4f}]")

                log = f"device:{args.device}  epoch:[{e + 1}|{args.epochs}], step:[{i+1}|{per_epoch_steps}], lr:[{lr:.4f}], loss:{loss.item():.4f}"
                logger.write(log)
                if args.wandb:
                    if dist.get_rank() == 0:
                        run.log({"epoch": e+1,
                                 "step": i+1,
                                 "loss":loss.item()})
                if (i + 1) % 1000 == 0:
                    if dist.get_rank() == 0:
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()
                        model_save_path = os.path.join(args.model_save_dir, f"sft_model_{e * per_epoch_steps + i}.pth")
                        torch.save(state_dict, model_save_path)
        pbar.update()

    if dist.get_rank() == 0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        model_save_path = os.path.join(args.model_save_dir, f"pretrain_model.pth")
        print(f"model saved {model_save_path}")
        torch.save(state_dict, model_save_path)
    x = [e for e in range(all_steps)]
    all_losses = exp_moving_average(all_losses)
    plt.figure()
    plt.plot(x, all_losses)
    plt.title("loss | epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    if not os.path.exists("loss_pngs"):
        os.mkdir("loss_pngs")
    plt.savefig(os.path.join("loss_pngs", "sft_loss.png"))

    if args.wandb:
        run.finish()



