import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shutil

from models import MiniKLModel, MiniKLConfig
from dataset import SFTDataset
from tokenizer import BaseTokenizer, TokenizerConfig

import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torch.distributed as dist


import math

import argparse
from tqdm import tqdm
import wandb
from logger import Logger
import matplotlib.pyplot as plt

def get_lr(now_step, all_step, lr):
    return lr / 10 + 0.5 * lr * ( 1 +math.cos(math.pi * now_step / all_step))

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

    parser = argparse.ArgumentParser("Pretrain Arguments")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=str)
    parser.add_argument("--lr", default=5e-7, type=float)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--pretrain_model_path", default="pretrain_model.pth", type=str)
    parser.add_argument("--data_jsonl_path",default=r"/home/kkyyxhll/Projects/PythonProjects/MiniKL/data/test_sft.jsonl",  type=str)
    parser.add_argument("--vocab_dict_path", default=r"/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json", type=str)
    parser.add_argument("--model_save_dir", default="saved_sft_model", type=str)
    parser.add_argument("--load_model_path", default="pretrain_model.pth", type=str)
    parser.add_argument("--wandb_entity", default="loukang", type=str)
    parser.add_argument("--wandb_project", default="test", type=str)
    args = parser.parse_args()

    logger = Logger(task_name="sft")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print(f"device: cpu, only support cuda.")
        exit(-1)

    if os.path.exists(args.model_save_dir):
        shutil.rmtree(args.model_save_dir)
    os.mkdir(args.model_save_dir)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"{device}:{local_rank}"

    print(f"device:{device}")

    # load wandb
    if local_rank == 0:
        print(f"args:{args}")
        if args.wandb:
            run = wandb.init(entity=args.wandb_entity,
                             project=args.wandb_project,)

    # load tokenzier
    tokenizer_config = TokenizerConfig(args.vocab_dict_path)
    tokenizer = BaseTokenizer(tokenizer_config)
    vocab_size = tokenizer.get_vocab_size()

    # load model
    model_config = MiniKLConfig(vocab_size=vocab_size, )
    model = MiniKLModel(model_config).to(device)
    if os.path.exists(args.pretrain_model_path):
        print(f"Loading Pretrain Model .......")
        print(f"Pretrain_model_path:{args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    scaler = torch.amp.GradScaler()

    dataset = SFTDataset(tokenizer, args.data_jsonl_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    per_epoch_step = len(dataloader)
    all_step = args.epochs * per_epoch_step
    all_losses = []
    for e in range(args.epochs):

        with tqdm(dataloader, unit="train") as pbar:
            for i, (x, y, masks) in enumerate(pbar):
                x, y, masks = x.to(device), y.to(device), masks.to(device)
                lr = get_lr(e*per_epoch_step+i, all_step, args.lr)
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad()
                # amp
                with torch.amp.autocast("cuda", dtype=torch.float16, ):
                    pred_y = model(x).transpose(-1, -2)
                    loss = torch.mean(criterion(pred_y, y) * masks)


                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                all_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.set_description(
                    f"device:{device} epoch:[{e + 1}|{args.epochs}] step:[{i + 1}|{per_epoch_step}] lr:[{lr}]")

                log = f"device:{device}  epoch:[{e + 1}|{args.epochs}], step:[{i + 1}|{per_epoch_step}], lr:[{lr}], loss:{loss.item():.4f}"
                logger.write(log)
                if args.wandb:
                    if dist.get_rank() == 0:
                        run.log({"epoch": e + 1,
                                 "step": i + 1,
                                 "loss": loss.item()})
                if (i+1) % 1000 == 0:
                    if dist.get_rank() == 0:
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            state_dict = model.module.state_dict()
                        else :
                            state_dict = model.state_dict()
                        model_save_path = os.path.join(args.model_save_dir, f"sft_model_{e*per_epoch_step+i}.pth")
                        torch.save(state_dict, model_save_path)
        pbar.update()


    if dist.get_rank() == 0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        model_save_path = os.path.join(args.model_save_dir, f"sft_model.pth")
        print(f"model saved {model_save_path}")
        torch.save(state_dict, model_save_path)

    x = [e for e in range(all_step)]
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




