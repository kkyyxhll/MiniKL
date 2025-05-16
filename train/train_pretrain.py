import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse

from logger import Logger
from models import MiniKLModel, MiniKLConfig
from dataset import PretrainDataset
from tokenizer import BaseTokenizer, TokenizerConfig
from tqdm import tqdm

import math

import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser("MiniKL Pretrain Args")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--vocab_dict_path", type=str, default=r"/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json")
    parser.add_argument("--data_jsonl_path", type=str, default=r"/home/kkyyxhll/Projects/PythonProjects/MiniKL/data/out/data0.jsonl")

    args = parser.parse_args()

    logger = Logger(task_name="pretrain", )

    print(f"device: {args.device}")

    tokenizer_config = TokenizerConfig(mode="test", vocab_dict_path=args.vocab_dict_path, max_seq_len=args.max_seq_len)
    tokenizer = BaseTokenizer(tokenizer_config)

    vocab_size = tokenizer.get_vocab_size()

    pretrain_dataset = PretrainDataset(tokenizer, args.data_jsonl_path)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True)

    model_config = MiniKLConfig(vocab_size=vocab_size,)
    model = MiniKLModel(model_config).to(args.device)

    optimizer = optim.AdamW(model.parameters(),)
    criterion = nn.CrossEntropyLoss(reduction="none")



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
                pred_y = model(x)

                pred_y = pred_y.transpose(-1, -2)

                loss_masked = criterion(pred_y, y) * padding_masks

                loss = torch.mean(loss_masked)

                loss.backward()
                optimizer.step()

                all_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.set_description(f"epoch:[{e+1}|{args.epochs}] step:[{i+1}|{per_epoch_steps}] lr:[{lr:.4f}]")

                log = f"epoch:[{e + 1}|{args.epochs}], step:[{i+1}|{per_epoch_steps}], lr:[{lr:.4f}], loss:{loss.item():.4f}"
                logger.write(log)

                if (i+1) % 100 == 0:
                    torch.save(model.state_dict(), "pretrain_model.pth")
        torch.save(model.state_dict(), "pretrain_model.pth")

        pbar.update()

    x = [e for e in range(all_steps)]
    all_losses = exp_moving_average(all_losses)
    plt.figure()
    plt.plot(x, all_losses)
    plt.title("loss | epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

