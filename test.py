from models import MiniKLModel, MiniKLConfig
from tokenizer import TokenizerConfig, BaseTokenizer

import torch

import os

import argparse
import torch.nn.functional as F
import warnings

def apply_temperature(next_token_logits:torch.Tensor, temperature: float=0.993):
    scaled_logits = next_token_logits  / temperature
    return scaled_logits

def top_k(next_token_logits: torch.Tensor, k: int=4, ):
    topk_logits, topk_indices = torch.topk(next_token_logits, k)
    topk_probs = F.softmax(topk_logits, -1)
    sample = torch.multinomial(topk_probs, 1).item()
    num = topk_indices[0][sample].item()
    return num

def top_p(next_token_logits: torch.Tensor, p):
    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, -1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    flag = torch.where(cumsum_probs>p, 1, 0).squeeze()
    first_index = flag.nonzero(as_tuple=True)[0][0]
    logits = sorted_logits[0, :first_index+1]
    probs = F.softmax(logits, -1)
    sample = torch.multinomial(probs, 1).item()
    num = sorted_indices[0][sample].item()
    return num

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser("Test Args")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--vocab_dict_path", type=str, default=r'/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json')
    parser.add_argument("--model_path", type=str, default=r'/home/kkyyxhll/Projects/PythonProjects/MiniKL/saved_pretrain_model/pretrain_model_5999.pth')
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--sample_way", type=str, default="top_p")
    parser.add_argument("--prompt", type=str, default="<s>太阳在哪里")
    args = parser.parse_args()

    tokenizer_config = TokenizerConfig(mode="test", vocab_dict_path=args.vocab_dict_path, max_seq_len=args.max_seq_len)
    tokenizer = BaseTokenizer(tokenizer_config)

    vocab_size = tokenizer.get_vocab_size()

    model_config = MiniKLConfig(vocab_size=vocab_size)
    model = MiniKLModel(model_config).to(args.device)
    if os.path.exists(args.model_path):
        print("模型载入")
        model.load_state_dict(torch.load(args.model_path))

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    prompt = args.prompt
    resp = ""
    print(f"prompt:{prompt}")
    prompt = "<s>" + prompt
    while len(prompt) < 512:

        tokens = torch.tensor(tokenizer.tokenize(prompt)).to(args.device)

        output = model(tokens)

        next_token_logit = apply_temperature(output[:, -1, :], args.temperature)
        if args.sample_way == "top_k":
            num = top_k(next_token_logit, args.k)
        elif args.sample_way == "top_p":
            num = top_p(next_token_logit, args.p)

        token = tokenizer.decode([num])[0][0]
        if token == "</s>":
            break
        prompt += token
        resp += token
    print(f"response: {resp}")
