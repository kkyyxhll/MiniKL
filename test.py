from models import MiniKLModel, MiniKLConfig
from tokenizer import TokenizerConfig, BaseTokenizer

import torch

import os

import argparse

import warnings
if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser("Test Args")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--vocab_dict_path", type=str, default=r'/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json')
    parser.add_argument("--model_path", type=str, default=r'/home/kkyyxhll/Projects/PythonProjects/MiniKL/pretrain_model.pth')
    parser.add_argument("--prompt", type=str, default="谈谈大数据对未来的影响。")
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
    prompt = args.prompt
    resp = ""
    print(f"prompt:{prompt}")
    prompt = "<s>" + prompt
    for i in range(0, 1000):

        tokens = torch.tensor(tokenizer.tokenize(prompt)).to(args.device)

        output = model(tokens)

        num = torch.argmax(output[:, -1, :]).item()

        token = tokenizer.decode([num])[0][0]
        if token == "</s>":
            break
        prompt += token
        resp += token
    print(f"response: {resp}")
