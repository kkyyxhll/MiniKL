
import json

import torch
from torch.utils.data import Dataset

import time
from tqdm import tqdm
class PretrainDataset(Dataset):
    def __init__(self,tokenizer, data_jsonl_path):
        self.tokenizer = tokenizer
        self.dataset = self._init_dataset(data_jsonl_path)

    def _init_dataset(self, data_jsonl_path):
        print(f"开始载入数据集,path:{data_jsonl_path}")
        start_time = time.time()
        dataset = []
        with open(data_jsonl_path, "r", encoding="utf-8") as f:
            with tqdm(f, desc="处理数据行", unit="行") as pbar:
                for line in pbar:
                    content = json.loads(line)["content"]
                    if type(content) != list:
                        #[]
                        # dict{"input_ids":list(tuple...), "padding_masks":list(tuple...)}
                        chunks= self.tokenizer.pretrain_tokenize(content)

                    for i in range(len(chunks["input_ids"])):
                        temp_dict = {"input_ids": chunks["input_ids"][i], "padding_masks": chunks["padding_masks"][i]}
                        dataset.append(temp_dict)

        print(f"数据集载入完成,time:{time.time()-start_time}")
        return dataset

    def __getitem__(self, index):

        content = self.dataset[index]

        input_ids = content["input_ids"]
        padding_masks = content["padding_masks"]
        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_padding_masks = torch.tensor(padding_masks, dtype=torch.long)
        return all_input_ids[0:-1], all_input_ids[1:], all_padding_masks[1:]

    def __len__(self):
        return len(self.dataset)

class SFTDataset(Dataset):
    """
    {"conversations": [
    {"role": "user", "content": "请用一句话介绍阿里巴巴集团。"},
    {"role": "assistant", "content": "阿里巴巴集团是一家总部位于中国杭州的全球领先的电子商务和科技公司。"}
    ]}
    alpaca
    <|user|>user_content</s>
    <|assistant|>assistant_content</s>
    [(text, mask), (text, mask), (text, mask)]

    """

    def __init__(self, tokenizer, data_jsonl_path="/home/kkyyxhll/Projects/PythonProjects/MiniKL/data/test_sft.jsonl"):
        prompt = ("<|user|>{user_content}</s>\n"
                  "<|assistant|>{assistant_content}</s>")
        self.tokenizer = tokenizer
        self.tokens_masks_list = self._init_tokens_masks_list(data_jsonl_path)

    def _init_tokens_masks_list(self, data_jsonl_path):
        print("loading SFTDataset .......")
        start_time = time.time()
        tokens_masks_list = []
        with open(data_jsonl_path, "r", encoding="utf-8") as f:
            with tqdm(f, desc="处理数据行", unit="行") as pbar:
                for line in pbar:
                    conversations = json.loads(line)["conversations"]
                    prompt = ""
                    for conversation in conversations:
                        role = conversation["role"]
                        content = conversation["content"]
                        prompt += f"<|{role}|>{content}</s>\n"
                    tokens = self.tokenizer.tokenize(prompt)[0]
                    if len(tokens) >= self.tokenizer.max_seq_len:
                        continue
                    masks = []
                    flag_assistant = 0
                    assistant_start = self.tokenizer.vocab_dict["<|assistant|>"]
                    pad = self.tokenizer.vocab_dict["<pad>"]
                    end = self.tokenizer.vocab_dict["</s>"]
                    for token in tokens:
                        if flag_assistant == 0:
                            masks.append(0)
                        if flag_assistant == 1:
                            masks.append(1)
                            if token == end:
                                flag_assistant = 0
                        if token == assistant_start:
                            flag_assistant = 1

                    while len(tokens) < self.tokenizer.max_seq_len:
                        tokens.append(pad)
                        masks.append(0)

                    tokens_masks_list.append((tokens, masks))
        print(f"SFTDataset had loaded, spend_time:{time.time()-start_time:.4f}")
        return tokens_masks_list
    def __getitem__(self, idx):
        tokens, masks = self.tokens_masks_list[idx]
        tokens, masks = torch.tensor(tokens, dtype=torch.long), torch.tensor(masks, dtype=torch.long)
        return tokens[0:-1], tokens[1:], masks[1:]
    def __len__(self):
        return len(self.tokens_masks_list)

class DPODataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass 


if __name__ == "__main__":

    from tokenizer import BaseTokenizer, TokenizerConfig
    vocab_dict_path = "/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json"
    tokenizer = BaseTokenizer(TokenizerConfig(vocab_dict_path))
    dataset = SFTDataset(tokenizer,)

