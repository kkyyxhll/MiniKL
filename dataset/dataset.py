import os

import json

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import BaseTokenizer, TokenizerConfig

import time
import random

class PretrainDataset(Dataset):
    def __init__(self,tokenizer, data_jsonl_path):
        self.tokenizer = tokenizer
        self.dataset = self._init_dataset(data_jsonl_path)


    def _init_dataset(self, data_jsonl_path):
        print(f"开始载入数据集,path:{data_jsonl_path}")
        start_time = time.time()
        dataset = []
        with open(data_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
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
        return all_input_ids[0:-1], all_input_ids[1:], all_padding_masks[0:-1]

    def __len__(self):

        return len(self.dataset)

    # def sample(self, content_size):
    #     contents = random.sample(self.dataset, content_size)
    #     all_input_ids = []
    #     all_padding_masks = []
    #     for content in contents:
    #         all_input_ids += content["input_ids"]
    #         all_padding_masks += content["padding_masks"]
    #
    #     return {"input_ids": all_input_ids, "padding_masks": all_padding_masks}



if __name__ == '__main__':
    data_jsonl_path = r"C:\Users\kkyyxhll\PycharmProjects\PythonProject\MiniKL\tokenizer\data.jsonl"
    vocab_dict_json_path = r"C:\Users\kkyyxhll\PycharmProjects\PythonProject\MiniKL\tokenizer\out_dir\vocab_dict.json"
    dataset = PretrainDataset(data_jsonl_path, vocab_dict_json_path)

    print(dataset.__getitem__(0))
