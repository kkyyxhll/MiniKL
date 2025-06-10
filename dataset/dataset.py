
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
    """
    DPO偏好数据集
    数据格式示例:
    {
        "prompt": "解释量子计算的基本概念",
        "chosen": "量子计算利用量子比特...",
        "rejected": "量子计算就是快的计算..."
    }
    """

    def __init__(self, tokenizer, data_jsonl_path):
        self.tokenizer = tokenizer
        self.dataset = self._init_dataset(data_jsonl_path)

    def _init_dataset(self, data_jsonl_path):
        print(f"开始载入DPO数据集, 路径: {data_jsonl_path}")
        start_time = time.time()
        dataset = []
        pad_id = self.tokenizer.vocab_dict["<pad>"]

        with open(data_jsonl_path, "r", encoding="utf-8") as f:
            with tqdm(f, desc="处理DPO数据", unit="行") as pbar:
                for line in pbar:
                    data = json.loads(line)
                    prompt = data["prompt"]
                    chosen = data["chosen"]
                    rejected = data["rejected"]

                    # 构建完整序列
                    chosen_seq = f"<|user|>{prompt}</s>\n<|assistant|>{chosen}</s>"
                    rejected_seq = f"<|user|>{prompt}</s>\n<|assistant|>{rejected}</s>"

                    # Tokenize序列
                    chosen_tokens = self.tokenizer.tokenize(chosen_seq)[0]
                    rejected_tokens = self.tokenizer.tokenize(rejected_seq)[0]

                    # 截断过长的序列
                    max_len = self.tokenizer.max_seq_len
                    chosen_tokens = chosen_tokens[:max_len]
                    rejected_tokens = rejected_tokens[:max_len]

                    # 创建padding mask (1=真实token, 0=padding)
                    chosen_mask = [1] * len(chosen_tokens)
                    rejected_mask = [1] * len(rejected_tokens)

                    # 填充到最大长度
                    chosen_tokens += [pad_id] * (max_len - len(chosen_tokens))
                    chosen_mask += [0] * (max_len - len(chosen_mask))

                    rejected_tokens += [pad_id] * (max_len - len(rejected_tokens))
                    rejected_mask += [0] * (max_len - len(rejected_mask))

                    dataset.append({
                        "chosen_tokens": chosen_tokens,
                        "rejected_tokens": rejected_tokens,
                        "chosen_mask": chosen_mask,
                        "rejected_mask": rejected_mask
                    })

        print(f"DPO数据集载入完成, 耗时: {time.time() - start_time:.2f}秒, 样本数: {len(dataset)}")
        return dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return (
            torch.tensor(item["chosen_tokens"], dtype=torch.long),
            torch.tensor(item["rejected_tokens"], dtype=torch.long),
            torch.tensor(item["chosen_mask"], dtype=torch.long),
            torch.tensor(item["rejected_mask"], dtype=torch.long)
        )

    def __len__(self):
        return len(self.dataset)

def test_sft():
    from tokenizer import BaseTokenizer, TokenizerConfig
    vocab_dict_path = "/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json"
    tokenizer = BaseTokenizer(TokenizerConfig(vocab_dict_path))
    dataset = SFTDataset(tokenizer, )
    print(dataset.__getitem__(0))

def pretrain():
    from tokenizer import BaseTokenizer, TokenizerConfig
    vocab_dict_path = "/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json"
    tokenizer = BaseTokenizer(TokenizerConfig(vocab_dict_path))
    text = "<s>鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。</s> <s>好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？</s>"
    output = tokenizer.pretrain_tokenize(text)


if __name__ == "__main__":
    test_sft()


