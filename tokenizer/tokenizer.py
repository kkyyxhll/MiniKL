import os

import time

import json

from collections import defaultdict
class TokenizerConfig:
    def __init__(self,
                 mode = "train",
                 vocab_dict_path:str = None,  #预训练的vocab_dict_path[json文件] key:vocab, value:seq
                 data_jsonl_path: str = None, #训练数据[jsonl]文件
                 out_dir: str = None,         #训练过程输出的checkpoint 下面包括vocab_dict(json), vocab_freq_dict(json), log(txt), data(jsonl)
                 vocab_size: int = 20000,
                 special_tokens=None,
                 savepoint_epoch:int=1,
                 max_seq_len:int  = 512,
                 ):

        self.mode = mode
        self.vocab_dict_path = vocab_dict_path
        self.data_jsonl_path = data_jsonl_path
        self.out_dir = out_dir
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.savepoint_epoch = savepoint_epoch
        self.max_seq_len = max_seq_len

        if special_tokens is None:
            self.special_tokens = {"<bos>": "<s>",
                                   "<eos>": "</s>",
                                   "<unk>": "<unk>",
                                   "<pad>": "<pad>", }

        print(f"[mode is {mode}]")

        if mode == "test":
            if vocab_dict_path is None:
                raise ValueError("vocab_dict_path is required")

        if mode == "train":
            if out_dir is None:
                if data_jsonl_path is None:
                    raise ValueError("while out_dir is None, data_jsonl_path is required")
                self.data_jsonl_path = data_jsonl_path
                self.out_dir = os.path.join(os.path.dirname("__file__"), "out_dir")
            else :
                self.out_dir = out_dir


class BaseTokenizer:
    def __init__(self, config:TokenizerConfig):
        self.vocab_dict = {}
        self.decode_vocab_dict = {}
        if config.vocab_dict_path is not None:
            self.vocab_dict_path = config.vocab_dict_path
            self._load_vocab_dict()

        self.data_jsonl_path = config.data_jsonl_path
        self.out_dir = config.out_dir
        self.vocab_size = config.vocab_size
        self.now_vocab_size = 0
        self.vocab_freq_dict = defaultdict(int)
        self.id_content_dict = {}
        self.pair_freq_dict = defaultdict(int)
        self.savepoint_epoch = config.savepoint_epoch
        self.special_tokens = config.special_tokens
        self.max_seq_len = config.max_seq_len

    def train(self,):
        if self._train_init_out_dir(): # 从out加载
            print("从out_dir中预先加载.......")
            self._train_load_data_jsonl_path(self.out_data_jsonl_path)
            self._load_vocab_dict()

        else :
            print("开始训练......")
            self._train_load_data_jsonl_path(self.data_jsonl_path)
        self._train_load_vocab_freq_dict()

        while self.now_vocab_size < self.vocab_size:
            self.now_vocab_size += 1
            start_time = time.time()
            flag, max_pair_info = self._train_find_max_pair()
            log_line = f"[轮次：{self.now_vocab_size}] | [merge_pair:{max_pair_info[0]}] | [find_max_time:{time.time() - start_time}]"
            print(log_line, end=" ")
            start_time = time.time()
            if not flag:
                break
            delete_token_info = self._train_update(max_pair_info)
            add_log_line = f"| [delete_token:{delete_token_info}] | [update_time:{time.time() - start_time}]"
            print(add_log_line)
            log_line += add_log_line
            self._train_log(log_line)
            if self.now_vocab_size % self.savepoint_epoch == 0:
                self._train_save_savepoint()

    def get_vocab_size(self):
        return len(self.vocab_dict)

    def pretrain_tokenize(self, texts):
        max_seq_len = self.max_seq_len
        seqs, max_len = self.tokenize(texts)
        """
            [[a, b, c], [], [], ]
            [[<bos>, a, b, c, <eos>, <pad>, <pad>, <pad>]]
        """
        # 文本长度
        #length = max_len+2 if max_len+2 < max_seq_len else max_seq_len
        length = max_seq_len
        chunks = []
        masks = []
        for seq in seqs:
            temp_chunks, temp_masks = self._chunk(seq, length)
            chunks += temp_chunks
            masks += temp_masks
        return {"input_ids": chunks, "padding_masks": masks}

    def _chunk(self, seq, length):
        length = length - 2 # <bos> <eos>
        chunks = []
        masks = []
        start_index = 0

        bos = self.special_tokens["<bos>"]
        eos = self.special_tokens["<eos>"]
        pad = self.special_tokens["<pad>"]
        bos_index = self.vocab_dict[bos]
        eos_index = self.vocab_dict[eos]
        pad_index = self.vocab_dict[pad]

        while True:
            end_index, flag = (start_index + length, False) if start_index + length < len(seq) else (len(seq), True)
            temp_pads = [pad_index for _ in range(length - len(seq) + start_index)]
            temp_chunks = tuple([bos_index] + seq[start_index:end_index] + [eos_index] + temp_pads)
            chunks.append(temp_chunks)
            temp_masks = tuple([1] * (len(chunks[-1]) - len(temp_pads)) + [0] * len(temp_pads))
            masks.append(temp_masks)
            start_index = end_index
            if flag:
                break
        return chunks, masks

    def get_special_tokens(self, ):
        return self.special_tokens

    def tokenize(self, texts:str or list[str]):
        max_token_length = len(self.decode_vocab_dict[0])
        if type(texts) == str:
            texts = [texts]
        seqs = []
        max_len = 0
        for text in texts:
            i = 0
            seq = []
            temp_length = 0
            while i < len(text):
                flag = False
                for length in range(max_token_length, 0, -1):
                    if i + length - 1< len(text):
                        chars = text[i:i+length]
                        if chars in self.vocab_dict:
                            seq.append(self.vocab_dict[chars])
                            flag = True
                            i = i + length
                            temp_length += 1
                            break
                if not flag:
                    seq.append(self.vocab_dict["<unk>"])
                    i += 1
                    temp_length += 1

            seqs.append(seq)
            if max_len < temp_length:
                max_len = temp_length
        return seqs, max_len

    def decode(self, seqs:list or list[list]):
        "index 2 token"
        if type(seqs[0]) == int:
            seqs = [seqs]
        texts = []
        for seq in seqs:
            text = []
            for num in seq:
                text.append(self.decode_vocab_dict[num])
            texts.append(text)
        return texts

    def _add_special_tokens(self):
        print(self.special_tokens)
        for special_token in self.special_tokens:
            if special_token not in self.vocab_dict:
                self.vocab_freq_dict[self.special_tokens[special_token]] = 0

    def _load_vocab_dict(self, ):
        start_time = time.time()
        with open(self.vocab_dict_path, "r", encoding="utf-8") as f:
            data = json.load(f,)
        self.vocab_dict = data
        for key, value in self.vocab_dict.items():
            self.decode_vocab_dict[value] = key
        print(f"[test] vocab_dict加载完成, time:{time.time() - start_time}")

    def _train_init_out_dir(self):
        print(f"[out_dir:{self.out_dir}]")
        self.out_log_path = os.path.join(self.out_dir, "log.txt")
        self.vocab_dict_path = os.path.join(self.out_dir, "vocab_dict.json")
        self.out_vocab_freq_dict_path = os.path.join(self.out_dir, "vocab_freq_dict.json")
        self.out_data_jsonl_path = os.path.join(self.out_dir, "data.jsonl")

        if os.path.exists(self.out_dir):
            for root, dirs, files in os.walk(self.out_dir):
                if not files:
                    return False
            return True
        else :
            os.mkdir(self.out_dir)
            return False

    def _train_log(self, log_line):
        with open(self.out_log_path, "a", encoding="utf-8") as f:
            f.write(log_line)
            f.write("\n")

    def _train_save_savepoint(self):
        self._add_special_tokens()
        self.vocab_dict = {}
        count = 0
        vocabs = self.vocab_freq_dict.keys()
        vocab_list = sorted(vocabs, key=lambda k: -len(k))
        for vocab in vocab_list:
            self.vocab_dict[vocab] = count
            count += 1

        with open(self.vocab_dict_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab_dict, f, ensure_ascii=False)

        with open(self.out_vocab_freq_dict_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab_freq_dict, f, ensure_ascii=False)

        with open(self.out_data_jsonl_path, "w", encoding="utf-8") as f:
            for id in self.id_content_dict.keys():
                temp_dict = {"id":id, "content":self.id_content_dict[id]}
                json.dump(temp_dict, f, ensure_ascii=False)
                f.write("\n")

    def _train_load_data_jsonl_path(self, data_jsonl_path:str):
        """
        目的：加载data.jsonl文件
        :return: dict{id:content}
        for example:
        data.jsonl:
            {"id": "1", content:"first"},
            {"id": "2", content:"second"},
            {"id": "3", content:"third"}
        return:
            {"1":"first",
            "2":"second",
            "3":"third"}
        """
        start_time = time.time()
        with open(data_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                json_data = json.loads(line)
                self.id_content_dict[json_data["id"]] = list(json_data["content"])
        print(f"data_jsonl已提取到id_content_dict, time:{time.time() - start_time}")
    def _train_load_vocab_freq_dict(self):
        """
        目的： 初始化词表，token细粒度初始化为字。
        input: id_content_dict
            sample: {"1":"first",
                    "2":"second",
                    "3":"third"}
        output: vocab_freq_dict
            sample: {"f":1, "i":2, "r":2, "s":2, "t":2, ...}
        """
        start_time = time.time()
        for id in self.id_content_dict.keys():
            content = self.id_content_dict[id]
            for token in content:
                self.vocab_freq_dict[token] += 1
        print(f"vocab_freq_dict加载完成 time:{time.time() - start_time}")

    def _train_find_max_pair(self) :
        max_pair_freq = 1
        max_pair = None
        flag = False

        self.pair_freq_dict = defaultdict(int)
        for id in self.id_content_dict.keys():
            content = self.id_content_dict[id]
            for i in range(0, len(content)-1):
                pair = (content[i], content[i+1])
                self.pair_freq_dict[pair] += 1

                if self.pair_freq_dict[pair] > max_pair_freq:
                    max_pair = pair
                    max_pair_freq = self.pair_freq_dict[pair]
                    flag = True

        return flag, (max_pair, max_pair_freq)

    def _train_update(self, max_pair_info):
        max_pair, max_pair_freq = max_pair_info
        delete_info = []
        for id in self.id_content_dict.keys():
            content = self.id_content_dict[id]
            temp_content = []
            i = 0
            while i < len(content):

                if i == len(content) - 1 or (content[i], content[i+1]) != max_pair:
                    temp_content.append(content[i])
                    i += 1
                else :
                    temp_content.append(content[i] + content[i+1])
                    self.vocab_freq_dict[content[i]] -= 1
                    self.vocab_freq_dict[content[i+1]] -= 1
                    self.vocab_freq_dict[content[i]+content[i+1]] += 1
                    if self.vocab_freq_dict[content[i]] == 0:
                        del self.vocab_freq_dict[content[i]]
                        delete_info.append(content[i])
                    if self.vocab_freq_dict[content[i+1]] == 0:
                        del self.vocab_freq_dict[content[i+1]]
                        delete_info.append(content[i+1])
                    i += 2
            self.id_content_dict[id] = temp_content

        return delete_info

if __name__ == "__main__":
   data_jsonl_path = "train.jsonl"
   vocab_size = 20000
   tokenizer = BaseTokenizer(config=TokenizerConfig(mode="test", vocab_dict_path="out_dir/vocab_dict.json"))
   output = tokenizer.pretrain_tokenize(["你好香啊"*1000,
                                        "你怎么这么香啊啊啊啊"])
   print(output)
   input_ids = output["input_ids"]
   output = tokenizer.decode(input_ids)
   print(output[-1])
