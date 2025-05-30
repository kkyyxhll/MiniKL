import json

import os

import shutil

import argparse

import time

def trans_minimind(minimind_path, out_dir):
    out_name = "minimind_data.jsonl"

    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        return out_path
    print(f"minimind格式的data开始载入，保存在{out_path}")

    count = 0
    temp = []
    start_time = time.time()
    with open(minimind_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)["text"].replace("<|im_start|>", "<s>").replace("<|im_end|>", "</s>")
            temp_dict = {"id": count, "content": data}
            temp.append(temp_dict)
            count += 1
    with open(out_path, "a", encoding="utf-8") as f:
        for temp_dict in temp:
            json.dump(temp_dict, f, ensure_ascii=False)
            f.write("\n")
    print(f"minimind格式的data转换完成，保存在{out_path}, 共{count}条, 耗时:{time.time() - start_time:.4f}s")
    return out_path

def parse_id(id):
    parts = id.split('|')
    book_seq = int(parts[0])
    chapter_seq = int(parts[1])
    return book_seq, chapter_seq

def write_data_jsonl(id_content_dict, out_data_jsonl_path):
    sorted_id = id_content_dict.keys()


    with open(out_data_jsonl_path, 'w', encoding='utf-8') as f:
        for id in sorted_id:
            temp_dict = {"id": id, "content": id_content_dict[id]}
            json.dump(temp_dict, f, ensure_ascii=False)
            f.write('\n')

def read_data_jsonl(jsonl_path):
    id_content_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            id_content_dict[data['id']] = data['content']
    return id_content_dict

def sort_data_jsonl(data_jsonl_path, out_data_jsonl_path):
    id_content_dict = read_data_jsonl(data_jsonl_path)
    write_data_jsonl(id_content_dict, out_data_jsonl_path)

def merge_data_jsonl(first_data_jsonl_path, second_data_jsonl_path, out_data_jsonl_path):
    id_content_dict = read_data_jsonl(first_data_jsonl_path)
    with open(second_data_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['id'] not in id_content_dict:
                id_content_dict[data['id']] = data['content']
    write_data_jsonl(id_content_dict, out_data_jsonl_path)

def spilt_data_jsonl(in_data_jsonl_path, out_dir_path, n):

    id_content_dict = read_data_jsonl(in_data_jsonl_path)
    keys = list(id_content_dict.keys())
    length = len(keys)
    step = length // n
    start_index = 0
    for i in range(n):
        end_index = step + start_index if step + start_index < length else length
        temp_dict = {k:id_content_dict[k] for k in keys[start_index:end_index]}

        start_index = end_index
        out_data_jsonl_path = os.path.join(out_dir_path, f"data{i}.jsonl")
        write_data_jsonl(temp_dict, out_data_jsonl_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimind", action="store_true")
    parser.add_argument("--in_dir", default="/home/kkyyxhll/Projects/PythonProjects/MiniKL/data/pretrain_hq.jsonl")
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--n", default=16)
    args = parser.parse_args()
    in_data_jsonl_path = args.in_dir


    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if args.minimind:
        print(f"{args.in_dir}为minimind格式")
        args.in_dir = trans_minimind(args.in_dir, args.out_dir)

    spilt_data_jsonl(args.in_dir, args.out_dir, args.n)
