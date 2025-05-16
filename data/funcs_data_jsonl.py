import json

import os

import shutil

def parse_id(id):
    parts = id.split('|')
    book_seq = int(parts[0])
    chapter_seq = int(parts[1])
    return book_seq, chapter_seq

def write_data_jsonl(id_content_dict, out_data_jsonl_path):
    sorted_id = sorted(id_content_dict.keys(), key=parse_id)
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
    if os.path.exists(out_dir_path):
        shutil.rmtree(out_dir_path)
    os.makedirs(out_dir_path)
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
    in_data_jsonl_path = "../tokenizer/train.jsonl"
    out_dir_path = "out"
    n = 64
    spilt_data_jsonl(in_data_jsonl_path, out_dir_path, n)
