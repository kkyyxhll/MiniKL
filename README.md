# MiniKL
`tokenizer`, `model`, `data`都由`PyTorch`从零实现，不包含其他库，如`transformers`,`deepspeed`。  
支持混合精度训练、 分布式数据并行。   

## 预训练脚本
预训练需要提供词表(JSON文件)和预训练数据集(JSONL文件)。
提供测试的放在 https://www.modelscope.cn/datasets/kkyyxhll/MiniKL-dataset/files 中。
- vocab_dict.json 为词表文件
- data0.jsonl 为mini版本的预训练数据集
这两个参数需要在命令行的 `vocab_dict_path`和 `data_jsonl_path`中显式给出。

下面假设词表绝对路径为 `/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json`
预训练数据集绝对路径为 `/home/kkyyxhll/Projects/PythonProjects/MiniKL/data/out/data0.jsonl`
实际替换对应的词表路径和预训练数据集路径。
单机单卡  
```commandline
 torchrun --nnodes=1 --nproc-per-node=1 train/train_pretrain.py --vocab_dict_path="/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json"
 --data_jsonl_path=/home/kkyyxhll/Projects/PythonProjects/MiniKL/data/out/data0.jsonl
```
如果要使用`wandb`监控，命令例子如下，具体参见 https://wandb.ai/site :  
```commandline
torchrun --nnodes=1 --nproc-per-node=1 train/train_pretrain.py --use_wandb --wandb_enti
ty=loukang wandb_project=test  --vocab_dict_path="/home/kkyyxhll/Projects/PythonProjects/MiniKL/tokenizer/out_dir/vocab_dict.json"
 --data_jsonl_path=/home/kkyyxhll/Projects/PythonProjects/MiniKL/data/out/data0.jsonl
```
`wandb_entity`和`wandb_project`对应
```commandline
run = wandb.init(entity=args.wandb_entity,
                         # Set the wandb project where this run will be logged.
                         project=args.project,
                         # Track hyperparameters and run metadata.
                         config={
                             "learning_rate": args.lr,
                             "architecture": "MiniKL Decoder",
                             "dataset": "books",
                             "epochs": args.epochs,
                             "batch_size": args.batch_size
                         }, )
```
单机多卡， 同单机单卡，只是`nproc-per-node` 不同。
```commandline
 torchrun --nnodes=1 --nproc-per-node=n train/train_pretrain.py 
```