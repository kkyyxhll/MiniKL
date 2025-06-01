import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ["all_proxy"]=""
os.environ["ALL_PROXY"]=""

import gradio as gr


import torch

import torch.nn.functional as F

from models import MiniKLModel, MiniKLConfig
from tokenizer import TokenizerConfig, BaseTokenizer


"""
https://blog.csdn.net/bsy1111/article/details/133245312
"""

def apply_temperature(next_token_logits:torch.Tensor, temperature=0.85):
    scaled_logits = next_token_logits  / temperature
    return scaled_logits

def top_k(next_token_logits: torch.Tensor, k: int=8, ):
    topk_logits, topk_indices = torch.topk(next_token_logits, k)
    topk_probs = F.softmax(topk_logits, -1)
    sample = torch.multinomial(topk_probs, 1).item()
    num = topk_indices[0][sample].item()
    return num

def top_p(next_token_logits: torch.Tensor, p=0.85):
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

def top_k_and_p(next_token_logits: torch.Tensor, k, p):
    topk_logits, topk_indices = torch.topk(next_token_logits, k)
    num = top_p(topk_logits, p)
    return topk_indices[0][num].item()


def init_tokenizer(vocab_dict_path):
    tokenizer_config = TokenizerConfig(vocab_dict_path)
    return BaseTokenizer(tokenizer_config)

def init_model(model_path, vocab_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    model_config = MiniKLConfig(vocab_size=tokenizer.get_vocab_size())
    model = MiniKLModel(model_config).to(device)
    if os.path.exists(model_path):
        print("模型载入")
        model.load_state_dict(torch.load(model_path))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model

def get_resp(text, model, tokenizer, temperature, k, p):

    now_qa = text
    flag = True
    resp = ""
    print("now_qa:", now_qa)

    while True:
        tokens = torch.tensor(tokenizer.tokenize(now_qa)[0]).to(device)
        output = model(tokens)
        next_token_logit = apply_temperature(output[:, -1, :], temperature)
        num = top_k_and_p(next_token_logit, k, p)
        token = tokenizer.decode([num])[0][0]
        if token == "</s>":
            break
        now_qa += token
        resp += token
        if len(now_qa) >= 512:
            flag = False
            break
    print(resp)
    return resp, flag


def get_resp_gr(message, histories, temperature, top_k, top_p):
    user_input = ""
    if histories is not None:
        for history in histories:
            role = history["role"]
            content = history["content"]
            user_input += f"<|{role}|>{content}</s>\n"
    if len(histories) == 4:
        gr.Info("因训练有限，目前只支持两轮对话。")
        histories = [histories[2] ,histories[3]]

    user_input += f"<|user|>{message}</s>\n<|assistant|>"

    resp, flag = get_resp(user_input, model, tokenizer, temperature, top_k, top_p)
    if not flag:
        gr.Info("提示: 对话历史已自动清空，(达到最大长度限制)。")
        return "", []

    new_qa = [{"role":"user", "content": message}, {"role":"assistant", "content":resp}]
    histories.extend(new_qa)

    return "", histories

def clear_chat():
    """清空聊天历史"""
    return [], ""

tokenizer = init_tokenizer("gui/vocab_dict.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = init_model("gui/sft_model.pth", tokenizer.get_vocab_size())

if __name__ == "__main__":

    temperature = 0.85
    k = 10
    p = 0.85

    qa = ""
    # while True:
    #     text = apply_prompt(input())
    #     resp, flag = get_resp(qa+text, model, tokenizer, temperature, k, p)
    #     if not flag:
    #         break
    #     qa = qa + text + resp
    #     print(qa)

    with gr.Blocks(title="MiniKL ChatBot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# MiniKL ChatBot")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="对话历史", type="messages")
                message = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入您的问题...",
                    lines=3
                )
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")
            with gr.Column(scale=1):
                gr.Markdown("### generate parameters")

                gr_temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.9,
                    step=0.05,
                    label="temperature"
                )
                gr_top_k = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=45,
                    step=1,
                    label="top_k"
                )
                gr_top_p = gr.Slider(
                    minimum=0.5,
                    maximum=0.99,
                    value=0.9,
                    step=0.01,
                    label="top_p"
                )
                samples = gr.Markdown("### question samples")
                q = ["今天天气怎么样？",
                     "请生成一段描述春天的短文。",
                     "请写一篇关于太空探索的文章，主题是探险精神。",
                     "你是谁？",
                     "写一篇关于人工智能发展趋势的文章。",
                     "告诉我苹果在2020年的全球销售额。",
                     "写一篇关于环境保护的文章。",
                     "介绍一下《三体》这本书。",
                     "介绍一道菜品以及详细做法。"]

                df = pd.DataFrame({
                    "示例问题": q
                })
                gr.DataFrame(
                    df,
                    headers=["示例问题"],
                    interactive=True,
                    show_label=False
                )
        submit_btn.click(
            get_resp_gr,
            inputs=[message, chatbot, gr_temperature, gr_top_k, gr_top_p],
            outputs=[message, chatbot]
         )
        clear_btn.click(
            clear_chat,
            inputs=[],
            outputs=[chatbot, message]
        )
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)