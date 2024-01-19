"""
Sample from the trained model with PyTorch
"""
import os
import json
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import numpy as np

# def compute_bleu(labels, preds, weights=None):
#     from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#     weights = weights or (0.25, 0.25, 0.25, 0.25)
#     return np.mean([sentence_bleu(references=[label],
#                                   hypothesis=pred,
#                                   smoothing_function=SmoothingFunction().method1,
#                                   weights=weights) for label, pred in zip(labels, preds)])
# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 100 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
compile = False # use PyTorch 2.0 to compile the model to be faster
#exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
# model4 
max_seq_len = 512
dim = 1024
n_layers = 16
n_heads = 16

# max_seq_len = 512
# dim = 1024
# n_layers = 12
# n_heads = 8

# max_seq_len = 1024
# dim = 1024
# n_layers = 12
# n_heads = 8
multiple_of = 32
dropout = 0.0 
model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=64793,#64793,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )  # s
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast()

# init from a model saved in a specific directory
ckpt_path = 'out/pretrain/model1_sft_epoch_1.pth'
state_dict = torch.load(ckpt_path, map_location=device)
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
#
# data = []
# with open('./test_data/test.json','r') as f:
#     for line in f:
#         data.append(json.loads(line))

#如果有标准答案，可以填到target里面，打开最后几行的注释，计算bleu分数。
#如果随便测试测试，那就只填你希望问的问题到question里面就可以。
data = [
    {"question": "北京大学", "target": ""},
]

ans_lst=[]
target_lst=[]
for p in data:
    # run generation
    prompt=p['question']
    x=tokenizer.encode(prompt,add_special_tokens=False)+[tokenizer.special_tokens['<bos>']]
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
    target = p['target']
    target_lst.append(target)
    with torch.no_grad():
        with ctx:
            y = model.generate(x, 2, max_new_tokens, temperature=temperature, top_k=top_k)
            #
            answer=tokenizer.decode(y[0].tolist())
            answer=answer.replace(prompt,'')
            ans_lst.append(answer)
            print('[prompt]:',prompt)
            print('[answer]:',answer)
            print('---------------')
#
# import jieba
# target_lst=[jieba.lcut(result.lower()) for result in target_lst]
# preds_lst=[jieba.lcut(result.lower()) for result in ans_lst]
# scores = compute_bleu(preds_lst, target_lst)
# print(scores)