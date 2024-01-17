import re
import os
import matplotlib.pyplot as plt

# 定义存储loss、lr和iter的列表
loss_list = []
lr_list = []
iter_list = []

# 读取文本文件
with open("./out/Llama2-Chinese-92M-v1/log.log", "r") as f:
    file = f.readlines()
for line in file:
    # 使用正则表达式匹配出loss、lr和iter的值
    match = re.search(r"loss:([\d.]+) lr:([\d.]+)", line)
    if match:
        loss = float(match.group(1))
        lr = float(match.group(2))
        loss_list.append(loss)
        lr_list.append(lr)

        # 提取iter值
        iter_match = re.search(r"\((\d+)/\d+\)", line)
        if iter_match:
            iter_value = int(iter_match.group(1))
            iter_list.append(iter_value)
#
batch=24
gpus=4
max_length=512
iter_list=[i*batch*gpus*max_length for i in iter_list]

loss_list1 = []
lr_list1 = []
iter_list1 = []

# 读取文本文件
with open("./out/Llama2-Chinese-92M-v2/log.log", "r") as f:
    file = f.readlines()
for line in file:
    # 使用正则表达式匹配出loss、lr和iter的值
    match = re.search(r"loss:([\d.]+) lr:([\d.]+)", line)
    if match:
        loss = float(match.group(1))
        lr = float(match.group(2))
        loss_list1.append(loss)
        lr_list1.append(lr)

        # 提取iter值
        iter_match = re.search(r"\((\d+)/\d+\)", line)
        if iter_match:
            iter_value = int(iter_match.group(1))
            iter_list1.append(iter_value)
#
batch=32
gpus=4
max_length=512
iter_list1=[i*batch*gpus*max_length for i in iter_list1]

loss_list2 = []
lr_list2 = []
iter_list2 = []

# 读取文本文件
with open("./out/Llama2-Chinese-218M-v1/log.log", "r") as f:
    file = f.readlines()
for f in file:
    for line in f.split('Epoch:[0/1]')[1:]:
        # 使用正则表达式匹配出loss、lr和iter的值
        match = re.search(r"loss:([\d.]+) lr:([\d.]+)", line)
        if match:
            loss = float(match.group(1))
            lr = float(match.group(2))
            loss_list2.append(loss)
            lr_list2.append(lr)

            # 提取iter值
            iter_match = re.search(r"\((\d+)/\d+\)", line)
            if iter_match:
                iter_value = int(iter_match.group(1))
                iter_list2.append(iter_value)
#
batch=10
gpus=4
max_length=1024
iter_list2=[i*batch*gpus*max_length for i in iter_list2]

print(len(iter_list), len(iter_list1), len(iter_list2))

# 绘制loss随着iter的变化趋势
curve_name = ['Llama2-Chinese-92M-v1', 'Llama2-Chinese-92M-v2', 'Llama2-Chinese-218M-v1']
plt.figure(figsize=(10, 6), dpi=200) # 设置画布大小为10x6
for i, (x, y) in enumerate([(iter_list, loss_list), (iter_list1, loss_list1), (iter_list2, loss_list2)], start=1):
    plt.plot(x, y, label=f'{curve_name[i-1]}') # 绘制第i条曲线

plt.xlabel('Tokens')
plt.ylabel('Loss')
plt.title('Loss vs Tokens')
plt.legend()
plt.savefig('loss_tokens.png', bbox_inches='tight')