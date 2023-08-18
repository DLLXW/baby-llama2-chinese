

## 介绍

本篇文章中，作者将使用自己的3090单卡在两天内从零开始预训练，并结合SFT（Supervised Fine-tuning）方法，打造出一个迷你版的中文Llama2模型，该模型将具备基础的中文问答能力。为了方便大家理解和应用，这里作者还将提供完整的预训练和SFT所需要的代码和数据、训练参数配置等。大家可以根据自己的需求和实际情况进行调整和应用，以实现更好的中文问答效果。

完整代码地址：https://github.com/DLLXW/baby-llama2-chinese

其中llama2模型结构和训练pipeline部分参考了[llama2.c](https://github.com/karpathy/llama2.c)，该repo的作者也是大名鼎鼎的特斯拉 AI 负责人 Andrej Karpathy。

## 引入中文分词器和词表

因为在llama官方所提供的词表中，中文的部分只有700个，这也是llama中文能力聊胜于无的原因。为了训练自己的中文LLaMa，这里将引入新的中文分词器。为了方便，这里选择采用ChatGLM2的分词器，词表大小64793，这是一个很妙的数字，因为它刚好在uint16的表示范围（0～65535的无符号整数），每一个token只需要两个字节即可表示，当我们的语料较大时候，相比常用的int32可以节省一半的存储空间。

分词器下载：https://huggingface.co/THUDM/chatglm2-6b/tree/main

## 中文预训练和SFT数据

### 预训练数据
这里选用的预训练数据主要为开源的百度百科和wiki百科数据集。
- Wiki中文百科（25w词条）[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- BaiduBaiKe（563w词条）
[百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb)
 提取码: bwvb
- [Medical DataSet](https://huggingface.co/datasets/shibing624/medical/tree/main)

除此之外，为了让模型具备在某一个专有领域的能力，这里选用了“医疗问答”作为切入点，尝试收集了很多的医疗数据和上面的通用语料一起喂给模型。

上面提到的数据相关的预处理代码见Github Repo.

### SFT数据
中文SFT语料最近陆陆续续开源了很多（[bell](https://huggingface.co/datasets/BelleGroup/train_1M_CN)、[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)等），但是坦白讲，质量都不高，大家可自行下载并可以选择性的进行清洗。清洗SFT数据是个耗时耗力的工作（可选），根据作者微调经验，一份高质量的SFT数据，哪怕只有几千条也可以媲美鱼龙混杂的bell之类百万级的SFT数据。

## 语料预处理和样本构建
### 预训练样本构建
数据预处理采取训练GPT的通用做法，对语料进行提前分词，对一个样本做完分词后在末尾加上一个结束符号，与下一个样本区分开。然后将所有的训练语料拼接成一个数组（np.uint16）以.bin二进制格式存储到磁盘上。当语料过大，为了避免内存溢出，可以使用memmap格式。完整代码可以到对应的Github Repo里面查看！

### SFT样本构建
因为SFT语料一般较小，我们没必要提前分词，而是在构建Dataloader的时候进行分词构建batch送给模型。

基本逻辑如下：
- prompt和answer之间一定要有一个开始符隔开，然后answer后需要一个结束符。
- 计算loss的时候，对prompt部分的loss进行mask，只计算answer部分的loss即可。

完整代码可以到对应的Github Repo里面查看，参考dataset_sft.py

## 预训练+SFT

本文的目的是使用有限的算力走完预训练+SFT的完整流程，同时保证得到一个可玩的中文baby llama2聊天机器人。因为在训练的时候需要严格控制模型的参数量和FLOPs。
大家可以根据自己算力的情况合理的调节以下参数，下面是我使用的配置：
- max_seq_len = 512
- dim = 512
- n_layers = 8
- n_heads = 8

整个预训练+SFT使用3090单卡即可完成，50亿token的语料过完一遍预计需要2天时间。当然也支持DDP训练，所以如果算力允许，可以加大模型和数据，将得到一个更好的模型！

推理脚本可以参考eval.py，因为在预训练的时候使用了很多医疗相关的语料，这里使用了一个医疗问答比赛的数据做了bleu指标的验证，大家感可以自行修改，后续作者也会不断完善代码。
特斯拉 AI 负责人 Andrej Karpathy 

## 训练效果评测
作者目前用了50亿中文token，单卡3090训练了2天时间，得到了一个参数量大概40M的极小的baby-llama2。经过SFT后可以达到很好的中文问答效果，特别是在医疗问答上，因为加了大量相关预训练语料，效果非常不错。但是缺乏全面严谨的开放问答评测指标，后续有时间会补上，也欢迎大家提pr，平时工作繁忙，只能周末更新，后续有时间了会持续更新语料，迭代模型。

下面是模型的问答效果图👇

[参考llama2.c](https://github.com/karpathy/llama2.c)


