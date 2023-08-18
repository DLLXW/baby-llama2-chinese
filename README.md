## baby-llama2-chinese
用于从头预训练+SFT一个小参数量的中文LLaMa2的仓库；24G单卡即可运行得到一个流畅中文问答的chat-llama2.

>20230818更新，因为第一版（50M参数）的版本，当时很多评测样例其实出现在了SFT数据中，所以让我误以为模型具备很流畅的问答能力，但是后面发现效果并没有那么好。后面使用了更多的数据和更大的模型，效果逐步提升。所以大家如果有充足的算力和时间，可以逐步尝试加大模型，将参数量扩到百M以上，其实消费级显卡也是完全可以接受的。

## 训练数据
- Wiki中文百科（25w词条）[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- BaiduBaiKe（563w词条）
[百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb)
 提取码: bwvb
- [Medical Dataset](https://huggingface.co/datasets/shibing624/medical/tree/main)

除此之外，为了让模型具备在某一个专有领域的能力，这里选用了“医疗问答”作为切入点，尝试收集了很多的医疗数据和上面的通用语料一起喂给模型。


## 中文分词器

因为在llama官方所提供的词表中，中文的部分只有700个，这也是llama中文能力聊胜于无的原因。为了训练自己的中文LLaMa，这里将引入新的中文分词器。为了方便，这里选择采用ChatGLM2的分词器，词表大小64793，这是一个很妙的数字，因为它刚好在uint16的表示范围（0～65535的无符号整数），每一个token只需要两个字节即可表示，当我们的语料较大时候，相比常用的int32可以节省一半的存储空间。

## 预训练语料预处理
```python
#脚本里面每一个函数对应一个语料库的预处理，搭建新加语料可以自行扩展。
python data_process.py
#运行结束后，会在./data目录下产生.bin文件
```
数据预处理采取GPT的通用做法，对语料进行提前分词，对一个样本做完分词后在末尾加上一个结束符号，与下一个样本区分开。然后将所有的训练语料拼接成一个数组（np.uint16）以.bin二进制格式存储到磁盘上。如果语料过大，避免内存溢出，可以选择mmap格式。

## SFT样本构建
中文SFT语料最近陆陆续续开源了很多（[bell](https://huggingface.co/datasets/BelleGroup/train_1M_CN)、[MOSS](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)、[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)等），但是坦白讲，质量都不高，大家可自行下载并需要进行清洗，清洗SFT数据是个耗时耗力的工作，但根据作者微调经验，一份高质量的SFT数据是相当重要的‼️（如果不清洗SFT数据，可能无法获得满意的SFT效果，建议大家在这块多花些时间）
中文SFT语料网上最近很多，大家自行下载。因为SFT语料一般较小，我们没必要提前分词，而是在构建Dataloader的时候进行分词构建batch送给模型。所以自行参考dataset_sft.py即可！

基本逻辑如下：
- prompt和answer之间一定要有一个开始符隔开，然后answer后需要一个结束符。
- 计算loss的时候，对prompt部分的loss进行mask，只计算answer部分的loss即可。

## 预训练+SFT

```python
#预训练
python pretrain.py
#SFT
python sft.py
```
根据自己算力的情况合理的调节以下参数，控制模型的计算量和参数量，这是第一版使用的参数
- max_seq_len = 512
- dim = 512
- n_layers = 8
- n_heads = 8

推理脚本可以参考eval.py，这里使用100条比赛数据做了bleu的验证，大家感兴趣可以自行修改，后续作者也会不断完善代码。

## 训练效果评测
作者目前用了20亿中文token，单卡3090训练了一个参数量大概50M的极小的baby-llama2。经过SFT后可以具备一定的中文问答效果，特别是在医疗问答上，因为加了大量相关预训练语料，效果不错。但是缺乏全面严谨的开放问答评测指标，后续有时间会补上，也欢迎大家提pr，平时工作繁忙，只能周末更新，后续有时间了会持续更新语料，迭代模型。
## 号召
平时工作繁忙，只能周末玩耍，欢迎大家一起共建这个小项目，这对于希望入门LLM的同学来说，是一次不可多得的练手机会，除了大规模预训练需要的（数据并行、模型并行、流水线并行）那一套，其余的LLM基本技术栈基本都有涵盖！


[参考llama2.c](https://github.com/karpathy/llama2.c)
