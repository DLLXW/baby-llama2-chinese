## baby-llama2-chinese
用于从头预训练+SFT一个小参数量的中文LLaMa2的仓库；24G单卡即可运行得到一个流畅中文问答的chat-llama2.

## 训练数据
- Wiki中文百科（25w词条）[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- BaiduBaiKe（563w词条）
[百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb)
 提取码: bwvb
- [Medical DataSet](https://huggingface.co/datasets/shibing624/medical/tree/main)

因为最开始有从头训练llama2的打算是为了做一个临床问答的比赛，所以收集了很多的医疗数据，这里也提供下数据预处理的脚本。


## 中文分词器

这里的中文分词器采用ChatGLM2的分词器，词表大小64793，在uint16的表示范围（0～65535的无符号整数），因此如果语料较大，可以节省一半的存储空间。

## 预训练语料预处理
```python
#脚本里面每一个函数对应一个语料库的预处理，搭建新加语料可以自行扩展。
python data_process.py
#运行结束后，会在./data目录下产生.bin文件
```
数据预处理采取GPT的通用做法，对语料进行提前分词，对一个样本做完分词后在末尾加上一个结束符号，与下一个样本区分开。然后将所有的训练语料拼接成一个数组（np.uint16）以.bin二进制格式存储到磁盘上。如果语料过大，避免内存溢出，可以使用mmap格式，当然我们这里语料不到10亿tokens，不必转成mmap。

## SFT样本构建
中文SFT语料网上最近很多，大家自行下载。因为SFT语料一般较小，我们没必要提前分词，而是在构建Dataloader的时候进行分词构建batch送给模型。所以自行参考dataset_sft.py即可！

基本逻辑如下：
- prompt和answer之间一定要有一个开始符隔开，然后answer后需要一个结束符。
- 计算loss的时候，对prompt部分的loss进行mask，只计算answer部分的loss即可。

## 预训练+SFT

```python
#预训练
python pretrain.py
#SFT
python train_sft.py
```
根据自己算力的情况合理的调节以下参数，控制模型的计算量和参数量
- max_seq_len = 512
- dim = 512
- n_layers = 8
- n_heads = 8

推理脚本可以参考eval.py，这里使用100条比赛数据做了bleu的验证，大家感兴趣可以自行修改，后续作者也会不断完善代码。

## 训练效果评测
作者目前用了10亿中文token，单卡3090训练了一个参数量大概40M的极小的baby-llama2。经过SFT后可以达到很好的中文问答效果，特别是在医疗问答上，因为加了大量相关预训练语料，效果非常不错。但是缺乏全面严谨的开放问答评测指标，后续有时间会补上，也欢迎大家提pr，平时工作繁忙，只能周末更新，后续有时间了会持续更新语料，迭代模型。

[参考llama2.c](https://github.com/karpathy/llama2.c)
