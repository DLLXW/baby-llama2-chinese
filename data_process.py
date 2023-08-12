import json
import glob
import numpy as np
import os
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
#from zhconv import convert
def process_wiki_clean():
    with open('./data/wikipedia-cn-20230720-filtered.json','r') as f:
        data=json.load(f)
    doc_ids=[]
    for line in tqdm(data):
        text=line['completion']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/wiki.bin','wb') as f:
        f.write(arr.tobytes())
#
def process_medical(data_path,name):
    f=open(data_path,'r')
    doc_ids=[]
    while True:
        line=f.readline()
        if not line:
            break
        line=json.loads(line)
        text=line['text']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/medical_{}.bin'.format(name),'wb') as f:
        f.write(arr.tobytes()) 
#
def sft_to_pretrain():
    df=pd.read_csv('/home/trojanjet/project/weiqin/track1/medical_qa_144w.csv')
    doc_ids=[]
    for _,q,a in tqdm(df.itertuples()):
        text = q+a
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    print(arr.shape)
    with open('./data/medical_qa.bin','wb') as f:
        f.write(arr.tobytes())
#
def sft_process():
    with open('./data/alpaca_gpt4_data_zh.json','r') as f:
        data=json.load(f)
    #
    q_lst=[]
    a_lst=[]
    for per in data:
        q=per['instruction']
        i=per['input']
        a=per['output']
        q=q+i
        if len(q)<10 or len(a)<5:
            continue
        if len(q)>256 or len(a)>256:
            continue
        q_lst.append(q)
        a_lst.append(a)
    #
    with open('../track1/train_valid.json','r') as f:
        data=json.load(f)
    #
    for l in data:
        q_lst.append(l['question'])
        a_lst.append(l['answer'])
    #
    df=pd.DataFrame(columns=['prompt','answer'])
    df['prompt']=q_lst
    df['answer']=a_lst
    df.to_csv('data/sft_data.csv',index=False)
    print(df)
if __name__=="__main__":
    os.makedirs('./data/',exist_ok=True)
    tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    process_wiki_clean()
    # process_medical('./data/medical_book_zh.json','book')
    # process_medical('./data/train_encyclopedia.json','encyclopedia')
    #sft_to_pretrain()
    #sft_process()