import json
import glob
import numpy as np
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer


def process_c4():
    c4_zh_paths = glob.glob('./data/c4_zh/*')
    c4_zh_paths=sorted(c4_zh_paths)
    print(len(c4_zh_paths))
    cnt=0
    token=0
    doc_ids=[]
    for per in tqdm(c4_zh_paths[:10]):
        with open(per,'r') as f:
            for line in f:
                text = json.loads(line)
                text = text['text']
                text_id=tokenizer.encode(text,add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id)>5:
                    doc_ids+=text_id
                #
                # if cnt%10000==0:
                #     print(cnt)
                cnt+=1
                #token+=len(text_id)
                #break
        #

        # arr = np.array(doc_ids,dtype=np.uint16)
        # with open('./data/c4-zh/{}.bin'.format(per.split('/')[-1].split('.')[0]),'wb') as f:
        #     f.write(arr.tobytes())
        # print(arr.shape)
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/c4_zh_0.bin','wb') as f:
        f.write(arr.tobytes())
    print(arr.shape)

if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

    process_c4()
