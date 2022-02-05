import os
import sys
import time
import numpy as np
from tqdm import tqdm
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(os.path.join(BASE_DIR, 'src/model'))
from MyDeepFm import *
sys.path.append(os.path.join(BASE_DIR, 'config'))
import conf
import pandas as pd
from gensim.models import word2vec

def get():
    data = pd.read_csv(conf.FEED_INFO)
    def get_(item):
        if pd.isna(item) or pd.isnull(item):
            return []
        item = str(item)
        item = item.split(';')
        lists = []
        for x in item:
            if x == '' or x == 'nan' or x == 'NAN' or x == 'Nan' or x == 'NaN':
                continue
            lists.append(x)
        return lists
    data['tag'] = data['manual_tag_list'].apply(get_)
    seq = []
    for idx in range(data.shape[0]):
        seq.append(data['tag'][idx])
    w2v = word2vec.Word2Vec(seq, min_count = 1, workers=8, size = 20)
    out_path = os.path.join(BASE_DIR, 'data/model/tag_w2v_emb_20.model')
    w2v.save(out_path)
        
if __name__ == '__main__':
    get()