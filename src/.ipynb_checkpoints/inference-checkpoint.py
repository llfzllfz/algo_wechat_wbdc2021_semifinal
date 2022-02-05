import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(os.path.join(BASE_DIR, 'config'))
import conf

sys.path.append(os.path.join(BASE_DIR, 'src/model'))
from MyDeepFm import *

from gensim.models.word2vec import Word2Vec
import time
import pandas as pd
from queue import Queue
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tqdm import tqdm
from joblib import load

import lightgbm as lgb
from joblib import load
from gensim.models.word2vec import Word2Vec

import gc
def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

def get_test(path):
    data = pd.read_csv(path)
    feed_info = pd.read_csv(conf.FEED_INFO)
    feed_info['group_videoplayseconds'] = feed_info['videoplayseconds'].apply(lambda x:int(x * 1000 / 60))
    mms = MinMaxScaler(feature_range=(0, 1))
    feed_info[['dense_videoplayseconds']] = mms.fit_transform(feed_info[['videoplayseconds']])
    # PCA
    feed_pca = pd.read_csv(os.path.join(conf.TEMP_DIR, 'PCA.csv'))
    def get_(item):
        item = item.replace('[', '')
        item = item.replace(']', '')
        item = item.replace(',', '')
        item = item.split(' ')
        lists = []
        for x in item:
            if x == ' ' or x == '  ' or x == '':
                continue
            lists.append(float(x))
        return np.array(lists, dtype = 'float16')
    feed_pca['pca_embedding'] = feed_pca['PCA_embedding'].apply(get_)

    for x in ['bgm_singer_id', 'bgm_song_id']:
        feed_info[x] = feed_info[x].fillna(feed_info[x].max()+1)
        feed_info[x] = feed_info[x].astype(int)
        
    feed_info['manual_tag_list'] = feed_info['manual_tag_list'].fillna('-1')
    feed_info['manual_keyword_list'] = feed_info['manual_keyword_list'].fillna('-1')
    max_tag = 0
    max_keyword = 0
    def get_tag(item):
        tag_list = item.split(';')
        lists = []
        for x in tag_list:
            if x == ' ' or x == '  ' or x == '':
                continue
            lists.append(int(x) + 1)
        for x in range(13):
            lists.append(0)
        return np.array(lists[:11])
    feed_info['tag_list'] = feed_info['manual_tag_list'].apply(get_tag)
    feed_info['max_tag'] = feed_info['tag_list'].apply(lambda x:max(x))

    def get_keyword(item):
        keyword_list = item.split(';')
        lists = []
        for x in keyword_list:
            if x == ' ' or x == '  ' or x == '':
                continue
            lists.append(int(x) + 1)
        for x in range(19):
            lists.append(0)
        return np.array(lists[:18])
    feed_info['keyword_list'] = feed_info['manual_keyword_list'].apply(get_keyword)
    feed_info['max_keyword'] = feed_info['keyword_list'].apply(lambda x:max(x))

    max_tag = feed_info['max_tag'].max()
    max_keyword = feed_info['max_keyword'].max()
    
    # sparse_tag, sparse_keyword
    for x in ['tag', 'keyword']:
        lb = load(os.path.join(conf.MODEL_DIR, '{}.joblib'.format(x)))
        feed_info['fillna_{}_list'.format(x)] = feed_info['manual_{}_list'.format(x)].fillna(-1)
        feed_info['fillna_{}_list'.format(x)] = feed_info['fillna_{}_list'.format(x)].apply(lambda x:str(x))
        feed_info['sparse_{}'.format(x)] = lb.transform(feed_info['fillna_{}_list'.format(x)])
    def get_max_tag(item):
        max_pro = 0
        tag = 0
        item = item.split(';')
        for x in item:
            y = x.split(' ')
            if len(y) < 2:
                continue
            if max_pro < float(y[1]):
                max_pro = float(y[1])
                tag = int(y[0])
        return tag
    feed_info['machine_tag_list'] = feed_info['machine_tag_list'].fillna('0')
    feed_info['max_pro_tag'] = feed_info['machine_tag_list'].apply(get_max_tag)
    
    data = pd.merge(data, feed_info[['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'dense_videoplayseconds', 'tag_list', 'keyword_list', 'group_videoplayseconds', 'sparse_tag', 'sparse_keyword', 'max_pro_tag']], how = 'left', on = 'feedid')
    data = pd.merge(data, feed_pca[['feedid', 'pca_embedding']], how = 'left', on = 'feedid')

    data = reduce_mem(data, data.columns)
    features = ['dense_videoplayseconds']
    return data, features

def nn():
    test_path = sys.argv[2]
    print('test_path: ', test_path)
    data,features = get_test(test_path)
    print(data.shape)
    y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
    
    sparse_feature_names = ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'userid', 'device','group_videoplayseconds', 'sparse_tag', 'sparse_keyword', 'max_pro_tag']
    dense_feature_names = ['pca_embedding']
    
    test_data_list = [data[x].values for x in sparse_feature_names]
    float_embedding = np.array([i for i in data['pca_embedding'].values])
    tag_list = np.array([i for i in data['tag_list'].values])
    keyword_list = np.array([i for i in data['keyword_list'].values])
    test_data_list.append(tag_list)
    test_data_list.append(keyword_list)
    test_data_list.append(float_embedding)
    test_data_list.extend([data[x].values for x in features])
    print('start preds...')
    
    for action in y_list:
        model = tf.keras.models.load_model(os.path.join(conf.MODEL_DIR, 'MyDeepFm_{}.h5').format(action), custom_objects={'DNN':DNN,'FM':FM,'PredictionLayer':PredictionLayer})
        data[action] = model.predict(test_data_list, batch_size = 5000)
        
    return data[['userid', 'feedid']+y_list]
    
if __name__ == '__main__':
    print('start...')
    data = nn()
    data.to_csv(os.path.join(conf.SUMIT_DIR, 'result.csv'), index = False)
    print('end...')