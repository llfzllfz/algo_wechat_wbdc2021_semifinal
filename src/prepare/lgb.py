random_seed = 502
import os
os.environ['PYTHONHASHSEED'] = str(random_seed)
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(os.path.join(BASE_DIR, 'config'))
import conf

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
# from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from collections import defaultdict
from queue import Queue
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
import time
from gensim.models.word2vec import Word2Vec
random.seed(random_seed)
np.random.seed(random_seed)
from joblib import load
import gc

def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols, ncols = 75):
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


## 从官方baseline里面抽出来的评测函数
def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc


def train():
#     user_action = pd.read_csv(conf.USER_ACTION, nrows = 100000)
#     user_action_a = pd.read_csv(conf.USER_ACTION_A, nrows = 100000)
    user_action = pd.read_csv(conf.USER_ACTION)
    user_action_a = pd.read_csv(conf.USER_ACTION_A)
    user_action = user_action.append(user_action_a)
#     user_action = user_action[user_action.date_ >= 6]
    feed_info = pd.read_csv(conf.FEED_INFO)
    feed_pca = pd.read_csv(os.path.join(conf.TEMP_DIR, 'PCA.csv'))
    feed_info['videoplayseconds'] *= 1000
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
    
    # PCA处理
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
    for idx in range(64):
        feed_pca['pca'+str(idx)] = feed_pca['pca_embedding'].apply(lambda x:x[idx])
    feed_pca = reduce_mem(feed_pca, feed_pca.columns)
    
    # w2v中的tag---mean
    tag_w2v = Word2Vec.load(os.path.join(BASE_DIR, 'data/model/tag_w2v_emb_20.model'))
    def get_(item):
        if pd.isna(item) or pd.isnull(item):
            return [0 for idx in range(20)]
        item = str(item)
        item = item.split(';')
        lists = np.array([0 for idx in range(20)])
        nums = 0
        for x in item:
            if x == '' or x == 'nan' or x == 'NAN' or x == 'Nan':
                continue
            lists = lists + np.array(tag_w2v[x], dtype = 'float16')
            nums = nums + 1
        return np.array(lists / nums, dtype = 'float16')
    feed_info['tag_emb_mean'] = feed_info['manual_tag_list'].apply(get_)
    for idx in range(20):
        feed_info['tag_emb_mean'+str(idx)] = feed_info['tag_emb_mean'].apply(lambda x:x[idx])
    
    # w2v中的tag---attention
    tag_w2v = Word2Vec.load(os.path.join(BASE_DIR, 'data/model/tag_w2v_emb_20.model'))
    def get_(item):
        if pd.isna(item) or pd.isnull(item):
            return [0 for idx in range(20)]
        item = str(item)
        item = item.split(';')
        lists = np.array([0 for idx in range(20)])
        nums = 0
        for x in item:
            x = x.split(' ')
            if len(x) < 2:
                continue
    #         if x == '' or x == 'nan' or x == 'NAN' or x == 'Nan':
    #             continue
            try:
                lists = lists + np.array(tag_w2v[x[0]]) * float(x[1])
            except:
                lists = lists + np.array([0 for idx in range(20)]) * float(x[1])
        return np.array(lists, dtype = 'float16')
    feed_info['tag_emb_att'] = feed_info['machine_tag_list'].apply(get_)
    for idx in range(20):
        feed_info['tag_emb_att'+str(idx)] = feed_info['tag_emb_att'].apply(lambda x:x[idx])
    
    # w2v中的keyword---mean
    keyword_w2v = Word2Vec.load(os.path.join(BASE_DIR, 'data/model/keyword_w2v_emb_20.model'))
    def get_(item):
        if pd.isna(item) or pd.isnull(item):
            return [0 for idx in range(20)]
        item = str(item)
        item = item.split(';')
        lists = np.array([0 for idx in range(20)])
        nums = 0
        for x in item:
            if x == '' or x == 'nan' or x == 'NAN' or x == 'Nan':
                continue
            lists = lists + np.array(keyword_w2v[x])
            nums = nums + 1
        return np.array(lists / nums, dtype = 'float16')
    feed_info['keyword_emb_mean'] = feed_info['manual_keyword_list'].apply(get_)
    for idx in range(20):
        feed_info['keyword_emb_mean'+str(idx)] = feed_info['keyword_emb_mean'].apply(lambda x:x[idx])
    
    feed_info = reduce_mem(feed_info, feed_info.columns)
    
    for x in ['bgm_singer_id', 'bgm_song_id']:
        feed_info[x] = feed_info[x].fillna(feed_info[x].max()+1)
        feed_info[x] = feed_info[x].astype(int)
    feed_info = feed_info[['feedid', 'authorid', 'videoplayseconds', 'bgm_singer_id', 'bgm_song_id', 'sparse_tag', 'sparse_keyword', 'max_pro_tag'] + ['tag_emb_mean'+str(idx) for idx in range(20)] + ['tag_emb_att'+str(idx) for idx in range(20)] + ['keyword_emb_mean'+str(idx) for idx in range(20)]]
    user_action = user_action.merge(feed_info, on = 'feedid', how = 'left')
    user_action = user_action.merge(feed_pca[['feedid'] + ['pca'+str(idx) for idx in range(64)]], on = 'feedid', how = 'left')
    user_action = reduce_mem(user_action, user_action.columns)
    
    file_lists = ['d_userid_feature.csv', 'd_authorid_feature.csv', 'd_feedid_feature.csv', 'd_sparse_tag_feature.csv', 'd_sparse_keyword_feature.csv', 'd_bgm_singer_id_feature.csv', 'd_bgm_song_id_feature.csv'] + ['stat_df_con_userid.csv', 'stat_df_con_feedid.csv', 'stat_df_con_authorid.csv', 'stat_df_con_bgm_singer_id.csv', 'stat_df_con_bgm_song_id.csv', 'stat_df_con_sparse_tag.csv', 'stat_df_con_sparse_keyword.csv']
    merge_feature = ['userid', 'authorid', 'feedid', 'sparse_tag', 'sparse_keyword', 'bgm_singer_id', 'bgm_song_id', ['userid', 'date_'], ['feedid', 'date_'], ['authorid', 'date_'], ['bgm_singer_id', 'date_'], ['bgm_song_id', 'date_'], ['sparse_tag', 'date_'], ['sparse_keyword', 'date_']]
    for idx in range(len(file_lists)):
        df = pd.read_csv(os.path.join(conf.TEMP_DIR, file_lists[idx]))
        df = reduce_mem(df, df.columns)
        user_action = pd.merge(user_action, df, how = 'left', on = merge_feature[idx])
    user_action = user_action.fillna(0)
#     user_action = reduce_mem(user_action, user_action.columns)
    dicts = {}
    for x in user_action.columns:
        print(type(user_action[x][0]))
        dicts[x] = type(user_action[x][0])
    
#     np.save(os.path.join(conf.TEMP_DIR, 'type.npy'), dicts)
    
    y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
#     trn_x = user_action[user_action.date_<14]
    val_x = user_action[user_action.date_==14]
#     del user_action
#     gc.collect()
    feature_cols = user_action.columns.tolist()
#     print(feature_cols)
    cols = [x for x in feature_cols if x not in y_list + ['date_', 'play', 'stay']]
    print(cols)
    print(len(cols))
    
    all_cols = []
    for y in y_list:
        lists = []
        for f in feature_cols:
            if f.find(y) != -1 and f not in y_list:
                lists.append(f)
        all_cols.append(lists)
    
    common_cols = [x for x in feature_cols if x not in y_list + ['date_', 'play', 'stay'] + all_cols[0] + all_cols[1] + all_cols[2] + all_cols[3] + all_cols[4] + all_cols[5] + all_cols[6]]

    uauc_list = []
    r_list = []
    iters = 0
    
    user_action.to_csv(os.path.join(conf.TEMP_DIR, 'lgb_data_emb.csv'), index = False)
#     trn_x.to_csv(os.path.join(conf.TEMP_DIR, 'trn.csv'), index = False)
    val_x.to_csv(os.path.join(conf.TEMP_DIR, 'val.csv'), index = False)
    
if __name__ == '__main__':
    train()
    