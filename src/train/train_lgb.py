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

print('start read test_data...')
val_x = pd.read_csv(os.path.join(conf.TEMP_DIR, 'val.csv'))
# val_x = pd.read_csv(r'/home/tione/notebook/B/src/train/t.csv')

uauc_list = []
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
epoch_list = {'read_comment':700, 'like':700, 'click_avatar':260, 'forward':150, 'favorite':30, 'comment':30, 'follow':50}
score_list = []
for y in y_list:
    print('=========', y, '=========')
    t = time.time()
    ignore_column = [x for x in y_list if x != y]
    ignore_name = 'name:date_,play,stay'
    for x in ignore_column:
        ignore_name = ignore_name +  ',' + x
    print(ignore_name)
    train_data = lgb.Dataset(os.path.join(conf.TEMP_DIR, 'lgb_data_emb.csv'))
    valid_data = lgb.Dataset(os.path.join(conf.TEMP_DIR, 'val.csv'))
    params = {
        'task':'train',
        'objective':'binary',
        'boosting':'gbdt',
#         'num_iteration':5,
        'num_iteration':epoch_list[y],
        'num_threads':60,
#         'early_stopping_round':50,
        'learning_rate':0.05,
        'num_leaves':127,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'metric':'auc',
        'two_round':True,
        'label_column':'name:'+y, 
#         'ignore_column':'name:like,click_avatar,forward,comment,follow,favorite,date_,play,stay',
        'ignore_column':ignore_name,
        'header':True,
        'categorical_feature':'name:userid,feedid,authorid,bgm_singer_id,bgm_song_id,device,sparse_tag,sparse_keyword'
#         'predict_disable_shape_check':True

    }
    clf = lgb.train(params = params, train_set = train_data, valid_sets = [valid_data], verbose_eval = 10)
    clf.save_model(os.path.join(conf.MODEL_DIR, 'lgb_{}_emb.txt'.format(y)))
    
#     clf = lgb.Booster(model_file = os.path.join(conf.MODEL_DIR, 'lgb_{}_.txt'.format(y)))
    
    feature_cols = val_x.columns.tolist()
    cols = [x for x in feature_cols if x != y and x not in score_list]
    print(len(cols))
    val_x[y + '_score'] = clf.predict(val_x[cols])
    score_list.append(y+'_score')
    val_uauc = uAUC(val_x[y].values, val_x[y + '_score'].values, val_x['userid'].values)
    uauc_list.append(val_uauc)
    print(val_uauc)
    print('runtime: {}\n'.format(time.time() - t))
print(uauc_list)
    