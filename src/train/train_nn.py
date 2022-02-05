random_seed = 1024
import os
os.environ['PYTHONHASHSEED'] = str(random_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(os.path.join(BASE_DIR, 'config'))
import conf

from gensim.models.word2vec import Word2Vec
from queue import Queue
from collections import defaultdict
from sklearn.metrics import log_loss, roc_auc_score
import random
from joblib import load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import save_model
random.seed(random_seed)
np.random.seed(random_seed)
tf.compat.v1.set_random_seed(random_seed)
sys.path.append(os.path.join(BASE_DIR, 'src/model'))
from MyDeepFm import *
# 按需分配GPU内存
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import pandas as pd
valid = sys.argv[1]
print(valid)
assert valid == 'valid_valid' or valid == 'valid' or valid == 'train'

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

# 数据读取
if valid == 'valid_valid':
    user_action = pd.read_csv(conf.USER_ACTION, nrows = 100000)
elif valid == 'valid':
    user_action = pd.read_csv(conf.USER_ACTION)
elif valid == 'train':
    user_action = pd.read_csv(conf.USER_ACTION)
    user_action_a = pd.read_csv(conf.USER_ACTION_A)
    user_action = user_action.append(user_action_a)
print(user_action.shape)

feed_info = pd.read_csv(conf.FEED_INFO)
feed_pca = pd.read_csv(os.path.join(conf.TEMP_DIR, 'PCA.csv'))
# feed_embedding = pd.read_csv(conf.FEED_EMBEDDINGS)

# feed_info数据处理
for x in ['bgm_singer_id', 'bgm_song_id']:
    feed_info[x] = feed_info[x].fillna(feed_info[x].max()+1)
    feed_info[x] = feed_info[x].astype(int)

feed_info['manual_tag_list'] = feed_info['manual_tag_list'].fillna('-1')
feed_info['manual_keyword_list'] = feed_info['manual_keyword_list'].fillna('-1')
max_tag = 0
max_keyword = 0
# tag处理（tag序列）
def get_tag(item):
    tag_list = item.split(';')
    lists = []
    for x in tag_list:
        if x == ' ' or x == '  ' or x == '':
            continue
        lists.append(int(x) + 1)
    for x in range(13):
        lists.append(0)
    return np.array(lists[:11], dtype = 'int16')
feed_info['tag_list'] = feed_info['manual_tag_list'].apply(get_tag)
feed_info['max_tag'] = feed_info['tag_list'].apply(lambda x:max(x))

# keyword处理（keyword序列）
def get_keyword(item):
    keyword_list = item.split(';')
    lists = []
    for x in keyword_list:
        if x == ' ' or x == '  ' or x == '':
            continue
        lists.append(int(x) + 1)
    for x in range(19):
        lists.append(0)
    return np.array(lists[:18], dtype = 'int16')
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

# check pca
feed_pca['len'] = feed_pca['pca_embedding'].apply(lambda x:len(x))
print(feed_pca['len'].max())

# videoplayseconds处理
feed_info['group_videoplayseconds'] = feed_info['videoplayseconds'].apply(lambda x:int(x * 1000 / 60))
mms = MinMaxScaler(feature_range=(0, 1))
feed_info[['dense_videoplayseconds']] = mms.fit_transform(feed_info[['videoplayseconds']])

# 数据合并
user_action = pd.merge(user_action, feed_info[['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'dense_videoplayseconds', 'tag_list', 'keyword_list', 'group_videoplayseconds', 'sparse_tag', 'sparse_keyword', 'max_pro_tag']], how = 'left', on = 'feedid')
user_action = pd.merge(user_action, feed_pca[['feedid', 'pca_embedding']], how = 'left', on = 'feedid')
user_action = reduce_mem(user_action, user_action.columns)


from tensorflow.keras.callbacks import Callback
import time
class uAUC(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val, self.user = validation_data
        self.exp = 1e-9
    def on_epoch_end(self, epoch, logs = {}):
        st_time = time.time()
        preds = self.model.predict(self.X_val, batch_size = 5000)
        ed_time = time.time()
        labels = self.y_val
        user_id_list = self.user
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
        print()
        print('uauc: ', user_auc, 'time: ', ed_time - st_time)



if valid == 'valid' or valid == 'valid_valid':
    train_data = user_action[user_action.date_!=14]
    test_data = user_action[user_action.date_==14]
elif valid == 'train':
    train_data = user_action


y_list = ['read_comment','like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
epoch_list = {'read_comment':2, 'like':2, 'click_avatar':2, 'forward':2, 'favorite':2, 'comment':3, 'follow':2}
sparse_feature_names = ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'userid', 'device','group_videoplayseconds', 'sparse_tag', 'sparse_keyword', 'max_pro_tag']

dense_feature_names = ['pca_embedding']
sparse_dict = {'feedid':112872,
              'authorid':18789,
              'bgm_song_id':25160,
              'bgm_singer_id':17501,
              'userid':250249,
              'device':user_action['device'].max()+1,
#               'play_times_max':user_action['play_times_max'].max()+1,
#               'play_times_mean':user_action['play_times_mean'].max()+1,
              'group_videoplayseconds':user_action['group_videoplayseconds'].max()+1,
              'sparse_tag':user_action['sparse_tag'].max()+1,
              'sparse_keyword':user_action['sparse_keyword'].max()+1,
              'max_pro_tag':user_action['max_pro_tag'].max()+1}
features = ['dense_videoplayseconds']
dense_dict = {}
for columns in dense_feature_names:
    dense_dict[columns] = 64

for columns in features:
    dense_dict[columns] = 1

train_data_list = [train_data[x].values for x in sparse_feature_names]
float_embedding = np.array([i for i in train_data['pca_embedding'].values])
tag_list = np.array([i for i in train_data['tag_list'].values])
keyword_list = np.array([i for i in train_data['keyword_list'].values])

train_data_list.append(tag_list)
train_data_list.append(keyword_list)
train_data_list.append(float_embedding)
train_data_list.extend([train_data[x].values for x in features])


if valid == 'valid' or valid == 'valid_valid':
    test_data_list = [test_data[x].values for x in sparse_feature_names]
    float_embedding = np.array([i for i in test_data['pca_embedding'].values])
    tag_list = np.array([i for i in test_data['tag_list'].values])
    keyword_list = np.array([i for i in test_data['keyword_list'].values])
    test_data_list.append(tag_list)
    test_data_list.append(keyword_list)
    test_data_list.append(float_embedding)
    test_data_list.extend([test_data[x].values for x in features])

print(user_action.info())

tag_feature = [11, max_tag+1]
keyword_feature = [18, max_keyword+1]

train_label = train_data[y_list]
if valid == 'valid' or valid == 'valid_valid':
    test_label = test_data[['userid'] + y_list]

del user_action
del train_data
del tag_list
gc.collect()
if valid == 'valid' or valid == 'valid_valid':
    del test_data
    gc.collect()

for action in y_list:
    print()
    print(action)
    model = DFX_TK2(sparse_dict, dense_dict, tag_feature = tag_feature, keyword_feature = keyword_feature)
    parallel_model = multi_gpu_model(model, gpus=2)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    parallel_model.compile(optimizer, loss = tf.keras.losses.binary_crossentropy, metrics = ['binary_crossentropy'])
    early_stopping = EarlyStopping(monitor='val_binary_crossentropy', patience=0, verbose=1)
    if valid == 'valid' or valid == 'valid_valid':
        uauc = uAUC(validation_data=(test_data_list, test_label[action].values, test_label['userid'].values))
        parallel_model.fit(train_data_list, train_label[action].values,
                  batch_size = 1024, verbose = 1, callbacks=[uauc], epochs = epoch_list[action]+1,
                  validation_data = (test_data_list, test_label[action].values))
    elif valid == 'train':
        parallel_model.fit(train_data_list, train_label[action].values, 
                  batch_size = 65536, verbose = 1, validation_split=0, epochs = epoch_list[action])
        save_model(model, os.path.join(conf.MODEL_DIR, 'MyDeepFm_{}.h5').format(action))
    del model
    gc.collect()
print('finish!')