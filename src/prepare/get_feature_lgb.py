import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(os.path.join(BASE_DIR, 'config'))
import conf


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import gc
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import time
pd.set_option('display.max_columns', None)

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


def get_feature():
    df_a = pd.read_csv(conf.USER_ACTION_A)
    df = pd.read_csv(conf.USER_ACTION)
    df = df.append(df_a)
#     df['date_'] = df['date_'].apply(lambda x:x%7+1)
    feed_info = pd.read_csv(conf.FEED_INFO)
    feed_info['videoplayseconds'] *= 1000
    for x in ['tag', 'keyword']:
        lb = load(os.path.join(conf.MODEL_DIR, '{}.joblib'.format(x)))
        feed_info['fillna_{}_list'.format(x)] = feed_info['manual_{}_list'.format(x)].fillna(-1)
        feed_info['fillna_{}_list'.format(x)] = feed_info['fillna_{}_list'.format(x)].apply(lambda x:str(x))
        feed_info['sparse_{}'.format(x)] = lb.transform(feed_info['fillna_{}_list'.format(x)])
    
    for x in ['bgm_singer_id', 'bgm_song_id']:
        feed_info[x] = feed_info[x].fillna(feed_info[x].max()+1)
        feed_info[x] = feed_info[x].astype(int)
    
    feed_info = feed_info[[
        'feedid', 'authorid', 'videoplayseconds', 'bgm_singer_id', 'bgm_song_id', 'sparse_tag', 'sparse_keyword'
    ]]

    df = df.merge(feed_info, on='feedid', how='left')
    df = reduce_mem(df, df.columns)
    
    df['is_finish'] = ((df['play'] / df['videoplayseconds']) >= 0.9).astype('int8')
    df['play_times'] = df['play'] / df['videoplayseconds']
    df['stay_times'] = df['stay'] / df['videoplayseconds']
    play_cols = [
        'is_finish', 'play_times', 'stay_times', 'play', 'stay'
    ]
    
    y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
    
    n_day = 5
    max_day = 15
    print('局部')
    for stat_cols in tqdm([
        ['userid'],
        ['feedid'],
        ['authorid'],
        ['bgm_singer_id'],
        ['bgm_song_id'],
        ['sparse_tag'],
        ['sparse_keyword']
    ]):
        f = '_'.join(stat_cols)
        stat_df = pd.DataFrame()
        for target_day in range(2, max_day + 1):
            left, right = max(target_day - n_day, 1), target_day - 1
    
            tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
            tmp['date_'] = target_day
    
            tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
            
            tmp['{}_{}day_nunique'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('nunique')
    
            g = tmp.groupby(stat_cols)
            tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
    
            feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day), '{}_{}day_nunique'.format(f, n_day)]
    
            for x in play_cols[1:]:
                for stat in ['max', 'mean']:
                    tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                    feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))
    
            for y in y_list:
                tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
                tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
                feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])
    
            tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
            # print(stat_df.shape)
            del g, tmp
        stat_df.to_csv(os.path.join(conf.TEMP_DIR, 'stat_df_con_{}.csv'.format(f)),index = False)
        del stat_df
        gc.collect()
    
    print('全局')
    ## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行
    for f in tqdm(['userid', 'feedid', 'authorid', 'bgm_singer_id', 'bgm_song_id', 'sparse_tag', 'sparse_keyword']):
        df[f + '_count'] = df[f].map(df[f].value_counts())
#     out_lists = ['userid_count', 'feedid_count', 'authorid_count', 'bgm_singer_id_count', 'bgm_song_id_count', 'sparse_tag_count', 'sparse_keyword_count']
#     df[out_lists].to_csv(os.path.join(conf.TEMP_DIR, 'all_day_feature.csv'),index = False)
    
    for f1, f2 in tqdm([
        ['userid', 'feedid'],
        ['userid', 'authorid'],
        ['userid', 'sparse_tag'],
        ['userid', 'sparse_keyword'],
        ['userid', 'bgm_singer_id'],
        ['userid', 'bgm_song_id']
    ]):
        df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')
        df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')
        
#     for f1, f2 in tqdm([
#         ['userid', 'authorid'],
#         ['userid', 'sparse_tag'],
#         ['userid', 'sparse_keyword'],
#         ['userid', 'bgm_singer_id'],
#         ['userid', 'bgm_song_id']
#     ]):
#         df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')
#         df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
#         df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)
    
    for x in tqdm(['userid', 'authorid', 'sparse_tag', 'sparse_keyword', 'bgm_singer_id', 'bgm_song_id']):
        df['videoplayseconds_in_{}_mean'.format(x)] = df.groupby(x)['videoplayseconds'].transform('mean')
    
    for x in tqdm(['authorid', 'sparse_tag', 'sparse_keyword', 'bgm_singer_id', 'bgm_song_id']):
        df['feedid_in_{}_nunique'.format(x)] = df.groupby(x)['feedid'].transform('nunique')
    
    userid_feature = ['userid', 'userid_count'] + ['{}_in_{}_nunique'.format(f2, f1) for f1, f2 in [
        ['userid', 'feedid'],
        ['userid', 'authorid'],
        ['userid', 'sparse_tag'],
        ['userid', 'sparse_keyword'],
        ['userid', 'bgm_singer_id'],
        ['userid', 'bgm_song_id']
    ]] + ['videoplayseconds_in_userid_mean']
    feedid_feature = ['feedid', 'feedid_count', 'userid_in_feedid_nunique']
    authorid_feature = ['authorid', 'authorid_count', 'userid_in_authorid_nunique', 'videoplayseconds_in_authorid_mean', 'feedid_in_authorid_nunique']
    sparse_tag_feature = ['sparse_tag', 'sparse_tag_count'] + [x.format('sparse_tag') for x in ['userid_in_{}_nunique', 'videoplayseconds_in_{}_mean', 'feedid_in_{}_nunique']]
    sparse_keyword_feature = ['sparse_keyword', 'sparse_keyword_count'] + [x.format('sparse_keyword') for x in ['userid_in_{}_nunique', 'videoplayseconds_in_{}_mean', 'feedid_in_{}_nunique']]
    bgm_singer_id_feature = ['bgm_singer_id', 'bgm_singer_id_count'] + [x.format('bgm_singer_id') for x in ['userid_in_{}_nunique', 'videoplayseconds_in_{}_mean', 'feedid_in_{}_nunique']]
    bgm_song_id_feature = ['bgm_song_id', 'bgm_song_id_count'] + [x.format('bgm_song_id') for x in ['userid_in_{}_nunique', 'videoplayseconds_in_{}_mean', 'feedid_in_{}_nunique']]
    
    for x1, x2 in tqdm([
        ['userid_feature',userid_feature],
        ['feedid_feature',feedid_feature],
        ['authorid_feature',authorid_feature],
        ['sparse_tag_feature',sparse_tag_feature],
        ['sparse_keyword_feature',sparse_keyword_feature],
        ['bgm_singer_id_feature',bgm_singer_id_feature],
        ['bgm_song_id_feature',bgm_song_id_feature]
    ]):
        to_df = df[x2]
        to_df.to_csv(os.path.join(conf.TEMP_DIR, '{}.csv'.format(x1)),index = False)
    
if __name__ == '__main__':
    get_feature()