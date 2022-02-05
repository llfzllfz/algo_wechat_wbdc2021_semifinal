import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(os.path.join(BASE_DIR, 'config'))
import conf

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import conf
import os
from joblib import dump, load
feed_info = pd.read_csv(conf.FEED_INFO)
for x in ['tag', 'keyword']:
    lb = LabelEncoder()
    feed_info['fillna_{}_list'.format(x)] = feed_info['manual_{}_list'.format(x)].fillna(-1)
    feed_info['fillna_{}_list'.format(x)] = feed_info['fillna_{}_list'.format(x)].apply(lambda x:str(x))
    feed_info['sparse_{}'.format(x)] = lb.fit_transform(feed_info['fillna_{}_list'.format(x)])
    dump(lb, os.path.join(conf.MODEL_DIR, '{}.joblib'.format(x)))