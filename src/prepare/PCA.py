import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(os.path.join(BASE_DIR, 'config'))
import conf


import pandas as pd
import numpy as np
feed_embeddingss = pd.read_csv(conf.FEED_EMBEDDINGS)

def str2list(x):
    x = x.split(' ')
    x = x[:512]
    for i in range(len(x)):
        x[i] = float(x[i])
    return x
    
feed_embeddingss['feed_embedding'] = feed_embeddingss['feed_embedding'].apply(lambda x:str2list(x))
from sklearn.decomposition import PCA
t = np.array([i for i in feed_embeddingss['feed_embedding'].values])
pca = PCA(n_components=64)
pca.fit(t)
feed_embeddingss['PCA_embedding'] = feed_embeddingss['feed_embedding'].apply(lambda x:pca.transform([x]))
feed_embeddingss[['feedid', 'PCA_embedding']].to_csv(os.path.join(conf.TEMP_DIR, 'PCA.csv'), index=False)
print(feed_embeddingss.head(10))