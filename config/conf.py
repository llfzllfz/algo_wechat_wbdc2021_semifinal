import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(BASE_DIR, '../data')
DATASET_PATH = os.path.join(ROOT_PATH, "wedata/wechat_algo_data2")
DATASET_PATH_A = os.path.join(ROOT_PATH, 'wedata/wechat_algo_data1')
USER_ACTION_A = os.path.join(DATASET_PATH_A, 'user_action.csv')
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")
MODEL_DIR = os.path.join(ROOT_PATH, 'model')
SUMIT_DIR = os.path.join(ROOT_PATH, 'submission')
TEMP_DIR = os.path.join(ROOT_PATH, 'temp')

