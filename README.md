# 赛题描述

​	本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。本次比赛以多个行为预测结果的加权uAUC值进行评分。

# 补充说明

初赛使用deepfm和lgb进行融合。

复赛无法run lgb，修改deepfm模型，模型中加入卷积部分，相较于单纯deepfm，约有几个千分点的提升。

/src/model/MyDeepFm.py中包含很多模型，基本都是中间尝试的时候修改的模型，最后采用的是DFX_TK2模型。

卷积分为两块：

- deepfm中DNN层附加一个卷积层
- keyword和tag序列采用卷积初始化

模型由以下四个模块组合成：

- 一次项
- 二次项
- DNN
- DNN Embedding+卷积+DNN

模型具体结构详见/src/model/MyDeepFm.py/DFX_TK2

deepfm中间实现部分参考了deepctr中的deepfm。



# 复赛提交

## 1.环境依赖

- tensorflow-gpu==2.1.0
- pandas==1.2.3
- numpy==1.19.5
- tqdm==4.59.0
- sklearn==0.24.1
- deepctr[gpu]==0.8.5
- lightgbm==3.2.0
- gensim==3.8.3
- python3.7

## 2.目录结构

```python
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing train/inference data and training models, including pretrained models
├── inference.sh, script for inference 
├── src
│   ├── prepare, codes for preparing train/inference dataset
|       ├──drop_duplicates_lgb_data.py
|       ├──get_feature_lgb.py
|       ├──get_keyword_w2v_emb.py
|       ├──get_tag_w2v_emb.py
|       ├──lb.py
|       ├──lgb.py
|       ├──PCA.py
│   ├── model, codes for model architecture
|       ├──MyDeepFm.py  
|   ├── train, codes for training
|       ├──train_lgb.py
|       ├──train_nn.py
|   ├── inference.py, main function for inference on test dataset
├── data
│   ├── wedata, dataset of the competition
│       ├── wechat_algo_data1, preliminary dataset
|       ├── wechat_algo_data2
│   ├── submission, prediction result after running inference.sh
│   ├── model, model files (e.g. tensorflow checkpoints)
|       ├──keyword.joblib
|       ├──MyDeepFm_click_avatar.h5
|       ├──MyDeepFm_comment.h5
|       ├──MyDeepFm_favorite.h5
|       ├──MyDeepFm_follow.h5
|       ├──MyDeepFm_forward.h5
|       ├──MyDeepFm_like.h5
|       ├──MyDeepFm_read_comment.h5
|       ├──tag.joblib
|   ├── temp, prepare files
|       ├──PCA.csv
├── config, configuration files for your method (e.g. yaml file)
|       ├──conf.py 
```

## 3.运行流程

- 进入目录：cd /home/tione/notebook/wbdc2021-semi
- 运行init.sh并进入环境：source init.sh
- 数据准备和训练模型：sh train.sh
- 预测并生成结果文件：sh inference.sh /home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data2/test_b.csv

## 4.模型及特征

- 模型：MyDeepFm
- 参数：
  - batch_size: 65536
  - embed_dim:16
  - learning_rate:0.001
  - epochs:{'read_comment':2, 'like':2, 'click_avatar':2, 'forward':2, 'favorite':2, 'comment':3, 'follow':2}

- 特征:
  - sparse_feature：'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'userid', 'device','group_videoplayseconds', 'sparse_tag', 'sparse_keyword', 'max_pro_tag'
  - dense_feature:pca_embedding, dense_features
  - other:tag_list, keyword_list



## 5.算法性能

- 资源配置：2*P40_48G显存_14核CPU_112G内存

- 预测耗时
  - 总预测时长：274s
  - 单个目标行为2000条样本的平均预测时长：18.4111ms



## 6.代码说明

模型预测部分代码位置如下：

| 路径             | 行数 | 内容                                                         |
| :--------------- | :--- | :----------------------------------------------------------- |
| src/inference.py | 161  | `data[action] = model.predict(test_data_list, batch_size = 5000)` |

