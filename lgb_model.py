# -*- coding: utf-8 -*-
"""

此处使用     lgb

"""
from genfeature import *
import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


def make_feat(train_data, test_data):
    train_id = train_data.member_id.values.copy()
    test_id = test_data.member_id.values.copy()
    data = pd.concat([train_data, test_data])

    train_feat = data[data.member_id.isin(train_id)]
    test_feat = data[data.member_id.isin(test_id)]

    return train_feat, test_feat


train_feat, test_feat = make_feat(train_data, test_data)

predictors = [f for f in test_feat.columns if f not in ['acc_now_deinq']]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('0.5mse', score, False)


print('开始训练...')

params = {
    'learning_rate': 0.001,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 18,
    'subsample': 0.9,
    'min_child_sample':80,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
    'silent': False,
}


print('开始CV 5折训练...')
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle = True, random_state = 520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['acc_now_delinq'], categorical_feature=['collection_recovery_fee'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['acc_now_delinq'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round = 1000,    # 数据量太大，计算过多会导致MemoryError, 内存溢出
                    valid_sets = lgb_train2,
                    verbose_eval = 100,
                    feval = evalerror,
                    early_stopping_rounds = 100)    #此处设置的提前结束的条件值得商榷
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_feat[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_feat['acc_now_delinq'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))
submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.4f')

np.savetxt('lgb_predict.csv',np.c_[range(1,len(test_data) + 1), test_preds],
                                      delimiter = ',', header = 'member_id, acc_now_delinq',
                                      comments = '', fmt = '%d')
