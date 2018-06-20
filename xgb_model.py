# encoding: utf-8

# coding:utf-8

import os
import pandas as pd
import numpy as np
import datetime
import time

import gc     # used to solve memory error
from scipy import stats
from scipy.stats import norm, skew
from scipy.interpolate import lagrange
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import average_precision_score
from sklearn.learning_curve import learning_curve
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn import preprocessing  #  Quantitative Scaling(mostly in (-1,1), (0,1)




train_data = pd.read_csv("H:/Python_projects/game/train.csv")
test_data = pd.read_csv("H:/Python_projects/game/test.csv")


n_folds = 5
#y_train, lambda_ = stats.boxcox(train_data['acc_now_delinq'])  # 其中一个值为负数，需要进一步处理
# train_data.plot(x = "member_id", y = "collection_recovery_fee")   # 由于用户ID的数量比较多，单纯的使用会导致图中的信息混乱

term_map = {'36 months': 36, '60 months': 60}
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
sub_grade_map = {'A1': 11, 'A2': 12, 'A3': 13, 'A4': 14, 'A5': 15,
                 'B1': 21, 'B2': 22, 'B3': 23, 'B4': 24, 'B5': 25,
                 'C1': 31, 'C2': 32, 'C3': 33, 'C4': 34, 'C5': 35,
                 'D1': 41, 'D2': 42, 'D3': 43, 'D4': 44, 'D5': 45,
                 'E1': 51, 'E2': 52, 'E3': 53, 'E4': 54, 'E5': 55,
                 'F1': 61, 'F2': 62, 'F3': 63, 'F4': 64, 'F5': 65,
                 'G1': 71, 'G2': 72, 'G3': 73, 'G4': 74, 'G5': 75}

pymnt_plan_map = {'n': 0, 'y': 1}  # 原始数据中存在一处null， 将其直接按照n处理，在此处进一步更改

purpose_map = {'debt_consolidation': 1,
               'credit_card': 2,
               'major_purchase': 3,
               'other': 4 ,
               'small_business': 5,
               'home_improvement': 6,
               'renewable_energy': 7,
               'car': 8,
               'house': 9,
               'medical': 10,
               'vacation': 11,
               'moving': 12,
               'wedding': 13,
               'educational': 14,
               }

addr_state_map = {'ND': 1,
                  'NE': 2, 'WY': 2,
                  'VT': 3, 'RI': 3, 'WV': 3, 'CT': 3, 'NJ': 3, 'MS': 3,
                  'IN': 4, 'TX': 4, 'TN': 4, 'MI': 4, 'AR': 4, 'NV': 4, 'NY': 4, 'NC': 4, 'MA': 4, 'NM': 4, 'AL': 4,
                  'LA': 5, 'MT': 5, 'KY': 5, 'SD': 5, 'OH': 5, 'OK': 5, 'SC': 5, 'GA': 5, 'MD': 5, 'FL': 5, 'PA': 5, 'OR': 5, 'AZ': 5,
                  'DE': 6, 'IL': 6, 'MO': 6, 'CA': 6, 'HI': 6, 'MN': 6, 'WA': 6, 'KS': 6, 'CO': 6, 'VA': 6, 'UT': 6,
                  'WI': 7, 'DC': 7, 'ME': 7, 'NH': 7,
                  'AK': 8,
                  'ID': 9, 'IA': 9
}

emp_length_map = {'n/a': -1,
                  '<1 year': 0,
                  '1 years': 1,
                  '2 years': 2,
                  '3 years': 3,
                  '4 years': 4,
                  '5 years': 5,
                  '6 years': 6,
                  '7 years': 7,
                  '8 years': 8,
                  '9 years': 9,
                  '10 years': 10,
                  '10+ years': 11
                  }

home_ownership_map = {'MORTGAGE': -1,
                      'OWN': 1,
                      'RENT': 2,
                      'ANY': 3,
                      'NONE': 4, 'OTHER': 4}

verification_status_map = {'Not Verified': 1,
                           'Source Verified': 2,
                           'Verified': 2}

loan_status_map = {'Late (31-120 days)': -3,
                   'Late (16-30 days)': -2,
                   'n': -1,
                   'Current': -1,
                   'Default': 1,
                   'Issued': 4,
                   'In Grace Period': 3,
                   'Charged Off': 1,
                   'Fully Paid': 5,
                   'Does not meet the credit policy. Status:Fully Paid': 6,
                   'Does not meet the credit policy. Status:Charged Off': 7
                   }

initial_list_status_map = {'f': 1, 'w': 2}

application_type_map = {'INDIVIDUAL': 1,
                        'JOINT': 2}


#def rmsle_cv(model):
#    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_data.values)
#    rmse = -cross_val_score(model, t_train.values, y_train, scoring=" neg_mean_squared_error", cv=kf)
#
#    print(rmse)
#    return rmse

GBoost = GradientBoostingRegressor(n_estimators = 2000, learning_rate = 0.05,  # 将数据进行进一步调整，在使用model时学习调参
                                   max_depth = 8, max_features = 'sqrt',
                                   min_samples_leaf = 15, min_samples_split = 10,
                                   loss = 'huber', random_state = 6)

#score = rmsle_cv(GBoost)
#print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

# 时间格式的转化


# 拉格朗日插值用于补全
def lgr(df, data):
    def ploy(s, n, k = 6):
        y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
        y = y[y.notnull()]
        return lagrange(y.index, list(y))(n)

    for i in data.columns:
        for j in range(len(data)):
            if (data[i].isnull())[j]:
                data[i][j] = ploy(data[i], j)
    return data

## 时间格式化及其相关的处理有待讨论

# 对于原始数据量在 600,000 ~ 700,000 之间的缺值数据使用 RandomForset 拟合补全
def set_missing_data(df):
    for data in ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']:
        # 将已有的数值类型特征提取出来放到Random Forest Regressor 中, 在下面的列中存在多组需要补全的值，相关的已知数值类型特征可能需要重新选择以及排列
        data_df = df[['tot_coll_amt', 'int_rate','annual_inc', 'tot_cur_bal', 'revol_bal', 'total_pymnt_inv', 'total_rev_hi_lim', 'dti']]

        known_data = data_df[data_df.data.notnull()].as_matrix()
        unknown_data = data_df[data_df.data.isnull()].as_matrix()

        y = known_data[:, 0]
        X = known_data[:, 1:]

        rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
        rfr.fit(X, y)

        predictedData = rfr.predict(unknown_data[:, 1::])

        df.loc[(df.data.isnull()), data] = predictedData

        return df, rfr




# 利用不同的模型对特征进行筛选，选出较为重要的特征
def get_top_n_features(train_data_X, train_data_Y, top_n_features):
    # random forest
    rf_est = RandomForestClassifier(random_state = 42)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs = 25, cv = 10, verbose = 1)
    rf_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending = False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:10]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state = 42)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs = 25, cv = 10, verbose = 1)
    ada_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Feature from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state = 0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs = 25, cv = 10, verbose = 1)
    et_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best ET Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending = False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state = 0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs = 25, cv = 10, verbose = 1)
    gb_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending = False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state = 0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs = 25, cv = 10, verbose = 1)
    dt_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending = False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # merge the three models
    features_top_n = pd.concat(
        [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
        ignore_index = True).drop_duplicates()

    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                     feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index = True)

    return features_top_n, features_importance



# 首先需要舍弃两个因填写purpose缺失导致数据左移的id数据
#train_data = train_data.drop(train_data['member_id'] == 381665, axis = 0)
#train_data = train_data.drop(train_data['member_id'] == 8418365, axis = 0)




# train_data 部分

# 还没有处理的时间 格式 暂时舍弃
train_data = train_data.drop(['issue_d', 'earliest_cr_line'], axis = 1)
test_data = test_data.drop(['issue_d', 'earliest_cr_line'], axis = 1)
# 时间的初步转换
#train_data['issue_d'] = time.strftime("%Y-%m-%d", time.local(train_data['issue_d']))
#train_data['earliest_cr_line'] = time.strftime("%Y-%m-%d", time.local(train_data['earliest_cr_line']))
#test_data['issue_d'] = time.strftime("%Y-%m-%d", time.local(test_data['issue_d']))
#test_data['earliest_cr_line'] = time.strftime("%Y-%m-%d", time.local(test_data['earliest_cr_line']))




# 将数据量在10,000以下的feature直接舍弃
train_data = train_data.drop(['annual_inc_joint', 'dti_joint', 'verification_status_joint'], axis = 1)

# 将数据中主要问字符串叙述的feature暂时舍弃 (文本部分）
train_data = train_data.drop(['emp_title', 'desc', 'title', 'zip_code'], axis = 1)

# 将数据量为10,000 - 100,000 之间的feature暂时舍弃
train_data = train_data.drop(['open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m',
                              'mths_since_rcnt_il', 'total_bal_il', 'il_util',
                              'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
                              'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m']
                              , axis = 1)

# test_data 部分
# 将数据量在10,000以下的feature直接舍弃
test_data = test_data.drop(['annual_inc_joint', 'dti_joint', 'verification_status_joint'], axis = 1)

# 将数据中主要问字符串叙述的feature暂时舍弃 (文本部分）
test_data = test_data.drop(['emp_title', 'desc', 'title', 'zip_code'], axis = 1)

# 将数据量为10,000 - 100,000 之间的feature暂时舍弃
test_data = test_data.drop(['open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m',
                              'mths_since_rcnt_il', 'total_bal_il', 'il_util',
                              'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
                              'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m'],
                           axis = 1)

print("drop finished")




# train_data 部分
#train_data = train_data.drop(['acc_now_delinq'], axis = 1)
train_data['term'] = train_data['term'].map(term_map)
train_data['grade'] = train_data['grade'].map(grade_map)
train_data['sub_grade'] = train_data['sub_grade'].map(sub_grade_map)
train_data['pymnt_plan'] = train_data['pymnt_plan'].map(pymnt_plan_map)
train_data['purpose'] = train_data['purpose'].map(purpose_map)
train_data['addr_state'] = train_data['addr_state'].map(addr_state_map)
train_data['emp_length'] = train_data['emp_length'].map(emp_length_map)
train_data['home_ownership'] = train_data['home_ownership'].map(home_ownership_map)
train_data['verification_status'] = train_data['verification_status'].map(verification_status_map)
train_data['loan_status'] = train_data['loan_status'].map(loan_status_map)
train_data['initial_list_status'] = train_data['initial_list_status'].map(initial_list_status_map)
train_data['application_type'] = train_data['application_type'].map(application_type_map)

# test_data 部分
#t_test = test_data
test_data['term'] = test_data['term'].map(term_map)
test_data['grade'] = test_data['grade'].map(grade_map)
test_data['sub_grade'] = test_data['sub_grade'].map(sub_grade_map)
test_data['pymnt_plan'] = test_data['pymnt_plan'].map(pymnt_plan_map)
test_data['purpose'] = test_data['purpose'].map(purpose_map)
test_data['addr_state'] = test_data['addr_state'].map(addr_state_map)
test_data['emp_length'] = test_data['emp_length'].map(emp_length_map)
test_data['home_ownership'] = test_data['home_ownership'].map(home_ownership_map)
test_data['verification_status'] = test_data['verification_status'].map(verification_status_map)
test_data['loan_status'] = test_data['loan_status'].map(loan_status_map)
test_data['initial_list_status'] = test_data['initial_list_status'].map(initial_list_status_map)
test_data['application_type'] = test_data['application_type'].map(application_type_map)

print("map finished")


# train_data 部分
# 对于数据缺失较少，同时上下关联并不紧密的原始数据进行当列的均值填充（原始数据的量在700,000以上）
train_data['annual_inc'].fillna(train_data['annual_inc'].median(axis=0), inplace=True)
train_data['revol_bal'].fillna(train_data['revol_bal'].median(axis=0), inplace=True)
# train_data['earliest_cr_line'].fillna(train_data['earliest_cr_line'].median(axis = 0), inplace = True)
train_data['pub_rec'].fillna(train_data['pub_rec'].median(axis=0), inplace=True)
train_data['revol_util'].fillna(train_data['revol_util'].median(axis=0), inplace=True)
train_data['total_acc'].fillna(train_data['total_acc'].median(axis=0), inplace=True)
train_data['emp_length'].fillna(train_data['emp_length'].median(axis=0), inplace=True)
train_data['collections_12_mths_ex_med'].fillna(train_data['collections_12_mths_ex_med'].median(axis=0),
                                                    inplace=True)

#---------------------------------------------------------------------------------------------------------
# 原始数据量在 110,000, 177,424数据的使用均值填补
train_data['mths_since_last_record'].fillna(train_data['mths_since_last_record'].median(axis = 0), inplace = True)
train_data['mths_since_last_major_derog'].fillna(train_data['mths_since_last_major_derog'].median(axis = 0), inplace = True)


#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# 暂时将下属述数据量在650,000左右的使用均值填充，相关的设置来节约计算的时间

train_data['tot_coll_amt'].fillna(train_data['tot_coll_amt'].median(axis=0), inplace=True)
train_data['tot_cur_bal'].fillna(train_data['tot_cur_bal'].median(axis=0), inplace=True)
train_data['total_rev_hi_lim'].fillna(train_data['total_rev_hi_lim'].median(axis=0), inplace=True)

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------


# test_data 部分
test_data['annual_inc'].fillna(test_data['annual_inc'].median(axis=0), inplace=True)
test_data['revol_bal'].fillna(test_data['revol_bal'].median(axis=0), inplace=True)
# test_data['earliest_cr_line'].fillna(test_data['earliest_cr_line'].median(axis = 0), inplace = True)
test_data['pub_rec'].fillna(test_data['pub_rec'].median(axis=0), inplace=True)
test_data['revol_util'].fillna(test_data['revol_util'].median(axis=0), inplace=True)
test_data['total_acc'].fillna(test_data['total_acc'].median(axis=0), inplace=True)
test_data['emp_length'].fillna(test_data['emp_length'].median(axis=0), inplace=True)
test_data['collections_12_mths_ex_med'].fillna(test_data['collections_12_mths_ex_med'].median(axis=0), inplace=True)


#---------------------------------------------------------------------------------------------------------
# 原始数据量在 110,000, 177,424数据的使用均值填补
test_data['mths_since_last_record'].fillna(test_data['mths_since_last_record'].median(axis = 0), inplace = True)
test_data['mths_since_last_major_derog'].fillna(test_data['mths_since_last_major_derog'].median(axis = 0), inplace = True)


#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# 暂时将下属述数据量在650,000左右的使用均值填充，相关的设置来节约计算的时间

test_data['tot_coll_amt'].fillna(test_data['tot_coll_amt'].median(axis=0), inplace=True)
test_data['tot_cur_bal'].fillna(test_data['tot_cur_bal'].median(axis=0), inplace=True)
test_data['total_rev_hi_lim'].fillna(test_data['total_rev_hi_lim'].median(axis=0), inplace=True)

print("fillna finished")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------



#train_data, rfr1 = set_missing_data(train_data)
#test_data, rfr2 = set_missing_data(test_data)

# 初步使用，后期： 1. 将数据缺失情况较为严重的feature暂时舍弃，对于其他情况的使用拟合以及拉格朗日插值以及使用其他进行替代
#t_train = t_train.fillna(0)



#for data in ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']:

"""
# 对剩下存在缺失值的所有数据做补全
def ploy(s, n, k=6):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)


for i in train_data.columns:
    for j in range(len(train_data)):
        if (train_data[i].isnull())[j]:
            train_data[i][j] = -999
#train_data.to_csv('../game/train.csv')


for i in test_data.columns:
    for j in range(len(test_data)):
        if (test_data[i].isnull())[j]:
            test_data[i][j] = -999
#test_data.to_csv('../game/test.csv')
"""

#方差选择法，返回值为特征选择后的数据
# 参数 threshold为方差的阈值
#VarianceThreshold(threshold = 3).fit_transform(train_data)

# cast the data, for some of the values are float or integer and some object
for f in train_data.columns:
    if train_data[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_data[f].values))
        train_data[f] = lbl.transform(list(train_data[f].values))

print("train_data cast finished")

for f in test_data.columns:
    if test_data[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(test_data[f].values))
        test_data[f] = lbl.transform(list(test_data[f].values))

print("test_data cast finished")


train_data.fillna((-999), inplace=True)
test_data.fillna((-999), inplace=True)



#train_data = np.array(train_data)
#test_data = np.array(test_data)
#train_data = train_data.astype(float)
#test_data = test_data.astype(float)

# combined all the test data and train data
#test_data['acc_now_delinq'] = 0
#combined_train_test = train_data.append(test_data)

train_data_X = train_data.drop(['acc_now_delinq'], axis = 1)
train_data_Y = train_data['acc_now_delinq']
test_data_X = test_data #.drop(['acc_now_delinq'], axis = 0)


feature_to_pick = 100     # needed to be changed while more feature found here
feature_top_n = get_top_n_features(train_data_X, train_data_Y, feature_to_pick)
train_data_X = train_data_X[feature_top_n]
test_data_X = test_data_X[feature_top_n]

print("top feature is selected")

# 回收内存
#del feature_top_n
#gc.collect()






from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
#import lightgbm as lgb
import xgboost as xgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

# 记录程序运行时间
start_time = time.time()

print("starting modeling.py")


def make_feat(train_data, test_data):
    train_id = train_data.member_id.values.copy()
    test_id = test_data.member_id.values.copy()
    data = pd.concat([train_data, test_data])

    train_feat = data[data.member_id.isin(train_id)]
    test_feat = data[data.member_id.isin(test_id)]

    return train_feat, test_feat


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('0.5mse', score, False)

# 数据标准化
#trian_data = StandardScaler().fit_transform(train_data)
#test_data = StandardScaler().fit_transform(test_data)

#  划分数据集
#  用sklearn.cross_validation进行训练数据集划分  7：3
train_xy,val = train_test_split(train_data, test_size = 0.3, random_state = 1)

#train_xy = train_xy.tolist()
#val = val.tolist()

y = train_xy.acc_now_delinq
X = train_xy.drop(['acc_now_delinq'], axis = 1)
val_y = val.acc_now_delinq
val_X = val.drop(['acc_now_delinq'], axis = 1)

target = 'acc_now_delinq'
IDcol = 'member_id'

#xgb矩阵赋值
xgb_val = xgb.DMatrix(val_X, label = val_y)
xgb_train = xgb.DMatrix(X, label = y)
xgb_test = xgb.DMatrix(test_data)

#   xgboost模型
params_xgb = {
        'booster':'gbtree',
        'objective':'multi:softmax',
        'num_calss': 2,   # 定义二分类，返回值为类别，使用binary:logistic 二分类时，返回的是预测的概率
        'n_estimators': 100,
        'gamma':2,
        'max_depth': 6,  # 使用的默认值，主要用来避免过拟合，值越大，模型会学到更具体更局部的样本
        'subsample':0.8,
        'colsample_bytree':0.8,
        'min_child_weight':2,
        'reg_alpha':0.005,
        'eta':0.1,
        'seed':0,
        #'eval_metric':'auc'
        }

plst = list(params_xgb.items())
num_rounds = 2000 #迭代次数
watchlist = [(xgb_train,'train'),(xgb_val,'val')]

#训练模型并保存
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds = 100)
model.save_model('H:/Python_projects/game/xgb.model')
print("best best_ntree_limit",model.best_ntree_limit)

#预测并保存
xgb_preds = model.predict(xgb_test, ntree_limit = model.best_ntree_limit)
np.savetxt('submission_xgb.csv',np.c_[range(1, len(test_data)+1), xgb_preds],
                                      delimiter = ',', header = 'member_id, acc_now_delinq',
                                      comments = '', fmt = '%d')

print("xgb success!")

"""
train_feat, test_feat = make_feat(train_data, test_data)

predictors = [f for f in test_feat.columns if f not in ['acc_now_deinq']]

print('开始训练...')

params_lgb = {
    'learning_rate': 0.001,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 18,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}


print('开始CV 5折训练...')
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))                                  #  该处的test_preds 将等效为lgb_preds来使用
kf = KFold(len(train_feat), n_folds = 5, shuffle = True, random_state = 520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['acc_now_delinq'], categorical_feature=['collection_recovery_fee'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['acc_now_delinq'])
    gbm = lgb.train(params_lgb,
                    lgb_train1,
                    num_boost_round = 1000,    # 数据量太大，计算过多会导致MemoryError, 内存溢出
                    valid_sets = lgb_train2,
                    verbose_eval = 100,
                    feval = evalerror,
                    early_stopping_rounds = 100)
    feat_imp = pd.Series(gbm.feature_importance(), index = predictors).sort_values(ascending = False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_feat[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_feat['acc_now_delinq'], train_preds) * 0.5))
#print('CV训练用时{}秒'.format(time.time() - t0))
submission = pd.DataFrame({'pred': test_preds.mean(axis = 1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header = None,
                  index = False, float_format = '%.4f')

np.savetxt('submission_lgb.csv',np.c_[range(1,len(test_data) + 1), test_preds],
                                      delimiter = ',', header = 'member_id, acc_now_delinq',
                                      comments = '', fmt = '%d')

print("lgb success")
"""

"""
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data_X, train_data_Y)
Y_pred_DTC = decision_tree.predict(test_data_X)
acc_decision_tree = round(decision_tree.score(train_data_X, train_data_Y) * 100, 2)
print("DecisionTreeClassifier finished")

submission = pd.DataFrame({
        "member_id": test_data["member_id"],
        "acc_now_delinq": Y_pred_DTC
    })
submission.to_csv('submission_DTC.csv', index=False)



# Random Forest
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(train_data_X, train_data_Y)
Y_pred_RFC = random_forest.predict(test_data_X)
random_forest.score(train_data_X, train_data_Y)
acc_random_forest = round(random_forest.score(train_data_X, train_data_Y) * 100, 2)
print("RandomForestClassifier finished")

submission = pd.DataFrame({
        "member_id": test_data["member_id"],
        "acc_now_delinq": Y_pred_RFC
    })
submission.to_csv('submission_RFC.csv', index=False)



# Random Forest 部分的调参

from sklearn.model_selection import StratifiedKFold, GridSearchCV


# 根据一下输出的数据，得出 下一部分程序中的参数值
forest = RandomForestClassifier(max_features='sqrt')
parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [100,200,250,500],
                 'criterion': ['gini','entropy']
                 }
cross_validation = StratifiedKFold(train_data_Y, n_folds=5)
grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(train_data_X, train_data_Y)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# 参照上一部分的程序输出结果
random_forest = RandomForestClassifier(n_estimators=250, max_depth=5, criterion='gini')
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)


submission = pd.DataFrame({
        "member_id": test_data["member_id"],
        "acc_now_delinq": Y_pred
    })
submission.to_csv('submission.csv', index=False)
"""
#输出运行时长
cost_time = time.time() - start_time
print("cost time :",cost_time,"(s).......")