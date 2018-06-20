# coding: utf-8

from featurepick import *
from xgb_model import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlens.visualization import corrmat

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler

from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


SEED = 300
np.random.seed(SEED)


def get_train_test(test_size = 0.95):

    y = 1 * (train_data.acc_now_delinq)
    X = train_data.drop(["acc_now_delinq"], axis = 1)
    X = pd.get_dummies(X, sparse = True)
    X.drop(X.columns[X.std() == 0], axis = 1, inplace = True)
    return train_test_split(X, y, test_size = test_size, random_state = SEED)


xtrain, xtest, ytrain, ytest = get_train_test()


rf = RandomForestClassifier(
    n_estimators = 10,
    max_features = 10,
    random_state = SEED
)

rf.fit(xtrain, ytrain)
p = rf.predict_proba(xtest)[:, 1]
print("Average of decision tree ROC-AUC score: %.3f" % roc_auc_score(ytest, p))


# add those features behind to this list and get the the plot and more accurate model
def get_models():
    """Generate a library of base learners."""
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)

    models = {
              'random forest': rf,
              'gbm': gb,

              }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((ytest.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(xtrain, ytrain)
        P.iloc[:, i] = m.predict_proba(xtest)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

models = get_models()
P = train_predict(models)
score_models(P, ytest)

corrmat(P.corr(), inflate = False)
plt.show()     # only shown to choose the model, will be deleted later

print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(ytest, P.mean(axis = 1)))


# used to choose the best model and generate a roc_curve plot
def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    cm = [plt.cm.rainbow(i)
          for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]

    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

    fpr, tpr, _ = roc_curve(ytest, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon = False)
    plt.show()


#plot_roc_curve(ytest, P.values, P.mean(axis = 1), list(P.columns), "ensemble")



# 利用VotingClassifer建立最终的预测模型
rf_est = RandomForestClassifier(n_estimators = 750, criterion = 'gini', max_features = 'sqrt',
                                             max_depth = 3, min_samples_split = 4, min_samples_leaf = 2,
                                             n_jobs = 50, random_state = 42, verbose = 1)
dtc_est = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                 max_features=7, max_leaf_nodes=None,
                                 min_impurity_split=0.005, min_samples_leaf=3,
                                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                                 presort=False, random_state=1, splitter='random')

voting_est = VotingClassifier(estimators = [('rf', rf_est),('dtc', dtc_est)],
                                       voting = 'soft', weights = [3, 7],
                                       n_jobs = 50)
voting_est.fit(train_data_X, train_data_Y)

test_data_X['acc_now_delinq'] = voting_est.predict(test_data_X)



final_submission = pd.DataFrame({'member_id': test_data.loc[: 'member_id'],
                                 'acc_now_delinq': test_data_X.loc[: 'acc_now_delinq']})

final_submission.to_csv('final_submission_result.csv', index = False, sep = ',')

