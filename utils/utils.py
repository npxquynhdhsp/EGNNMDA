# %%
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# %%
def calculate_score(true_list, predict_list):
    auc_list = []
    auprc_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for index, (true_fold, predict_fold) in enumerate(zip(true_list, predict_list)):
        auc = metrics.roc_auc_score(true_fold, predict_fold)
        precision, recall, thresholds = metrics.precision_recall_curve(true_fold, predict_fold)
        auprc = metrics.auc(recall, precision)
        result_fold = [0 if j < 0.5 else 1 for j in predict_fold]
        accuracy = metrics.accuracy_score(true_fold, result_fold)
        precision = metrics.precision_score(true_fold, result_fold)
        recall = metrics.recall_score(true_fold, result_fold)
        f1 = metrics.f1_score(true_fold, result_fold)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)
        auprc_list.append(auprc)
    print('AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_list), np.std(auc_list)),
          'AUPRC mean: %.4f, variance: %.4f \n' % (np.mean(auprc_list), np.std(auprc_list)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(accuracy_list), np.std(accuracy_list)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(precision_list), np.std(precision_list)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_list), np.std(recall_list)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_list), np.std(f1_list)),
          sep="")

    return np.mean(auc_list), np.mean(auprc_list), np.mean(accuracy_list), \
        np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

