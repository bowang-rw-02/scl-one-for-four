# This is the code file for T-3 new intent discovery and evaluation
# It applies a simple KMeans algorithm to find new categories from OOD Test-I, even without training by data of that part

# Notes:
# 1. Here, for a fair comparison, the proposed method and the baselines perform clustering on the original, full OOD test set.
# The K is also set as real number existed in OOD Test-I set, which is the same as the former baseline papers did.
# 2. However, when conducting T-4 - continual learning training, only the detected and filtered IND and OOD will be used
# for T-4 final model training

import numpy as np
import sys
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

np.set_printoptions(threshold=sys.maxsize)

# Hungary alignment algorithm to pair the most possible situations of real label and clustered label
def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    return {'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred) * 100, 2),
            'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2)
            }


OOD_test_features = np.load('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ood_test_sentvec.npy')
OOD_test_labels = np.load('./data/clinc/new_division_data/new_ood_label_test.npy')


NUM_OF_IND_CATES = 112
NUM_OF_OOD_CATES = 150 - NUM_OF_IND_CATES

print('Task-3 New Intent Discovery - evaluation starts!')
print('Running K-Means...')
# k-means clustering
km = KMeans(n_clusters=NUM_OF_OOD_CATES).fit(OOD_test_features)

y_pred = km.labels_
count = Counter(y_pred)
count_l = len(count)
print(f'We have discovered {count_l} categories, the details are: {count}')

y_true = OOD_test_labels

results = clustering_score(y_true, y_pred)

# Confusion matrix
ind, _ = hungray_aligment(y_true, y_pred)
map_ = {i[0]: i[1] for i in ind}
print('The mapped pred y to 112-149 labels (final pred result) is:')
y_pred = np.array([map_[idx] for idx in y_pred])

cm = confusion_matrix(y_true, y_pred)
print('confusion matrix')
print(cm)

print('The final results are:', results)