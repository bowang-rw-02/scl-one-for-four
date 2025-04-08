# This is the code file for T-2 OOD detection evaluation
# It is also realized by the centroid-Mahalanobis distance-based method
# the distances between the representations of test points and centroids of each IND category are calculated
# if the shortest distance is still far (compared to a threshold), then it is more likely to be an OOD input,
# and vice versa.
# However, since different threshold will result in different accuracies, we apply four threshold-free metrics:
# AUROC, AUPR, FPR95/90


import numpy as np
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc,precision_recall_curve


# For clinc training-small dataset, it now has 112 IND categories * 80 records each category
NETWORK_DIM = 768
NUM_OF_IND_CATES = 112
USED_IND_DATA_PER_CATE = 80


ind_ref_features = np.load('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ind_train_small_sentvec.npy')

test_features = np.load('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ind_ood_test_sentvec.npy')
Y = np.load('./data/clinc/new_division_data/new_ind_zero_ood_one_label_test.npy')
NUM_OF_TEST_SAMPLES = len(test_features)


c_avg_present = []
c_cov = np.zeros([NETWORK_DIM, NETWORK_DIM])


for i in range(NUM_OF_IND_CATES):
    cate_vec = ind_ref_features[0 + i * USED_IND_DATA_PER_CATE:USED_IND_DATA_PER_CATE + i * USED_IND_DATA_PER_CATE]
    cate_cov = np.cov(cate_vec.T)
    # centroids of each category
    c_avg_present.append(cate_vec.mean(axis=0))
    # sum of covs of each category
    c_cov = c_cov+cate_cov

c_avg_present = np.array(c_avg_present)
c_cov_avged = c_cov/NUM_OF_IND_CATES

iV = np.linalg.pinv(c_cov_avged)

test_distance = np.zeros(shape=NUM_OF_TEST_SAMPLES)

for i in range(NUM_OF_TEST_SAMPLES):
    min_distance = 1000
    for j in range(NUM_OF_IND_CATES):
        distance_now = distance.mahalanobis(test_features[i], c_avg_present[j], iV)
        if distance_now<min_distance:
            min_distance = distance_now
        if i%500 == 0 and j%200 == 0:
            print(f'Finished {i}/{NUM_OF_TEST_SAMPLES} points calculation.')
    test_distance[i] = min_distance

pred_y_P = test_distance

# Calculate the precision, recall and the final AUPR
precision, recall, thresholds = precision_recall_curve(Y, pred_y_P)
PR_auc = auc(recall, precision)

# Calculate FPR, TPR, and AUROC
fpr, tpr, thresholds = roc_curve(Y, pred_y_P)
fpr95 = 1
fpr90 = 1
for ffpr, ttpr in zip(fpr, tpr):
    if abs(ttpr - 0.95) < 0.01:
        fpr95 = ffpr
        break
for ffpr, ttpr in zip(fpr, tpr):
    if abs(ttpr - 0.90) < 0.01:
        fpr90 = ffpr
        break


roc_auc = auc(fpr, tpr)


print('For T-2 OOD detection, the results of this model are:')
print('AUROC: ', roc_auc, '.')
print('AUPR: ', PR_auc, '.')
print('fpr95: ', fpr95, ' . fpr90: ', fpr90, ' .')