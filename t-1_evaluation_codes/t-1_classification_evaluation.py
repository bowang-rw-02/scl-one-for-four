# This is the code file for T-1 (new) IND intent classification evaluation
# It realizes a centroid-Mahalanobis distance-based, KNN, K=1 easy classification
# the distances between the representations of test points and centroids of each IND category are calculated
# and the label is the label of test points' nearest category


import os
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, f1_score



# For clinc training-small dataset, it now has 112 IND categories * 80 records each category
NUM_OF_IND_CATES = 112
NUM_PER_CATE = 80

ind_ref_features = np.load('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ind_train_small_sentvec.npy')
ind_samples_label = np.load('./data/clinc/new_division_data/new_ind_label_train_small.npy')

test_features = np.load('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ind_test_sentvec.npy')
test_labels = np.load('./data/clinc/new_division_data/new_ind_label_test.npy')


ind_samples_label_mapping = []
for label in ind_samples_label:
    if label not in ind_samples_label_mapping:
        ind_samples_label_mapping.append(label)


ind_cate_num = np.ones(NUM_OF_IND_CATES)*NUM_PER_CATE
ind_cate_num = ind_cate_num.astype(np.int32)


ind_accu_cate_num = [0]
for i in range(NUM_OF_IND_CATES):
    ind_accu_cate_num.append((ind_accu_cate_num[i]+ind_cate_num[i]))


NUM_OF_TEST_DATA = test_features.shape[0]
USED_NETWORK_DIM = test_features.shape[1]


# Start calculation and judging using Mahalanobis distance
c_avg_present = []
c_cov = np.zeros([USED_NETWORK_DIM, USED_NETWORK_DIM])


for i in range(NUM_OF_IND_CATES):
    cate_vec = ind_ref_features[ind_accu_cate_num[i]: ind_accu_cate_num[i + 1]]
    cate_cov = np.cov(cate_vec.T)
    # centroids of each category
    c_avg_present.append(cate_vec.mean(axis=0))
    # sum of covs of each category
    c_cov = c_cov+cate_cov

c_avg_present = np.array(c_avg_present)
c_cov_avged = c_cov/NUM_OF_IND_CATES
iV = np.linalg.pinv(c_cov_avged)


test_distance = []
predictions = []

for i in range(NUM_OF_TEST_DATA):
    min_distance = 1000
    min_cate = -1
    for j in range(NUM_OF_IND_CATES):
        distance_now = distance.mahalanobis(test_features[i],c_avg_present[j], iV)
        if distance_now < min_distance:
            min_distance = distance_now
            min_cate = ind_samples_label_mapping[j]
    if i%500 == 0:
        print(f'Finished {i}/{NUM_OF_TEST_DATA} points distance calculation')
    test_distance.append(min_distance)
    predictions.append(min_cate)



# Calculate accuracy
micro_f1 = f1_score(test_labels, predictions, average='micro')
macro_f1 = f1_score(test_labels, predictions, average='macro')
weighted_f1 = f1_score(test_labels, predictions, average='weighted')

# Print the metrics
print('For Task-1 IND classification, the results are:')
print(f"Micro F1 Score: {micro_f1}")
print(f"Macro F1 Score: {macro_f1}")
print(f"Weighted F1 Score: {weighted_f1}")

