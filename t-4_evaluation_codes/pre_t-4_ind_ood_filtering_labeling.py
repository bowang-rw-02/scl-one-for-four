# This is the data preparation code file for T-4 continual learning
# It applies a Mahalanobis distance-based method to filter IND/OOD examples from Test-I set and gives them labels
# for the final model continual learning in T-4


import numpy as np
from scipy.spatial import distance



NETWORK_DIM = 768
NUM_OF_IND_CATES = 112
NUM_PER_CATE = 80
THRESHOLD = np.load('./data/clinc/best_auroc_fpr90thres.npy')[1]


ind_ref_features = np.load('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ind_train_small_sentvec.npy')
ind_samples_label = np.load('./data/clinc/new_division_data/new_ind_label_train_small.npy')
ind_samples_label_mapping = []
for label in ind_samples_label:
    if label not in ind_samples_label_mapping:
        ind_samples_label_mapping.append(label)



test_ind_text = np.load('./data/clinc/new_division_data/new_ind_text_test.npy', allow_pickle=True)
test_ood_text = np.load('./data/clinc/new_division_data/new_ood_text_test.npy', allow_pickle=True)
test_text = np.concatenate((test_ind_text,test_ood_text), axis=0)

test_features = np.load('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ind_ood_test_sentvec.npy')
NUM_OF_TEST_SAMPLES = len(test_features)

test_labels1 = np.load('./data/clinc/new_division_data/new_ind_label_test.npy')
test_labels2 = np.load('./data/clinc/new_division_data/new_ood_label_test.npy')
test_labels = np.concatenate((test_labels1, test_labels2))


Y = np.load('./data/clinc/new_division_data/new_ind_zero_ood_one_label_test.npy')



print('Pre T-4: Data preparation for T-4 New Intent Discovery start!')

c_avg_present = []
c_cov = np.zeros([NETWORK_DIM, NETWORK_DIM])

for i in range(NUM_OF_IND_CATES):
    cate_vec = ind_ref_features[0 + i * NUM_PER_CATE:NUM_PER_CATE + i * NUM_PER_CATE]
    cate_cov = np.cov(cate_vec.T)
    # centroids of each category
    c_avg_present.append(cate_vec.mean(axis=0))
    # sum of covs of each category
    c_cov = c_cov+cate_cov

c_avg_present = np.array(c_avg_present)
c_cov_avged = c_cov/NUM_OF_IND_CATES

iV = np.linalg.pinv(c_cov_avged)

test_distance = np.zeros(shape=NUM_OF_TEST_SAMPLES)
predictions = []

for i in range(NUM_OF_TEST_SAMPLES):
    min_distance = 1000
    min_cate = -1
    for j in range(NUM_OF_IND_CATES):
        distance_now = distance.mahalanobis(test_features[i], c_avg_present[j], iV)
        if distance_now<min_distance:
            min_distance = distance_now
            min_cate = ind_samples_label_mapping[j]
        if i%500 == 0 and j%200 == 0:
            print(f'Finished {i}/{NUM_OF_TEST_SAMPLES} points calculation.')
    test_distance[i] = min_distance
    predictions.append(min_cate)


detected_ood_texts = []
detected_ood_labels = []

detected_ind_texts = []
detected_ind_labels = []

for i, final_dist in enumerate(test_distance):
    if final_dist > THRESHOLD and test_labels[i] > 111:
        detected_ood_texts.append(test_text[i])
        detected_ood_labels.append(test_labels[i])
    if final_dist <= THRESHOLD:
        detected_ind_texts.append(test_text[i])
        detected_ind_labels.append(predictions[i])


np.save('./data/clinc/detected_data_for_t4/test1_detected_ood_text.npy', detected_ood_texts)
np.save('./data/clinc/detected_data_for_t4/test1_detected_ood_label.npy', detected_ood_labels)

np.save('./data/clinc/detected_data_for_t4/test1_detected_ind_text.npy', detected_ind_texts)
np.save('./data/clinc/detected_data_for_t4/test1_detected_ind_label.npy', detected_ind_labels)

print('Pre T-4: Test1 IND/OOD detection and data preparation complete.')
