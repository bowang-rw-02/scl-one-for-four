# This code file further split 20% from the new-built training-set to build Test-II set (for T-4 and prevent data leakage).
# The remaining training data is called ‘IND-training-small’.


import numpy as np


# Load new-built training data
texts = np.load('./data/clinc/new_division_data/new_ind_text_train.npy', allow_pickle=True)
labels = np.load('./data/clinc/new_division_data/new_ind_label_train.npy')


# Collect data for each category
from collections import defaultdict
data_dict = defaultdict(list)
for text, label in zip(texts, labels):
    data_dict[label].append(text)


train_texts, train_labels = [], []
test_texts, test_labels = [], []

# Take 20 records out of 100 records from each category to Test-II dataset
# The remaining 80 record/per category are saved as training set-small
for label, texts in data_dict.items():
    np.random.shuffle(texts)
    test_texts.extend(texts[:20])
    train_texts.extend(texts[20:])
    test_labels.extend([label] * 20)
    train_labels.extend([label] * (len(texts) - 20))


train_texts = np.array(train_texts)
train_labels = np.array(train_labels)
test_texts = np.array(test_texts)
test_labels = np.array(test_labels)

# Save the training-small set and Test-II set
np.save('./data/clinc/new_division_data/new_ind_text_train_small.npy', train_texts)
np.save('./data/clinc/new_division_data/new_ind_label_train_small.npy', train_labels)
np.save('./data/clinc/new_division_data/new_ind_text_test2.npy', test_texts)
np.save('./data/clinc/new_division_data/new_ind_label_test2.npy', test_labels)

print('The IND training data have been successfully divided into smaller training set (80%) and test2 set (20%)')

###################################################
# Need to conduct again to the OOD part
texts = np.load('./data/clinc/new_division_data/new_ood_text_train.npy', allow_pickle=True)
labels = np.load('./data/clinc/new_division_data/new_ood_label_train.npy')

from collections import defaultdict
data_dict = defaultdict(list)
for text, label in zip(texts, labels):
    data_dict[label].append(text)

train_texts, train_labels = [], []
test_texts, test_labels = [], []

for label, texts in data_dict.items():
    np.random.shuffle(texts)  # 随机打乱数据
    test_texts.extend(texts[:20])
    train_texts.extend(texts[20:])
    test_labels.extend([label] * 20)
    train_labels.extend([label] * (len(texts) - 20))

train_texts = np.array(train_texts)
train_labels = np.array(train_labels)
test_texts = np.array(test_texts)
test_labels = np.array(test_labels)

np.save('./data/clinc/new_division_data/new_ood_text_train_small.npy', train_texts)
np.save('./data/clinc/new_division_data/new_ood_label_train_small.npy', train_labels)
np.save('./data/clinc/new_division_data/new_ood_text_test2.npy', test_texts)
np.save('./data/clinc/new_division_data/new_ood_label_test2.npy', test_labels)

print('The OOD training data have been successfully divided into smaller training set (80%) and test2 set (20%)')