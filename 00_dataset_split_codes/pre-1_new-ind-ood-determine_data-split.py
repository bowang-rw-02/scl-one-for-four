# This is the code file splitting original 150 categories data of CLINC into new IND (112 categories)
# and new OOD (38 categories)

import os
import numpy as np
import pandas as pd


np.random.seed(42)  # For reproducibility
num_total_classes = 150  # Original number of IND classes


# Split the classes into new IND and new OOD categories (150*0.75=112 IND categories, 38 OOD categories)
num_new_IND_classes = 112
num_new_OOD_classes = num_total_classes - num_new_IND_classes

# Randomly select new IND class ids
new_IND_labels = np.random.choice(range(num_total_classes), num_new_IND_classes, replace=False)
new_OOD_labels = np.setdiff1d(range(num_total_classes), new_IND_labels)
print(f'The new divided IND categories are: {new_IND_labels}')
print(f'The new divided IND categories are: {new_OOD_labels}')


def process_and_split_data(df_train, df_val, df_test, new_IND_labels, new_OOD_labels):
    # Mapping the old ids-new ids
    new_IND_mapping = {old_label: new_label for new_label, old_label in enumerate(new_IND_labels)}
    new_OOD_mapping = {old_label: new_label for new_label, old_label in enumerate(new_OOD_labels, start=len(new_IND_mapping))}

    def map_labels(row):
        if row['label'] in new_IND_mapping:
            return new_IND_mapping[row['label']]
        elif row['label'] in new_OOD_mapping:
            return new_OOD_mapping[row['label']]
        return -1

    # Apply label mapping
    df_train['new_label'] = df_train['label'].apply(lambda x: map_labels({'label': x}))
    df_val['new_label'] = df_val['label'].apply(lambda x: map_labels({'label': x}))
    df_test['new_label'] = df_test['label'].apply(lambda x: map_labels({'label': x}))

    # Divide the dataset into new IND, new OOD datasets
    new_ind_df_train = df_train[df_train['label'].isin(new_IND_labels)].copy()
    new_ood_df_train = df_train[df_train['label'].isin(new_OOD_labels)].copy()
    new_ind_df_val = df_val[df_val['label'].isin(new_IND_labels)].copy()
    new_ood_df_val = df_val[df_val['label'].isin(new_OOD_labels)].copy()
    new_ind_df_test = df_test[df_test['label'].isin(new_IND_labels)].copy()
    new_ood_df_test = df_test[df_test['label'].isin(new_OOD_labels)].copy()

    # Change label to new one
    new_ind_df_train['label'] = new_ind_df_train['new_label']
    new_ood_df_train['label'] = new_ood_df_train['new_label']
    new_ind_df_val['label'] = new_ind_df_val['new_label']
    new_ood_df_val['label'] = new_ood_df_val['new_label']
    new_ind_df_test['label'] = new_ind_df_test['new_label']
    new_ood_df_test['label'] = new_ood_df_test['new_label']

    # Delete temp data
    new_ind_df_train.drop(columns='new_label', inplace=True)
    new_ood_df_train.drop(columns='new_label', inplace=True)
    new_ind_df_val.drop(columns='new_label', inplace=True)
    new_ood_df_val.drop(columns='new_label', inplace=True)
    new_ind_df_test.drop(columns='new_label', inplace=True)
    new_ood_df_test.drop(columns='new_label', inplace=True)

    return new_ind_df_train, new_ind_df_val, new_ind_df_test, new_ood_df_train, new_ood_df_val, new_ood_df_test, new_IND_mapping, new_OOD_mapping



# Specify the paths of old (original) IND data
old_data_path = './data/clinc/original_data'
old_ind_text_train_path = os.path.join(old_data_path, 'original_ind_text_train.npy')
old_ind_label_train_path = os.path.join(old_data_path, 'original_ind_label_train.npy')
old_ind_text_val_path = os.path.join(old_data_path, 'original_ind_text_val.npy')
old_ind_label_val_path = os.path.join(old_data_path, 'original_ind_label_val.npy')
old_ind_text_test_path = os.path.join(old_data_path, 'original_ind_text_test.npy')
old_ind_label_test_path = os.path.join(old_data_path, 'original_ind_label_test.npy')

# Load the original IND data
old_ind_text_train = np.load(old_ind_text_train_path, allow_pickle=True)
old_ind_label_train = np.load(old_ind_label_train_path)
old_ind_text_val = np.load(old_ind_text_val_path, allow_pickle=True)
old_ind_label_val = np.load(old_ind_label_val_path)
old_ind_text_test = np.load(old_ind_text_test_path, allow_pickle=True)
old_ind_label_test = np.load(old_ind_label_test_path)

# Convert to DataFrame for easy manipulation
df_train = pd.DataFrame({'text': old_ind_text_train, 'label': old_ind_label_train})
df_val = pd.DataFrame({'text': old_ind_text_val, 'label': old_ind_label_val})
df_test = pd.DataFrame({'text': old_ind_text_test, 'label': old_ind_label_test})

# Process the train, val and test dataframes
new_ind_df_train, new_ind_df_val, new_ind_df_test, new_ood_df_train, new_ood_df_val, new_ood_df_test, \
new_ind_mapping, new_ood_mapping = process_and_split_data(df_train, df_val, df_test, new_IND_labels, new_OOD_labels)


###################################################
# Data saving

# Define new data saving path
new_data_path = './data/clinc/new_division_data'
if not os.path.exists(new_data_path):
    os.makedirs(new_data_path)

# Save the new IND and OOD datasets into separate files
np.save(os.path.join(new_data_path, 'new_ind_text_train.npy'), new_ind_df_train['text'].values)
np.save(os.path.join(new_data_path, 'new_ind_label_train.npy'), new_ind_df_train['label'].values)
np.save(os.path.join(new_data_path, 'new_ind_text_val.npy'), new_ind_df_val['text'].values)
np.save(os.path.join(new_data_path, 'new_ind_label_val.npy'), new_ind_df_val['label'].values)
np.save(os.path.join(new_data_path, 'new_ind_text_test.npy'), new_ind_df_test['text'].values)
np.save(os.path.join(new_data_path, 'new_ind_label_test.npy'), new_ind_df_test['label'].values)

np.save(os.path.join(new_data_path, 'new_ood_text_train.npy'), new_ood_df_train['text'].values)
np.save(os.path.join(new_data_path, 'new_ood_label_train.npy'), new_ood_df_train['label'].values)
np.save(os.path.join(new_data_path, 'new_ood_text_val.npy'), new_ood_df_val['text'].values)
np.save(os.path.join(new_data_path, 'new_ood_label_val.npy'), new_ood_df_val['label'].values)
np.save(os.path.join(new_data_path, 'new_ood_text_test.npy'), new_ood_df_test['text'].values)
np.save(os.path.join(new_data_path, 'new_ood_label_test.npy'), new_ood_df_test['label'].values)


# Additionally save a combined set of IND (label 0) and OOD (label 1) for easy evaluations
# for model training and T-2 evaluation
combined_val_df = pd.concat([new_ind_df_val.assign(label=0), new_ood_df_val.assign(label=1)])
np.save(os.path.join(new_data_path, 'new_ind_zero_ood_one_text_dev.npy'), combined_val_df['text'].values)
np.save(os.path.join(new_data_path, 'new_ind_zero_ood_one_label_dev.npy'), combined_val_df['label'].values)

combined_test_df = pd.concat([new_ind_df_test.assign(label=0), new_ood_df_test.assign(label=1)])
np.save(os.path.join(new_data_path, 'new_ind_zero_ood_one_text_test.npy'), combined_test_df['text'].values)
np.save(os.path.join(new_data_path, 'new_ind_zero_ood_one_label_test.npy'), combined_test_df['label'].values)

# Save the label mapping dictionaries
np.save(os.path.join(new_data_path, 'new_ind_label_mapping_dict.npy'), new_ind_mapping)
np.save(os.path.join(new_data_path, 'new_ood_label_mapping_dict.npy'), new_ood_mapping)
print(f'The old-new categories mapping for new IND categories are: {new_ind_mapping}')
print(f'The old-new categories mapping for new IND categories are: {new_ood_mapping}')

# Confirm all files have been saved
print("New divided data files saved successfully in '/data/clinc/new_division_data/'")
