# This code FREEZE the base model trained on pre-step 01, and encodes both training and test data to sentence embeddings
# in advance for an easier computation. Note that the test set data are just encoded and never further learned by the
# model, so no risk in data leakage.

import os
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from sentence_transformers import SentenceTransformer
import numpy as np


# Data to be encoded
# The training data are encoded for the cluster centroid calculation and for the classification/detection reference
Texts_to_encode_no1 = np.load('./data/clinc/new_division_data/new_ind_text_train_small.npy', allow_pickle=True)
# The test data are encoded for the vector distance calculation and testing
Texts_to_encode_no2 = np.load('./data/clinc/new_division_data/new_ind_text_test.npy', allow_pickle=True)
Texts_to_encode_no3 = np.load('./data/clinc/new_division_data/new_ood_text_test.npy', allow_pickle=True)
Texts_to_encode_no4 = np.concatenate((Texts_to_encode_no2, Texts_to_encode_no3), axis=0)


print('Loading trained base model...')
model = SentenceTransformer('./trained_models/clinc/supcon_mpnet_clinc_112_ind_small_training_withdev')

print('Encoding necessary data...')

sentence_embeddings1 = model.encode(Texts_to_encode_no1)
sentence_embeddings2 = model.encode(Texts_to_encode_no2)
sentence_embeddings3 = model.encode(Texts_to_encode_no3)
sentence_embeddings4 = model.encode(Texts_to_encode_no4)



np.save('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ind_train_small_sentvec.npy', sentence_embeddings1)
np.save('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ind_test_sentvec.npy', sentence_embeddings2)
np.save('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ood_test_sentvec.npy', sentence_embeddings3)
np.save('./data/clinc/sent_embedding/supcon_mpnet_clinc_112_ind_ood_test_sentvec.npy', sentence_embeddings4)

print('All data needed encoded.')

