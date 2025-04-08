# This is the code file to train a base embedding (encoding) model using pure supervised contrastive learning
# After training, the model parameter will be kept unchanged through T-1 to T-4.
# The training framework is based on sbert easy training, while the SCL loss and evaluator are realized by ourselves


import os
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset
from sentence_transformers import models
from torch.utils.data import DataLoader

import math
import numpy as np

# Our made supcon loss is realized in supcon_nlp_loss_normalized
from libraries import supcon_nlp_loss_normalized, roc_evaluator_cmaha


# If you find your VRAM insufficient, please change to a lower value of batch_size
train_batch_size = 256
num_epochs = 20

# Load training data
train_sentences = np.load('./data/clinc/new_division_data/new_ind_text_train_small.npy', allow_pickle=True)
train_labels = np.load('./data/clinc/new_division_data/new_ind_label_train_small.npy')
dev_sentences = np.load('./data/clinc/new_division_data/new_ind_zero_ood_one_text_dev.npy', allow_pickle=True)
dev_labels = np.load('./data/clinc/new_division_data/new_ind_zero_ood_one_label_dev.npy')


# This time we use mpnet as the base network
model_name = 'sentence-transformers/all-mpnet-base-v2'
word_embedding_model = models.Transformer(model_name, max_seq_length=64)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Training data formatting
INPUT_DATA_NUM = len(train_labels)
train_data = []
for i in range(INPUT_DATA_NUM):
    train_data.append(InputExample(texts=[train_sentences[i],train_sentences[i]], label=train_labels[i]))
train_dataset = SentencesDataset(train_data, model)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

# A numpy arrary to record the threshold when TPR=90%. This will be used for IND/OOD filtering from Test-I
# to train an incremental model in T-4.
best_auroc_fpr90thres = np.array([1,0])
best_auroc_filename = './data/clinc/best_auroc_fpr90thres.npy'
np.save(best_auroc_filename, best_auroc_fpr90thres)

train_loss = supcon_nlp_loss_normalized.SupervisedContrastiveNLPLoss(model)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

# The evaluation also need the training data points as reference points.
# evaluation is based on centroid-Mahalanobis distance OOD detection.
dev_evaluator = roc_evaluator_cmaha.ROCEvaluator(ind_train_sentences=train_sentences, ind_ood_dev_sentences=dev_sentences, ind_ood_dev_labels=dev_labels, best_auroc_filename=best_auroc_filename)
OUTPUT_PATH = './trained_models/clinc/supcon_mpnet_clinc_112_ind_small_training_withdev'


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluator=dev_evaluator,
    output_path=OUTPUT_PATH,
    show_progress_bar=True,
    save_best_model=True
)
