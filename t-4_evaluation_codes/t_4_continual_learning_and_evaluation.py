# This is the code file for T-4 Continual learning and Full intent classification evaluation
# It trains the model obtained in Pre-II step incrementally with both detected IND and OOD from Test-I set.
# The training method is still SCL, same as Pre-II.
# After training, the new model will classify Test-II data by distance-based algorithm as T-1
# and the F1 scores on all categories (150 classes), OLD IND categories (class id: 0-111)
# and NEW OOD categories (class id: 112-149) will be calculated and shown here.


import os
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset
from sentence_transformers import models
from torch.utils.data import DataLoader
import math
import numpy as np
from libraries import supcon_nlp_loss_normalized, cls_evaluator_cmaha


train_batch_size = 256
num_epochs = 5

# Bothe the 'training' data are not used for training, but are used as the reference points to determine the final categories.
ref_sentences1 = np.load('./data/clinc/new_division_data/new_ind_text_train_small.npy', allow_pickle=True)
ref_labels1 = np.load('./data/clinc/new_division_data/new_ind_label_train_small.npy')
ref_sentences2 = np.load('./data/clinc/new_division_data/new_ood_text_train_small.npy', allow_pickle=True)
ref_labels2 = np.load('./data/clinc/new_division_data/new_ood_label_train_small.npy')

ref_sentences = np.concatenate((ref_sentences1, ref_sentences2), axis=0)
ref_labels = np.concatenate((ref_labels1, ref_labels2), axis=0)

# The real training data for T-4, which are detected IND and OOD data from Test-I
Texts11 = np.load('./data/clinc/detected_data_for_t4/test1_detected_ood_text.npy', allow_pickle=True)
Texts12 = np.load('./data/clinc/detected_data_for_t4/test1_detected_ind_text.npy', allow_pickle=True)
Texts = np.concatenate((Texts11,Texts12), axis=0)

labels11 = np.load('./data/clinc/detected_data_for_t4/test1_detected_ood_label.npy')
labels12 = np.load('./data/clinc/detected_data_for_t4/test1_detected_ind_label.npy')
labels = np.concatenate((labels11,labels12), axis=0)


# Test2 data for final test and F1 score calculations
Texts21 = np.load('./data/clinc/new_division_data/new_ind_text_test2.npy', allow_pickle=True)
Texts22 = np.load('./data/clinc/new_division_data/new_ood_text_test2.npy', allow_pickle=True)
labels21 = np.load('./data/clinc/new_division_data/new_ind_label_test2.npy')
labels22 = np.load('./data/clinc/new_division_data/new_ood_label_test2.npy')
dev_sentences = np.concatenate((Texts21, Texts22), axis=0)
dev_labels = np.concatenate((labels21,labels22), axis=0)


# Load the old model by Pre-I
model_name = './trained_models/clinc/supcon_mpnet_clinc_112_ind_small_training_withdev'
word_embedding_model = models.Transformer(model_name, max_seq_length=64)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


INPUT_DATA_NUM = len(labels)
train_data = []
for i in range(INPUT_DATA_NUM):
    train_data.append(InputExample(texts=[Texts[i],Texts[i]], label=labels[i]))
train_dataset = SentencesDataset(train_data, model)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)


train_loss = supcon_nlp_loss_normalized.SupervisedContrastiveNLPLoss(model)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm
dev_evaluator = cls_evaluator_cmaha.CLSEvaluator(ref_sent_full_category=ref_sentences, ref_label_full_category=ref_labels, test_two_sent_full_category=dev_sentences, test_two_label_full_category=dev_labels)


print('T-4 continual learning and evaluation start!')

# Call the fit method
# The F1 results on Test-II are output directly during evaluation without saving the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluator=dev_evaluator,
    # output_path=OUTPUT_PATH,
    show_progress_bar=True,
    save_best_model=False
)
