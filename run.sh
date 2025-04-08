#!/usr/bin/env bash

export PYTHONPATH="$PYTHONPATH:./"

echo "------------------------------------------"
echo "Start Pre-00 step, new dataset building and Test-II splitting..."
echo "------------------------------------------"

python ./00_dataset_split_codes/pre-1_new-ind-ood-determine_data-split.py
wait
echo "New IND/OOD dataset building complete."
python ./00_dataset_split_codes/pre-2_data-split-for-train-small-and-test2.py
wait
echo "Training data-small, Test-II set building complete."


echo "------------------------------------------"
echo "Start Pre-01 step, base model training using supervised contrastive learning..."
echo "------------------------------------------"

python ./01_base_model_training_codes/t-1_to_t-4_base_model_training.py
wait
echo "Base model training finished, encoding necessary sentences used in following experiments into embeddings in advance..."
python ./02_sentence_embedding_encoding/sentence_encoding_using_01_base_model.py
wait
echo "Sentence encoding and all preparations finished! We will soon start T-1 evaluation."


echo "------------------------------------------"
echo "Start T-1 IND intent classification evaluation..."
echo "------------------------------------------"

python ./t-1_evaluation_codes/t-1_classification_evaluation.py
wait
echo "T-1 evaluation finished. We will soon start T-2 evaluation."

echo "------------------------------------------"
echo "Start T-2 OOD detection evaluation..."
echo "------------------------------------------"

python ./t-2_evaluation_codes/t-2_ood_detection_evaluation.py
wait
echo "T-2 evaluation finished. We will soon start T-3 evaluation."

echo "------------------------------------------"
echo "Start T-3 New intent discovery evaluation..."
echo "------------------------------------------"

python ./t-3_evaluation_codes/t-3_new_intent_discovery_evaluation_only_ood.py
wait
echo "T-3 evaluation finished. We will soon start T-4 evaluation."

echo "------------------------------------------"
echo "Start T-4 Continual learning and classification evaluation..."
echo "------------------------------------------"

echo "Filtering IND/OOD from Test-I set with threshold recorded from Pre-01 base model training.."
python ./t-4_evaluation_codes/pre_t-4_ind_ood_filtering_labeling.py
wait
echo "Training model by IND/OOD detected from Test-I, and conducting classification evaluation on Test-II set..."
python ./t-4_evaluation_codes/t_4_continual_learning_and_evaluation.py
wait


echo "All evaluation finished!"