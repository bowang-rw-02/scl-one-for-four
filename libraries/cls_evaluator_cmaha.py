# This is the code file realizing the classification evaluation of T-4

from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
import logging
from typing import List

from scipy.spatial import distance
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

class CLSEvaluator(SentenceEvaluator):
    def __init__(self, ref_sent_full_category: List[str], ref_label_full_category, test_two_sent_full_category: List[str], test_two_label_full_category, show_progress_bar: bool = False, batch_size: int = 128, name: str = '', write_csv: bool = True):

        self.ref_sent_full_category = ref_sent_full_category
        self.ref_label_full_category = ref_label_full_category
        self.test_two_sent_full_category = test_two_sent_full_category
        self.test_two_label_full_category = test_two_label_full_category
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

    def __call__(self, model, output_path, epoch  = -1, steps = -1):
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        ref_sent_full_category_embd = model.encode(self.ref_sent_full_category, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=True)
        test_two_sent_full_category_embd = model.encode(self.test_two_sent_full_category, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=True)


        NUM_OF_IND_CATES = 150
        USED_IND_DATA_per_cate = 80

        USED_NETWORK_DIM = ref_sent_full_category_embd.shape[1]

        ref_ind_samples = ref_sent_full_category_embd
        test_two_samples = test_two_sent_full_category_embd


        c_avg_present = []
        c_cov = np.zeros([USED_NETWORK_DIM, USED_NETWORK_DIM])

        for i in range(NUM_OF_IND_CATES):
            cate_vec = ref_ind_samples[0 + i * USED_IND_DATA_per_cate: USED_IND_DATA_per_cate + i * USED_IND_DATA_per_cate]
            cate_cov = np.cov(cate_vec.T)

            c_avg_present.append(cate_vec.mean(axis=0))
            c_cov = c_cov + cate_cov

        c_avg_present = np.array(c_avg_present)

        c_cov_avged = c_cov / NUM_OF_IND_CATES

        iV = np.linalg.pinv(c_cov_avged)


        TEST_DATA_SIZE = test_two_sent_full_category_embd.shape[0]
        test_distance = np.zeros(shape=TEST_DATA_SIZE)

        ind_samples_label_mapping = []
        for label in self.ref_label_full_category:
            if label not in ind_samples_label_mapping:
                ind_samples_label_mapping.append(label)

        predictions = []

        for i in range(TEST_DATA_SIZE):
            min_distance = 1000
            min_cate = -1
            for j in range(NUM_OF_IND_CATES):
                distance_now = distance.mahalanobis(test_two_samples[i], c_avg_present[j], iV)
                if distance_now < min_distance:
                    min_distance = distance_now
                    min_cate = ind_samples_label_mapping[j]
            if i % 1000 == 0:
                print('Finished ', i, 'points evaluation')
            test_distance[i] = min_distance
            predictions.append(min_cate)

        test_two_labels = self.test_two_label_full_category

        # Overall F1 accuracy on all categories
        macro_f1 = f1_score(test_two_labels, predictions, average='macro')
        micro_f1 = f1_score(test_two_labels, predictions, average='micro')
        weighted_f1 = f1_score(test_two_labels, predictions, average='weighted')

        print('F1 scores on the overall 150 categories:')
        print(f"Micro F1 Score: {micro_f1}")
        print(f"Macro F1 Score: {macro_f1}")
        print(f"Weighted F1 Score: {weighted_f1}")


        NUM_OF_IND_CATES_BEFORE = 112
        split_point = NUM_OF_IND_CATES_BEFORE * (100-USED_IND_DATA_per_cate)
        # OLD IND results (label 0-111)
        first_half_preds = predictions[:split_point]
        first_half_true_labels = test_two_labels[:split_point]

        calculate_cates = list(range(NUM_OF_IND_CATES_BEFORE))
        f1_micro_first_half = f1_score(first_half_true_labels, first_half_preds, labels=calculate_cates,
                                       average='micro')
        f1_macro_first_half = f1_score(first_half_true_labels, first_half_preds, labels=calculate_cates,
                                       average='macro')
        f1_weighted_first_half = f1_score(first_half_true_labels, first_half_preds, labels=calculate_cates,
                                          average='weighted')

        print('F1 scores on the old IND categories (label 0-111):')
        print(f"Micro F1 Score: {f1_micro_first_half}")
        print(f"Macro F1 Score: {f1_macro_first_half}")
        print(f"Weighted F1 Score: {f1_weighted_first_half}")


        # New IND (former as OOD) results (label 112-149)
        second_half_preds = predictions[split_point:]

        second_half_true_labels = test_two_labels[split_point:]

        calculate_cates_ood = list(range(NUM_OF_IND_CATES_BEFORE, NUM_OF_IND_CATES))

        f1_micro_second_half = f1_score(second_half_true_labels, second_half_preds, average='micro', labels=calculate_cates_ood)
        f1_macro_second_half = f1_score(second_half_true_labels, second_half_preds, average='macro', labels=calculate_cates_ood)
        f1_weighted_second_half = f1_score(second_half_true_labels, second_half_preds, average='weighted', labels=calculate_cates_ood)


        print('F1 scores on the new IND categories (former as OOD, label 112-149):')
        print(f"Micro F1 Score: {f1_micro_second_half}")
        print(f"Macro F1 Score: {f1_macro_second_half}")
        print(f"Weighted F1 Score: {f1_weighted_second_half}")




        return micro_f1
