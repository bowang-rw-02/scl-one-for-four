# This code file override the ROCEvaluator by SentenceBert for OOD detection effect evaluation during model training


from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
import logging
import os
import csv
from typing import List

from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc


logger = logging.getLogger(__name__)

class ROCEvaluator(SentenceEvaluator):
    def __init__(self, ind_train_sentences: List[str], ind_ood_dev_sentences: List[str], ind_ood_dev_labels, best_auroc_filename, show_progress_bar: bool = False, batch_size: int = 128, name: str = '', write_csv: bool = True):

        self.ind_train_sent = ind_train_sentences
        self.ind_ood_dev_sent = ind_ood_dev_sentences
        self.ind_ood_dev_label = ind_ood_dev_labels
        self.best_auroc_filename = best_auroc_filename

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.csv_file = "auroc_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "AUROC"]
        self.write_csv = write_csv

    def __call__(self, model, output_path, epoch  = -1, steps = -1):
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        print('Start evaluation...')

        ind_train_embeddings = model.encode(self.ind_train_sent, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=True)
        ind_ood_dev_embeddings = model.encode(self.ind_ood_dev_sent, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=True)

        # AUROC calculation for dev
        # Since in the new division, the num of new IND categories are 112 of 150
        # while for training set (small) there are 80 records per category
        NUM_OF_IND_CATES = 112
        USED_IND_DATA_per_cate = 80

        USED_NETWORK_DIM = ind_train_embeddings.shape[1]

        ind_samples = ind_train_embeddings
        test_samples = ind_ood_dev_embeddings

        Y = self.ind_ood_dev_label


        c_avg_present = []
        c_cov = np.zeros([USED_NETWORK_DIM, USED_NETWORK_DIM])

        for i in range(NUM_OF_IND_CATES):
            cate_vec = ind_samples[0 + i * USED_IND_DATA_per_cate: USED_IND_DATA_per_cate + i * USED_IND_DATA_per_cate]
            cate_cov = np.cov(cate_vec.T)

            c_avg_present.append(cate_vec.mean(axis=0))
            c_cov = c_cov + cate_cov

        c_avg_present = np.array(c_avg_present)

        c_cov_avged = c_cov / NUM_OF_IND_CATES

        iV = np.linalg.pinv(c_cov_avged)

        EVAL_DATA_SIZE = ind_ood_dev_embeddings.shape[0]
        eval_distance = np.zeros(shape=EVAL_DATA_SIZE)

        for i in range(EVAL_DATA_SIZE):
            min_distance = 1000
            for j in range(NUM_OF_IND_CATES):
                distance_now = distance.mahalanobis(test_samples[i], c_avg_present[j], iV)
                if distance_now < min_distance:
                    min_distance = distance_now
            eval_distance[i] = min_distance

        pred_y_P = eval_distance

        fpr, tpr, thresholds = roc_curve(Y, pred_y_P)
        fpr90 = 1
        fpr90_thres = 10000

        for ffpr, ttpr, thres in zip(fpr, tpr, thresholds):
            if abs(ttpr - 0.90) < 0.01:
                fpr90 = ffpr
                fpr90_thres = thres
                break

        auroc = auc(fpr, tpr)
        auroc *= 100
        print(f'AUROC this epoch is: {auroc}')
        print(f'fpr90: {fpr90}, distance threshold when tpr=0.9: {fpr90_thres}')

        best_auroc_thres = np.load(self.best_auroc_filename)
        past_auroc, past_fpr90_thres = best_auroc_thres[0], best_auroc_thres[1]
        if auroc > past_auroc:
            print('A better model is found, fpr90-threshold updated.')
            np.save(self.best_auroc_filename, np.array([auroc,fpr90_thres]))


        logger.info("AUROC evaluation (higher = better) on "+self.name+" dataset"+out_txt)
        logger.info("AUROC (*100):\t{:4f}".format(auroc))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, auroc])

        return auroc