import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

CLASS_NAME2LABEL_DICT = {
        'no_gesture': 0,
        'stop': 1,
        'victory': 2,
        'mute': 3,
        'ok': 4,
        'like': 5,
        'dislike': 6
    }
TARGET_FPR = 0.002


def main(my_train, my_scores):
    test_df = my_train
    scores_df = my_scores

    merged_df = pd.merge(left=scores_df, right=test_df, how='outer', on='frame_path')

    if merged_df.isnull().values.any():
        raise Exception('Something is wrong in the submission file')

    y_true = merged_df.label.values

    class_name2score_dict = {}
    for class_name, label in CLASS_NAME2LABEL_DICT.items():
        # Skipping not-target "no gesture" class
        if class_name == 'no_gesture':
            continue

        y_pred = merged_df[class_name].values

        fpr, tpr, thr = roc_curve(y_true, y_pred, pos_label=label)

        if fpr[0] < TARGET_FPR:
            target_tpr = tpr[fpr < TARGET_FPR][-1]
        else:
            target_tpr = 0.0

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label=f'ROC curve, {target_tpr}')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, .003])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


        class_name2score_dict[class_name] = target_tpr
        print(f'{class_name} score is {target_tpr}')

    mean_target_tpr = np.mean(list(class_name2score_dict.values()))
    print(f'Score: {mean_target_tpr}')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_df_path', type=str, help='Path to GT test df with frame_path and labels')
    parser.add_argument('--scores_df_path', type=str, default='', help='Path to submit file with scores')
    return parser.parse_args(argv)

if __name__ == '__main__':
    my_scores = pd.read_csv('answers.csv')
    # my_scores1 = pd.read_csv('answers_to_eval_71-90000.csv')
    # my_scores = pd.concat([my_scores, my_scores1])
    # my_scores1 = pd.read_csv('answers_to_eval_90-161.csv')
    # my_scores = pd.concat([my_scores, my_scores1])
    my_train = pd.read_csv('train.csv')

    # my_train = my_train[:161000]
    my_train['frame_path'] = my_train['frame_path'].apply(lambda x: 'F:/Train_DS/' + str(x))
    my_train = my_train.loc[my_train['frame_path'].isin(my_scores['frame_path'].values)]
    my_train['label'] = my_train['class_name'].apply(lambda x: CLASS_NAME2LABEL_DICT[x])

    main(my_train, my_scores)
