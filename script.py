import cv2
import mediapipe as mp
from tensorflow.keras import models
import pandas as pd
from joblib import Parallel, delayed
import numpy as np

def crop_calc(my_image, landmarks, image_width, image_height, pixels_to_expend_x, pixels_to_expend_y):
    min_x = 1
    min_y = 1
    max_x = 0
    max_y = 0
    for lm in landmarks:
        min_x = min(min_x, lm.x)
        min_y = min(min_y, lm.y)
        max_x = max(max_x, lm.x)
        max_y = max(max_y, lm.y)
    min_x = int(min_x * image_width)
    min_y = int(min_y * image_height)
    max_x = int(max_x * image_width)
    max_y = int(max_y * image_height)

    gap_to_cut_x_l = int((max_x - min_x) / 10 + pixels_to_expend_x)
    gap_to_cut_y_l = int((max_y - min_y) / 10 + pixels_to_expend_y)

    image = my_image[max(0, min_y - gap_to_cut_y_l): min(image_height, max_y + gap_to_cut_y_l),
                     max(0, min_x - gap_to_cut_x_l): min(image_width, max_x + gap_to_cut_x_l)]
    cv2.imshow('', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cv2.resize(image, (150, 150)), [min_y - gap_to_cut_y_l, max_y + gap_to_cut_y_l,
                                           min_x - gap_to_cut_x_l, max_x + gap_to_cut_x_l]


def crop_for_recicle(image, results):
    """
    This function gets an image to work with and results of first MediaPipe usage (contains normalized face,
    pose and hand landmarks). It processes image depending on whether there are palm palm landmarks or only wrist
    landmarks and makes crops. Also it paints black rectangles on face and hands and passes the image through MediaPipe
    again to find another person if there is one.
    :return: hand crops as numpy arrays
    """
    my_image = image.copy()
    image_height, image_width, _ = my_image.shape
    pixels_to_expend_x = int(image_width / 50) + 15
    pixels_to_expend_y = int(image_height / 25) + 35
    return_list = np.array([], ndmin=4)
    return_list = return_list.reshape((0, 150, 150, 3))
    flag_lh = 0
    flag_rh = 0
    flag_lw = 0
    flag_rw = 0

    if results.left_hand_landmarks:
        flag_lh = 1
        image, list_to_crop_l = crop_calc(my_image, results.left_hand_landmarks.landmark,
                                          image_width, image_height, pixels_to_expend_x, pixels_to_expend_y)
        return_list = np.append(return_list, image)
    elif results.pose_landmarks.landmark[19].visibility > 0.75:
        flag_lw = 1
        lm_list = [results.pose_landmarks.landmark[15],
                   results.pose_landmarks.landmark[17],
                   results.pose_landmarks.landmark[19],
                   results.pose_landmarks.landmark[21]]
        image, list_to_crop_l = crop_calc(my_image, lm_list,
                                          image_width, image_height, pixels_to_expend_x, pixels_to_expend_y)
        return_list = np.append(return_list, image)

    if results.right_hand_landmarks:
        flag_rh = 1
        image, list_to_crop_r = crop_calc(my_image, results.right_hand_landmarks.landmark,
                                          image_width, image_height, pixels_to_expend_x, pixels_to_expend_y)
        return_list = np.append(return_list, image)

    elif results.pose_landmarks.landmark[20].visibility > 0.75:
        flag_rw = 0
        lm_list = [results.pose_landmarks.landmark[16],
                   results.pose_landmarks.landmark[18],
                   results.pose_landmarks.landmark[20],
                   results.pose_landmarks.landmark[22]]
        image, list_to_crop_r = crop_calc(my_image, lm_list, image_width, image_height,
                                          pixels_to_expend_x, pixels_to_expend_y)
        return_list = np.append(return_list, image)

    if results.face_landmarks:
        f_min_x = 1
        f_min_y = 1
        f_max_x = 0
        f_max_y = 0
        for lm in results.face_landmarks.landmark:
            f_min_x = min(f_min_x, lm.x)
            f_min_y = min(f_min_y, lm.y)
            f_max_x = max(f_max_x, lm.x)
            f_max_y = max(f_max_y, lm.y)
        f_min_x = int(f_min_x * image_width)
        f_min_y = int(f_min_y * image_height)
        f_max_x = int(f_max_x * image_width)
        f_max_y = int(f_max_y * image_height)

        gap_to_cut_x = int((f_max_x - f_min_x) / 5)
        gap_to_cut_y = int((f_max_y - f_min_y) / 5)
        my_image = cv2.rectangle(my_image, (max(0, f_min_x - gap_to_cut_x), max(0, f_min_y - gap_to_cut_y)),
                                 (min(image_width, f_max_x + gap_to_cut_x),
                                  min(image_height, f_max_y + gap_to_cut_y)), (0, 0, 0), -1)
    if flag_lh == 1 or flag_lw == 1:
        my_image = cv2.rectangle(my_image, (max(0, list_to_crop_l[2]), max(0, list_to_crop_l[0])),
                                 (min(image_width, list_to_crop_l[3]),
                                  min(image_height, list_to_crop_l[1])), (0, 0, 0), -1)
    if flag_rh == 1 or flag_rw == 1:
        my_image = cv2.rectangle(my_image, (max(0, list_to_crop_r[2]), max(0, list_to_crop_r[0])),
                                 (min(image_width, list_to_crop_r[3]),
                                  min(image_height, list_to_crop_r[1])), (0, 0, 0), -1)

    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2) as holistic:
        results = holistic.process(cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB))
        if not results.face_landmarks:
            return return_list
        if results.left_hand_landmarks:
            image, list_to_crop_l = crop_calc(my_image, results.left_hand_landmarks.landmark,
                                              image_width, image_height,
                                              pixels_to_expend_x, pixels_to_expend_y)
            return_list = np.append(return_list, image)
        elif results.pose_landmarks.landmark[19].visibility > 0.75:
            lm_list = [results.pose_landmarks.landmark[15],
                       results.pose_landmarks.landmark[17],
                       results.pose_landmarks.landmark[19],
                       results.pose_landmarks.landmark[21]]
            image, list_to_crop_l = crop_calc(my_image, lm_list,
                                              image_width, image_height,
                                              pixels_to_expend_x, pixels_to_expend_y)
            return_list = np.append(return_list, image)

        if results.right_hand_landmarks:
            image, list_to_crop_r = crop_calc(my_image, results.right_hand_landmarks.landmark,
                                              image_width, image_height,
                                              pixels_to_expend_x, pixels_to_expend_y)
            return_list = np.append(return_list, image)

        elif results.pose_landmarks.landmark[20].visibility > 0.75:
            lm_list = [results.pose_landmarks.landmark[16],
                       results.pose_landmarks.landmark[18],
                       results.pose_landmarks.landmark[20],
                       results.pose_landmarks.landmark[22]]
            image, list_to_crop_r = crop_calc(my_image, lm_list,
                                              image_width, image_height,
                                              pixels_to_expend_x, pixels_to_expend_y)
            return_list = np.append(return_list, image)
        return return_list


def answer_from_df(df):
    """
    If there are several crops this function chooses a crop containing gesture with highest possibility
    :param df: dataframe with model predictions for each crop
    """
    my_max = df[['stop', 'victory', 'mute', 'ok', 'like', 'dislike']]
    my_max = my_max.max(axis=1)
    temp = my_max.to_list()
    index = temp.index(max(temp))
    return list(df.iloc[index])


def collect_crops(base_df):
    """
    This function takes a path to a picture, opens it and looks for a human. It detects face, pose and hand landmarks
    via MediaPipe lib. Finally it return a list with all hand-crops found and paths to original pictures for each crop.
    :param base_df: dataframe with paths to pictures
    :return: crops as numpy arrays
    """

    my_list = []
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2) as holistic:
        for idx, file in enumerate(base_df['frame_path']):
            image = cv2.imread(file)
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.face_landmarks:
                my_list.append((0, file))
                continue
            crops = crop_for_recicle(image, results)
            if len(crops) > 0:
                crops = crops / 255.
                my_list.append((crops, file))
            else:
                my_list.append((0, file))
    return my_list


if __name__ == '__main__':
    json_file = open("MobileNetV2_1.json", 'r')
    model = json_file.read()
    json_file.close()
    model = models.model_from_json(model)
    model.load_weights("MobileNetV2_1.h5")

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    INPUT_PATH = 'train.csv'
    NUM_WORKERS = 8

    base_csv = pd.read_csv(INPUT_PATH)

    len_df = len(base_csv) // NUM_WORKERS
    list_for_parallel = []
    for i in range(NUM_WORKERS - 1):
        list_for_parallel.append(base_csv[i * len_df: (i + 1) * len_df])
    list_for_parallel.append(base_csv[(NUM_WORKERS - 1) * len_df:])
    results = Parallel(n_jobs=NUM_WORKERS, verbose=0, backend="loky")(
        map(delayed(collect_crops), list_for_parallel))
    answer = (results[0] + results[1] + results[2] + results[3] +
              results[4] + results[5])

    df = pd.DataFrame(columns=['dislike', 'like', 'mute', 'no_gesture', 'ok', 'stop', 'victory', 'frame_path'])
    for elem in answer:
        if type(elem[0]) == int:
            df.loc[len(df)] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, elem[1]]
            continue
        else:
            predictions = model.predict(elem[0])

        if len(predictions) > 1:
            temp_df = pd.DataFrame(predictions,
                                   columns=['dislike', 'like', 'mute', 'no_gesture', 'ok', 'stop', 'victory'])
            tmp_list = answer_from_df(temp_df)
            tmp_list.append(elem[1])
            df.loc[len(df)] = tmp_list
        else:
            tmp_list = list(predictions[0])
            tmp_list.append(elem[1])
            df.loc[len(df)] = tmp_list
    df.to_csv('answers.csv', index=False)
