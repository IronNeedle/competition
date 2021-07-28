import cv2
import mediapipe as mp
from tensorflow.keras import models
import pandas as pd
from joblib import Parallel, delayed
import numpy as np


def crop_for_recycle(image, results):
    my_image = image.copy()
    image_height, image_width, _ = my_image.shape
    pixels_to_expend_x = int(image_width / 50) + 15
    pixels_to_expend_y = int(image_height / 25) + 35
    return_list = np.array([], ndmin=4)
    return_list = return_list.reshape((0, 150, 150, 3))
    num_of_crops = 0
    flag_lh = 0
    flag_rh = 0
    flag_lw = 0
    flag_rw = 0
    if results.left_hand_landmarks:
        flag_lh = 1
        lh_min_x = 1
        lh_min_y = 1
        lh_max_x = 0
        lh_max_y = 0
        for lm in results.left_hand_landmarks.landmark:
            lh_min_x = min(lh_min_x, lm.x)
            lh_min_y = min(lh_min_y, lm.y)
            lh_max_x = max(lh_max_x, lm.x)
            lh_max_y = max(lh_max_y, lm.y)
        lh_min_x = int(lh_min_x * image_width)
        lh_min_y = int(lh_min_y * image_height)
        lh_max_x = int(lh_max_x * image_width)
        lh_max_y = int(lh_max_y * image_height)

        gap_to_cut_x_l = int((lh_max_x - lh_min_x) / 10 + pixels_to_expend_x)
        gap_to_cut_y_l = int((lh_max_y - lh_min_y) / 10 + pixels_to_expend_y)

        lh_image = my_image[max(0, lh_min_y - gap_to_cut_y_l): min(image_height, lh_max_y + gap_to_cut_y_l),
                            max(0, lh_min_x - gap_to_cut_x_l): min(image_width, lh_max_x + gap_to_cut_x_l)]

        lh_image = cv2.resize(lh_image, (150, 150))
        return_list = np.append(return_list, lh_image)
        num_of_crops += 1

    else:
        if results.pose_landmarks.landmark[19].visibility > 0.75:  # found left_wrist
            flag_lw = 1
            lh_min_x = 1
            lh_min_y = 1
            lh_max_x = 0
            lh_max_y = 0
            lm_list = [results.pose_landmarks.landmark[15],
                       results.pose_landmarks.landmark[17],
                       results.pose_landmarks.landmark[19],
                       results.pose_landmarks.landmark[21]]
            for lm in lm_list:  # left palm points
                lh_min_x = min(lh_min_x, lm.x)
                lh_min_y = min(lh_min_y, lm.y)
                lh_max_x = max(lh_max_x, lm.x)
                lh_max_y = max(lh_max_y, lm.y)
            lh_min_x = int(lh_min_x * image_width)
            lh_min_y = int(lh_min_y * image_height)
            lh_max_x = int(lh_max_x * image_width)
            lh_max_y = int(lh_max_y * image_height)

            gap_to_cut_x_l = int((lh_max_x - lh_min_x) * 2 + pixels_to_expend_x)
            gap_to_cut_y_l = int((lh_max_y - lh_min_y) * 2 + pixels_to_expend_y)

            lh_image = my_image[max(0, lh_min_y - gap_to_cut_y_l): min(image_height, lh_max_y + gap_to_cut_y_l),
                                max(0, lh_min_x - gap_to_cut_x_l): min(image_width, lh_max_x + gap_to_cut_x_l)]

            lh_image = cv2.resize(lh_image, (150, 150))
            return_list = np.append(return_list, lh_image)
            num_of_crops += 1

    if results.right_hand_landmarks:
        flag_rh = 1
        rh_min_x = 1
        rh_min_y = 1
        rh_max_x = 0
        rh_max_y = 0
        for lm in results.right_hand_landmarks.landmark:
            rh_min_x = min(rh_min_x, lm.x)
            rh_min_y = min(rh_min_y, lm.y)
            rh_max_x = max(rh_max_x, lm.x)
            rh_max_y = max(rh_max_y, lm.y)
        rh_min_x = int(rh_min_x * image_width)
        rh_min_y = int(rh_min_y * image_height)
        rh_max_x = int(rh_max_x * image_width)
        rh_max_y = int(rh_max_y * image_height)

        gap_to_cut_x_r = int((rh_max_x - rh_min_x) / 10 + pixels_to_expend_x)
        gap_to_cut_y_r = int((rh_max_y - rh_min_y) / 10 + pixels_to_expend_y)
        rh_image = my_image[max(0, rh_min_y - gap_to_cut_y_r): min(image_height, rh_max_y + gap_to_cut_y_r),
                            max(0, rh_min_x - gap_to_cut_x_r): min(image_width, rh_max_x + gap_to_cut_x_r)]

        rh_image = cv2.resize(rh_image, (150, 150))
        return_list = np.append(rh_image, return_list)
        num_of_crops += 1

    else:
        if results.pose_landmarks.landmark[20].visibility > 0.75:  # found right_wrist
            flag_rw = 1
            rh_min_x = 1
            rh_min_y = 1
            rh_max_x = 0
            rh_max_y = 0
            lm_list = [results.pose_landmarks.landmark[16],
                       results.pose_landmarks.landmark[18],
                       results.pose_landmarks.landmark[20],
                       results.pose_landmarks.landmark[22]]
            for lm in lm_list:  # right palm points
                rh_min_x = min(rh_min_x, lm.x)
                rh_min_y = min(rh_min_y, lm.y)
                rh_max_x = max(rh_max_x, lm.x)
                rh_max_y = max(rh_max_y, lm.y)
            rh_min_x = int(rh_min_x * image_width)
            rh_min_y = int(rh_min_y * image_height)
            rh_max_x = int(rh_max_x * image_width)
            rh_max_y = int(rh_max_y * image_height)
            gap_to_cut_x_r = int((rh_max_x - rh_min_x) * 2 + pixels_to_expend_x)
            gap_to_cut_y_r = int((rh_max_y - rh_min_y) * 2 + pixels_to_expend_y)
            rh_image = my_image[max(0, rh_min_y - gap_to_cut_y_r): min(image_height, rh_max_y + gap_to_cut_y_r),
                                max(0, rh_min_x - gap_to_cut_x_r): min(image_width, rh_max_x + gap_to_cut_x_r)]

            rh_image = cv2.resize(rh_image, (150, 150))
            return_list = np.append(return_list, rh_image)
            num_of_crops += 1

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
    if flag_lh == 1:
        my_image = cv2.rectangle(my_image, (max(0, lh_min_x - gap_to_cut_x_l), max(0, lh_min_y - gap_to_cut_y_l)),
                                 (min(image_width, lh_max_x + gap_to_cut_x_l),
                                  min(image_height, lh_max_y + gap_to_cut_y_l)), (0, 0, 0), -1)
    if flag_rh == 1:
        my_image = cv2.rectangle(my_image, (max(0, rh_min_x - gap_to_cut_x_r), max(0, rh_min_y - gap_to_cut_y_r)),
                                 (min(image_width, rh_max_x + gap_to_cut_x_r),
                                  min(image_height, rh_max_y + gap_to_cut_y_r)), (0, 0, 0), -1)
    if flag_lw == 1:
        my_image = cv2.rectangle(my_image, (max(0, lh_min_x - gap_to_cut_x_l), max(0, lh_min_y - gap_to_cut_y_l)),
                                 (min(image_width, lh_max_x + gap_to_cut_x_l),
                                  min(image_height, lh_max_y + gap_to_cut_y_l)), (0, 0, 0), -1)
    if flag_rw == 1:
        my_image = cv2.rectangle(my_image, (max(0, rh_min_x - gap_to_cut_x_r), max(0, rh_min_y - gap_to_cut_y_r)),
                                 (min(image_width, rh_max_x + gap_to_cut_x_r),
                                  min(image_height, rh_max_y + gap_to_cut_y_r)), (0, 0, 0), -1)

    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2) as holistic:
        results = holistic.process(cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB))
        if not results.face_landmarks:
            return return_list.reshape((num_of_crops, 150, 150, 3))
        if results.left_hand_landmarks:
            lh_min_x = 1
            lh_min_y = 1
            lh_max_x = 0
            lh_max_y = 0
            for lm in results.left_hand_landmarks.landmark:
                lh_min_x = min(lh_min_x, lm.x)
                lh_min_y = min(lh_min_y, lm.y)
                lh_max_x = max(lh_max_x, lm.x)
                lh_max_y = max(lh_max_y, lm.y)
            lh_min_x = int(lh_min_x * image_width)
            lh_min_y = int(lh_min_y * image_height)
            lh_max_x = int(lh_max_x * image_width)
            lh_max_y = int(lh_max_y * image_height)

            gap_to_cut_x_l = int((lh_max_x - lh_min_x) / 10 + pixels_to_expend_x)
            gap_to_cut_y_l = int((lh_max_y - lh_min_y) / 10 + pixels_to_expend_y)

            lh_image = my_image[max(0, lh_min_y - gap_to_cut_y_l): min(image_height, lh_max_y + gap_to_cut_y_l),
                                max(0, lh_min_x - gap_to_cut_x_l): min(image_width, lh_max_x + gap_to_cut_x_l)]

            lh_image = cv2.resize(lh_image, (150, 150))
            return_list = np.append(return_list, lh_image)
            num_of_crops += 1

        else:
            if results.pose_landmarks.landmark[19].visibility > 0.75:  # found left_wrist
                lh_min_x = 1
                lh_min_y = 1
                lh_max_x = 0
                lh_max_y = 0
                lm_list = [results.pose_landmarks.landmark[15],
                           results.pose_landmarks.landmark[17],
                           results.pose_landmarks.landmark[19],
                           results.pose_landmarks.landmark[21]]
                for lm in lm_list:  # left palm points
                    lh_min_x = min(lh_min_x, lm.x)
                    lh_min_y = min(lh_min_y, lm.y)
                    lh_max_x = max(lh_max_x, lm.x)
                    lh_max_y = max(lh_max_y, lm.y)
                lh_min_x = int(lh_min_x * image_width)
                lh_min_y = int(lh_min_y * image_height)
                lh_max_x = int(lh_max_x * image_width)
                lh_max_y = int(lh_max_y * image_height)

                gap_to_cut_x_l = int((lh_max_x - lh_min_x) * 2 + pixels_to_expend_x)
                gap_to_cut_y_l = int((lh_max_y - lh_min_y) * 2 + pixels_to_expend_y)

                lh_image = my_image[max(0, lh_min_y - gap_to_cut_y_l): min(image_height, lh_max_y + gap_to_cut_y_l),
                                    max(0, lh_min_x - gap_to_cut_x_l): min(image_width, lh_max_x + gap_to_cut_x_l)]

                lh_image = cv2.resize(lh_image, (150, 150))
                return_list = np.append(return_list, lh_image)
                num_of_crops += 1

        if results.right_hand_landmarks:
            rh_min_x = 1
            rh_min_y = 1
            rh_max_x = 0
            rh_max_y = 0
            for lm in results.right_hand_landmarks.landmark:
                rh_min_x = min(rh_min_x, lm.x)
                rh_min_y = min(rh_min_y, lm.y)
                rh_max_x = max(rh_max_x, lm.x)
                rh_max_y = max(rh_max_y, lm.y)
            rh_min_x = int(rh_min_x * image_width)
            rh_min_y = int(rh_min_y * image_height)
            rh_max_x = int(rh_max_x * image_width)
            rh_max_y = int(rh_max_y * image_height)

            gap_to_cut_x_r = int((rh_max_x - rh_min_x) / 10 + pixels_to_expend_x)
            gap_to_cut_y_r = int((rh_max_y - rh_min_y) / 10 + pixels_to_expend_y)
            rh_image = my_image[max(0, rh_min_y - gap_to_cut_y_r): min(image_height, rh_max_y + gap_to_cut_y_r),
                                max(0, rh_min_x - gap_to_cut_x_r): min(image_width, rh_max_x + gap_to_cut_x_r)]

            rh_image = cv2.resize(rh_image, (150, 150))
            return_list = np.append(return_list, rh_image)
            num_of_crops += 1

        else:
            if results.pose_landmarks.landmark[20].visibility > 0.75:  # found right_wrist
                rh_min_x = 1
                rh_min_y = 1
                rh_max_x = 0
                rh_max_y = 0
                lm_list = [results.pose_landmarks.landmark[16],
                           results.pose_landmarks.landmark[18],
                           results.pose_landmarks.landmark[20],
                           results.pose_landmarks.landmark[22]]
                for lm in lm_list:  # right palm points
                    rh_min_x = min(rh_min_x, lm.x)
                    rh_min_y = min(rh_min_y, lm.y)
                    rh_max_x = max(rh_max_x, lm.x)
                    rh_max_y = max(rh_max_y, lm.y)
                rh_min_x = int(rh_min_x * image_width)
                rh_min_y = int(rh_min_y * image_height)
                rh_max_x = int(rh_max_x * image_width)
                rh_max_y = int(rh_max_y * image_height)
                gap_to_cut_x_r = int((rh_max_x - rh_min_x) * 2 + pixels_to_expend_x)
                gap_to_cut_y_r = int((rh_max_y - rh_min_y) * 2 + pixels_to_expend_y)
                rh_image = my_image[max(0, rh_min_y - gap_to_cut_y_r): min(image_height, rh_max_y + gap_to_cut_y_r),
                                    max(0, rh_min_x - gap_to_cut_x_r): min(image_width, rh_max_x + gap_to_cut_x_r)]

                rh_image = cv2.resize(rh_image, (150, 150))
                return_list = np.append(return_list, rh_image)
                num_of_crops += 1
        return_list = return_list.reshape((num_of_crops, 150, 150, 3))
        return return_list


def answer_from_df(df):
    my_max = df[['stop', 'victory', 'mute', 'ok', 'like', 'dislike']]
    my_max = my_max.max(axis=1)
    t = my_max.to_list()
    index = t.index(max(t))
    return list(df.iloc[index])


def collect_crops(answers):
    my_list = []
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2) as holistic:
        for idx, file in enumerate(answers['frame_path']):
            print(idx)
            image = cv2.imread(file)
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.face_landmarks:
                my_list.append((0, file))
                continue
            crops = crop_for_recycle(image, results)
            if len(crops) > 0:
                crops = crops / 255.
                my_list.append((crops, file))
            else:
                my_list.append((0, file))
    return my_list


def total(base_csv):

    NUM_WORKERS = 8
    len_df = len(base_csv) // NUM_WORKERS
    list_for_parallel = []
    for i in range(NUM_WORKERS - 1):
        list_for_parallel.append(base_csv[i * len_df: (i + 1) * len_df])
    list_for_parallel.append(base_csv[(NUM_WORKERS - 1) * len_df:])
    results = Parallel(n_jobs=NUM_WORKERS, verbose=0, backend="loky")(
        map(delayed(collect_crops), list_for_parallel))
    answer = (results[0] + results[1] + results[2] + results[3] +
              results[4] + results[5] + results[6] + results[7])
    df = pd.DataFrame(columns=['dislike', 'like', 'mute', 'no_gesture', 'ok', 'stop', 'victory', 'frame_path'])

    json_file = open("MobileNetV2_1.json", 'r')
    model = json_file.read()
    json_file.close()
    model = models.model_from_json(model)
    model.load_weights("MobileNetV2_1.h5")

    for elem in answer:
        if type(elem[0]) == int:
            df.loc[len(df)] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, elem[1]]
            continue
        else:
            predictions = model.predict(elem[0])

        if len(predictions) > 1:
            temp_df = pd.DataFrame(predictions, columns=['dislike', 'like', 'mute', 'no_gesture',
                                                         'ok', 'stop', 'victory'])
            tmp_list = answer_from_df(temp_df)
            tmp_list.append(elem[1])
            df.loc[len(df)] = tmp_list
        else:
            tmp_list = list(predictions[0])
            tmp_list.append(elem[1])
            df.loc[len(df)] = tmp_list
    return df

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    INPUT_PATH = 'train.csv'
    base_csv = pd.read_csv(INPUT_PATH)
    base_csv = base_csv.sample(frac=1)
    count = 2
    base_csv = base_csv[:count * 1000]
    base_csv['frame_path'] = base_csv['frame_path'].apply(lambda x: 'F:/Train_DS/' + str(x))
    df_res = pd.DataFrame(columns=['dislike', 'like', 'mute', 'no_gesture', 'ok', 'stop', 'victory', 'frame_path'])
    for i in range(count):
        base_csv1 = base_csv[i * 1000: (i + 1) * 1000]
        df = total(base_csv1)
        df_res = df_res.append(df)
        df_res.to_csv('answers.csv', index=False)
        print(f'{1000*(i+1)} done')

