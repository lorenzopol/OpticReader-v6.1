import os

import cv2
import numpy as np
import imutils

from joblib import load

from custom_utils import Utils as u
from dataclasses import dataclass, field

from typing import *


@dataclass(order=True)
class User:
    index: str = field(compare=False)
    score: float
    per_sub_score: list = field(repr=False)
    score_list: list = field(compare=False, repr=False)
    sorted_user_answer_dict: dict = field(compare=False, repr=False)


def raise_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(12, 12))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


def get_left_side_y_values(bgr_image: np.array, sample_pos: int) -> [list, list]:
    """estra tutti i valori neri data una specifica x. Superset di u.find_n_black_point_on_row"""
    gr_image: np.array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    LY_begin_question_box, LY_end_question_box = u.find_n_black_point_on_row(gr_image, sample_pos, False)[:2]
    return LY_begin_question_box, LY_end_question_box


def compute_rotation_angle_for_image_alignment(bgr_image: np.array, left_side_sample_pos: int) -> float:
    """data l'immagine e la x per trovare le barre nere, ritorna l'angolo in DEG per centrare l'immgaine. Da sostituire
    con WarpAffine dopo l'aggiunta dei 4 quadrati ai bordi del foglio"""
    gr_test: np.array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    LY_begin_question_box, LY_end_question_box = get_left_side_y_values(bgr_image, left_side_sample_pos)

    y_delta: int = 15
    LY_begin_sample_point: int = LY_begin_question_box - y_delta
    LY_end_sample_point: int = LY_end_question_box + y_delta

    top_left_x_pos_for_arctan: int = u.find_n_black_point_on_row(gr_test, LY_begin_sample_point, True)[0]
    bottom_left_x_pos_for_arctan: int = u.find_n_black_point_on_row(gr_test, LY_end_sample_point, True)[0]
    angle: float = np.arctan((top_left_x_pos_for_arctan - bottom_left_x_pos_for_arctan) /
                             (LY_end_sample_point - top_left_x_pos_for_arctan))
    return np.rad2deg(angle)


def align_black_border(bgr_image: np.array, sample_pos: int) -> np.array:
    """rotate and crop image for column alignment"""
    angle: float = abs(compute_rotation_angle_for_image_alignment(bgr_image, sample_pos))
    BGR_SC_ROT_test: np.array = imutils.rotate(bgr_image, angle)
    tan_res = np.tan(np.deg2rad(angle))
    # crop for deleting black zones from rotation
    # Cropping sequence:
    #   top y crop
    #   bottom y crop
    #   left x crop
    #   right x crop
    BGR_SC_ROT_CROP_test: np.array = BGR_SC_ROT_test[round(BGR_SC_ROT_test.shape[1] * tan_res):
                                                     BGR_SC_ROT_test.shape[0] - round(
                                                         BGR_SC_ROT_test.shape[1] * tan_res),

                                                     round(BGR_SC_ROT_test.shape[0] * tan_res):
                                                     BGR_SC_ROT_test.shape[1] - round(
                                                         BGR_SC_ROT_test.shape[0] * tan_res)]
    return BGR_SC_ROT_CROP_test


def get_cols_x_pos(bgr_image: np.array, y_sample_pos: int):
    """ritorna tutti le posizioni delle colonne per le domande. Superset di u.find_n_black_point_on_row"""
    gr_image: np.array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return u.find_n_black_point_on_row(gr_image, y_sample_pos, True)[:5]


def detect_answers(bgr_image: np.array, bgr_img_for_debug: np.array,
                   x_cut_positions: List[int], y_cut_positions: Tuple[int],
                   is_60_question_sim, debug: str):
    question_multiplier: int = 15 if is_60_question_sim else 20

    letter: Tuple[str, ...] = ("L", "", "A", "B", "C", "D", "E")
    user_answer_dict: Dict[int, str] = {i: "" for i in range(1, 61 - 20 * int(not is_60_question_sim))}

    gr_image: np.array = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # load SVM model
    load_path = os.getcwd()
    clf = load(os.path.join(load_path, "reduced.joblib"))

    for y_index in range(len(y_cut_positions) - 1):
        for x_index in range(len(x_cut_positions) - 1):

            # se sei su una colonna di numeri saltala
            if not (x_index - 1) % 7:
                continue

            x_top_left = int(not x_index % 7) * 7 + x_cut_positions[x_index]
            x_bottom_right = int(not x_index % 7) * 2 + x_cut_positions[x_index + 1]

            y_top_left: int = y_cut_positions[y_index]
            y_bottom_right: int = y_cut_positions[y_index + 1]
            if debug == "all":
                cv2.rectangle(bgr_img_for_debug,
                              (x_top_left, y_top_left),
                              (x_bottom_right, y_bottom_right),
                              u.CYAN if x_index % 2 else u.BLUE, 1)
            crop_for_prediction: np.array = gr_image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
            crop_for_prediction: np.array = cv2.resize(crop_for_prediction, (18, 18))

            # category = ("QB", "QS", "QA", "CB", "CA")
            #               0     1     2     3     4

            crop_for_prediction: np.array = np.append(crop_for_prediction,
                                                      [x_index % 7, int(np.mean(crop_for_prediction))])
            predicted_category_index: int = clf.predict([crop_for_prediction])[0]

            # è un quadrato bianco o un cerchio bianco
            if predicted_category_index in (0, 3):
                continue

            if x_index % 7:
                # è una crocetta nei quadrati
                if predicted_category_index in (1, 4) and debug != "No":
                    cv2.rectangle(bgr_img_for_debug, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                  u.GREEN, 1)

                # è una quadrato annerito
                elif predicted_category_index == 2:
                    if debug != "No":
                        cv2.rectangle(bgr_img_for_debug, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                      u.RED, 1)
                    continue
            else:
                # è un cerchio annerito
                if predicted_category_index in (1, 2, 4) and debug != "No":
                    cv2.rectangle(bgr_img_for_debug, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                  u.RED, 1)

            question_letter = letter[x_index % 7]
            question_number = (question_multiplier * (x_index // 7) + (y_index + 1))

            cv2.putText(bgr_img_for_debug, f"{question_number}",
                        (x_cut_positions[x_index], y_cut_positions[y_index]),
                        cv2.FONT_HERSHEY_SIMPLEX, .3, u.RED, 1)

            user_answer_dict[question_number] = question_letter if user_answer_dict[question_number] != "L" else "L"

    # ------------- DEBUG WINDOW ------------------------
    if debug != "No":
        cv2.imshow("result", bgr_img_for_debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return user_answer_dict


def get_x_cuts(cols_x_pos: List[int], is_60_question_sim) -> List[int]:
    correction_sequence: Tuple[int, ...] = (2, 1, 0, 0, 0, 0, 0)
    x_cut_positions: List[int] = [cols_x_pos[0 + int(not is_60_question_sim)]]
    for col in range(0 + int(not is_60_question_sim), 4 - int(not is_60_question_sim)):
        rect_width: int = (cols_x_pos[col + 1] - cols_x_pos[col]) // 7
        for cut_number in range(7):
            x_pos: int = cols_x_pos[col] + rect_width * (cut_number + 1)
            x_cut_positions.append(x_pos + correction_sequence[cut_number])
    return x_cut_positions


def generate_score_list(user_answer_dict: Dict[int, str],
                        how_many_people_got_a_question_right_dict: Dict[int, int]):
    correct_answers: list = u.retrieve_or_display_answers()

    score_list: List[float] = []

    for i in range(len(user_answer_dict)):
        number, letter = correct_answers[i].split(";")[0].split(" ")
        user_letter = user_answer_dict[i + 1]
        if letter == "*" or user_letter == "L":
            score_list.append(0)
            user_answer_dict[i + 1] = ""
            continue
        if not user_letter:
            score_list.append(0)
        else:
            if user_letter == letter:
                score_list.append(1.5)
                how_many_people_got_a_question_right_dict[i] += 1
            else:
                score_list.append(-0.4)
    return score_list


def calculate_single_sub_score(score_list):
    """Divide scores for each subject. L'ORDINE IN RETURN CONTA, seguire quello indicato nel bando ministeriale"""
    noq_for_sub = {
        "logicaCultura": [0, 9],
        "biologia": [9, 32],
        "chimica": [32, 47],
        "fisicaMatematica": [47, 60]
    }
    risultati_biologia = score_list[noq_for_sub.get("biologia")[0]:noq_for_sub.get("biologia")[1]]
    risultati_chimica = score_list[noq_for_sub.get("chimica")[0]:noq_for_sub.get("chimica")[1]]
    risultati_fisicaMate = score_list[noq_for_sub.get("fisicaMatematica")[0]:noq_for_sub.get("fisicaMatematica")[1]]
    risultati_logicaCultura = score_list[noq_for_sub.get("logicaCultura")[0]:noq_for_sub.get("logicaCultura")[1]]

    return [sum(risultati_biologia), sum(risultati_chimica),
            sum(risultati_fisicaMate), sum(risultati_logicaCultura)]


def evaluator(abs_img_path: str, valid_ids,
              how_many_people_got_a_question_right_dict: Dict[int, int], all_users: List[User],
              is_60_question_sim: bool, debug: str):
    """given ONE img_abs_path, compute scores, and update global scoring dict"""
    BGR_test: np.array = cv2.imread(abs_img_path)

    cropped_bar_code_id = u.decode_ean_barcode(BGR_test[((BGR_test.shape[0]) * 3) // 4:])
    if cropped_bar_code_id not in valid_ids:
        cropped_bar_code_id = input(f"BARCODE fallito per {abs_img_path} >>")

    BGR_SC_test: np.array = imutils.resize(BGR_test, height=900)

    bgr_img_for_debug = None
    question_multiplier: int = 15 if is_60_question_sim else 20

    x_sample_pos: int = 32
    BGR_SC_ROT_CROP_test: np.array = align_black_border(BGR_SC_test, x_sample_pos)

    LY_begin_question_box, LY_end_question_box = get_left_side_y_values(BGR_SC_ROT_CROP_test, x_sample_pos)
    if debug != "No":
        bgr_img_for_debug: np.array = BGR_SC_ROT_CROP_test.copy()
    if debug == "all":
        cv2.line(bgr_img_for_debug,
                 (x_sample_pos, 0), (x_sample_pos, BGR_SC_ROT_CROP_test.shape[0]),
                 u.CYAN, 1)
        cv2.circle(bgr_img_for_debug,
                   (x_sample_pos, LY_begin_question_box), 3,
                   u.RED, -1)
        cv2.circle(bgr_img_for_debug,
                   (x_sample_pos, LY_end_question_box), 3,
                   u.RED, -1)
    cols_x_pos = get_cols_x_pos(BGR_SC_ROT_CROP_test, y_sample_pos=LY_end_question_box + 20)
    x_cut_positions = get_x_cuts(cols_x_pos, is_60_question_sim)

    question_square_height: int = (LY_end_question_box - LY_begin_question_box) // question_multiplier

    y_cut_positions = tuple(
        round(y + 0.05 * y) for y in range(LY_begin_question_box - question_square_height + 6,
                                           LY_end_question_box - question_square_height, question_square_height))
    BGR_SC_ROT_CROP_test = raise_contrast(BGR_SC_ROT_CROP_test)
    user_answer_dict: Dict[int, str] = detect_answers(BGR_SC_ROT_CROP_test, bgr_img_for_debug,
                                                      x_cut_positions, y_cut_positions,
                                                      is_60_question_sim, debug)

    score_list: List[float] = generate_score_list(user_answer_dict, how_many_people_got_a_question_right_dict)
    per_sub_score = calculate_single_sub_score(score_list) if is_60_question_sim else []
    user = User(cropped_bar_code_id, round(sum(score_list), 2), per_sub_score, score_list, user_answer_dict)
    all_users.append(user)

    return all_users, how_many_people_got_a_question_right_dict
