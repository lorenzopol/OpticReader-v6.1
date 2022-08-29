import numpy as np
import os
from pyzbar.pyzbar import decode


class Utils:
    """dummy class for GPF"""
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    YELLOW = (0, 220, 220)

    @staticmethod
    def write_txt(full_path: str, data: str):
        with open(full_path, "w") as cf:
            cf.write(data)

    @staticmethod
    def read_txt(full_path: str) -> list:
        with open(full_path, "r") as cf:
            content = cf.readlines()
        return content

    @staticmethod
    def one_d_slice(img: np.ndarray, pos: int, row: bool) -> np.ndarray:
        return img[pos:pos + 1, :] if row else img[:, pos:pos + 1]

    @staticmethod
    def find_n_black_point_on_row(ori_image, sample_point_pos, row):
        """refactor may be needed"""
        bool_threshold: int = 125

        one_d_sliced: np.ndarray = Utils.one_d_slice(ori_image, sample_point_pos, row)
        if not row:
            one_d_sliced = one_d_sliced.reshape((1, one_d_sliced.shape[0]))
        bool_arr: np.ndarray = (one_d_sliced < bool_threshold)[0]

        positions = np.where(bool_arr == 1)[0]

        out = [i for i in positions]
        popped = 0
        for index in range(1, len(positions)):
            if positions[index] - positions[index-1] < 10:
                del out[index-popped]
                popped += 1
        return out

    @staticmethod
    def decode_ean_barcode(cropped_img):
        """read a EAN13-barcode and return the candidate IDC (ID Candidato) from -> N-GG-MM-AAAA-IDC"""
        mid = decode(cropped_img)
        if mid:
            string_number = mid[0].data.decode("utf-8")
            return string_number[-4:-1]
        else:
            return input("Rilevamento BARCODE fallito: inserire numero della prova >>")

    # --------------------------- OLD GUI FUNCS ---------------------------
    @staticmethod
    def retrieve_or_display_answers():
        path: str = os.path.join(os.getcwd(), "risposte.txt")
        content = Utils.read_txt(path)
        return content

    @staticmethod
    def answer_modifier(number, correct):
        path: str = os.path.join(os.getcwd(), "risposte.txt")
        if os.path.isfile(path):
            content = Utils.read_txt(path)
            number = int(number)
            content[number - 1] = "".join([str(number), " ", correct.upper(), ";\n"])
            Utils.write_txt(path, "".join(content))
        else:
            print("non è stato salvato alcun file come risposte, creane uno scegliendo l'opzione 1")

    # --------------------------- EXCEL FUNCS ---------------------------
    @staticmethod
    def from_index_to_excel_column_letter(column_int):
        start_index = 0  # it can start either at 0 or at 1
        letter = ''
        while column_int > 25 + start_index:
            letter += chr(65 + int((column_int - start_index) / 26) - 1)
            column_int = column_int - (int((column_int - start_index) / 26)) * 26
        letter += chr(65 - start_index + (int(column_int)))
        return letter

    @staticmethod
    def xlsx_dumper(user, placement, correct_answers, workbook, is_60_question_sim):
        formats = [workbook.add_format({'border': 1,
                                        'align': 'center',
                                        'valign': 'vcenter'}),
                   workbook.add_format({'bg_color': 'red',
                                        'border': 1,
                                        'align': 'center',
                                        'valign': 'vcenter'
                                        }),
                   workbook.add_format({'border': 1,
                                        'align': 'center',
                                        'valign': 'vcenter',
                                        'bg_color': 'green'})]
        worksheet = workbook.worksheets()[0]
        v_delta = 4

        worksheet.merge_range('A1:C1', 'n° Domanda', workbook.add_format({'bold': 1,
                                                                          'border': 1,
                                                                          'align': 'center',
                                                                          'valign': 'vcenter'})
                              )
        # Create question number header
        _0_header = [*range(1, 61-(20*int(not is_60_question_sim)))]
        for col_num, data in enumerate(_0_header):
            worksheet.write(0, col_num + 3, data, workbook.add_format({'border': 1,
                                                                       'align': 'center',
                                                                       'valign': 'vcenter',
                                                                       "color": "white",
                                                                       "bg_color": "#4287F5"}))

        worksheet.merge_range('A2:C2', "RISPOSTA ESATTA", workbook.add_format({'bold': 1,
                                                                               'border': 1,
                                                                               'align': 'center',
                                                                               'valign': 'vcenter',
                                                                               }))
        # Create correct answer header
        _1_header = [*[correct_answers[i].split(";")[0].split(" ")[1]
                       for i in range(len(correct_answers[:60-(20*int(not is_60_question_sim))]))]]
        for col_num, data in enumerate(_1_header):
            worksheet.write(1, col_num + 3, data, workbook.add_format({'border': 1,
                                                                       'align': 'center',
                                                                       'valign': 'vcenter',
                                                                       "color": "white",
                                                                       "bold": 1,
                                                                       "bg_color": "#4287F5"}))
        # for percentage mod *range(60)
        _3_header = ["Posizione", "ID", "Punteggio", *[0] * 0*(60-(20*int(not is_60_question_sim)))]
        for col_num, data in enumerate(_3_header):
            worksheet.write(3, col_num, data, workbook.add_format({'bold': 1,
                                                                   'border': 1,
                                                                   'align': 'center',
                                                                   'valign': 'vcenter'}))

        worksheet.write(f'A{placement + v_delta}', f'{placement}', workbook.add_format({'border': 1,
                                                                                        'align': 'center',
                                                                                        'valign': 'vcenter', }))
        worksheet.write(f'B{placement + v_delta}', f'{user.index}', workbook.add_format({'border': 1,
                                                                                         'align': 'center',
                                                                                         'valign': 'vcenter', }))
        worksheet.write(f'C{placement + v_delta}', f'{user.score}', workbook.add_format({'border': 1,
                                                                                         'align': 'center',
                                                                                         'valign': 'vcenter', }))

        h_delta = 3
        for number in range(h_delta, 60 + h_delta-(20*int(not is_60_question_sim))):
            worksheet.write(placement + v_delta - 1, number,
                            f'{user.sorted_user_answer_dict[number + 1 - h_delta]}',
                            formats[round(abs(user.score_list[number - h_delta]) + 0.2)])