from utils.PreprocessingUtils import *
import configparser

def main():
    config = configparser.ConfigParser()
    config.read('C:/Users/Dabin/PycharmProjects/hyundai_final/config.ini', encoding='utf-8')

    data_path = config['PATH']['data_path']
    result_path = config['PATH']['result_path']
    line_info_path = config['PATH']['line_info_path']
    alarm_path = config["PATH"]['alarm_path']

    plant_name = config['PLANT']['plant_name']

    hours_alarm1 = 24 * int(config['PREPROCESSING']['days_alarm1'])
    hours_alarm2 = 24 * int(config['PREPROCESSING']['days_alarm2'])

    unify_sec = int(config['PREPROCESSING']['unify_sec'])

    window_size = int(config['PREPROCESSING']['window_size'])
    shift_size = int(config['PREPROCESSING']['shift_size'])

    # 경로 설정 작업
    path_x = data_path + "raw/공장_세타(4,5월)" + "/"

    data_list = os.listdir(path_x)

    alarm_record = pd.read_excel(alarm_path)
    line_info = pd.read_excel(line_info_path, header=16).loc[:, ['Line', 'Mach_ID']]
    target_list = get_file_list(data_list)
    target_machine_list = get_machine_list(target_list)

    base_folder = result_path + 'result ' + datetime.now().strftime('%m-%d %H_%M') + '/'
    ###
    b = base_folder + plant_name + "_BLOCK/"
    h = base_folder + plant_name + "_HEAD/"
    c = base_folder + plant_name + "_CRANK/"
    line_folder_list = [b, h, c]

    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
        for x in line_folder_list:
            os.mkdir(x)
            os.mkdir(x + 'unified_raw')
            os.mkdir(x + 'window')

    print("저장경로: ", base_folder)

    # 전처리작업 시작
    for mach_id in target_machine_list:

        # 파일 정보 획득
        line, matched_file, matched_file_name = get_file_info(mach_id, target_list, line_info)
        print('processing file name: ', matched_file_name)

        # 타겟파일에 매칭되는 알람 기록 참조
        alarm, period = get_matched_alarm_record(matched_file, mach_id, alarm_record)

        # 로드 및 설비ID에 매칭된 파일 하나로 합치기
        dat_t = read_concat_file(path_x, matched_file_name)

        # 시간 단위 통합
        u_dat_t, jump_idx_t, idx_log_t = unify_time_unit_before_labeling(dat_t, unify_sec=unify_sec, idx_logging=False,
                                                                      verbose=True)

        del dat_t

        # 알람 레이블링
        u_dat_t = alarm_labeling(u_dat_t, alarm, hours_alarm1, hours_alarm2)

        u_dat_t_x = u_dat_t.iloc[:, :52]
        u_dat_t_y = u_dat_t.iloc[:, 52:]

        del u_dat_t

        # 윈도윙
        X_t, y_t = windowing(u_dat_t_x, u_dat_t_y, jump_idx_t, window_size=window_size, shift_size=shift_size)

        w_multilabel = make_multilable(y_t)

        # 윈도윙 후 저장 (/window)
        save_pickle(line_folder_list, period, matched_file_name, line, False, X_t, w_multilabel)

if __name__ == "__main__":
    main()

