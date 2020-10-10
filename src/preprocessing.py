from utils import *

# Path Setting #

# set data path
path_data = "//163.152.174.30/NewShare/1. 프로젝트/2020_현대자동차/data/"

# set output(result) path
result_path = '//163.152.174.30/NewShare/1. 프로젝트/2020_현대자동차/data/preprocessed/'

# set plant name
plant_name = "세타"
# plant_name = "누신U"
# plant_name = "신R"

# set line-machine mapping data path
line_info_path = '//163.152.174.30/NewShare/1. 프로젝트/2020_현대자동차/data/references/설비정보(4-5월)/● (2차) 신R(110)_설비 가공 상태, 알람, 설비 정보 데이터(4-5월 업데이트).xlsx'


# Hyper Parameter Setting #

###### labeling parameters ######
hours_normal = 24 * 7 # 알람시점 몇 시간 전까지 전조구간으로 labeling 할 것인가
hours_alarm1 = 24 * 3 # 알람시점 몇 시간 전까지 이상구간1으로 labeling 할 것인가
hours_alarm2 = 24 * 1 # 알람시점 몇 시간 전까지 이상구간2으로 labeling 할 것인가

##### unifying parameters #####
unify_sec = 1 # 스텝간 시간차이 통합 간격 (단위: 초)

##### windowing parameters #####
window_size = 60
shift_size = 30
threshold = None  #미사용


def main():

    # 경로 설정 작업
    path_x = path_data + "raw/test_세타" + "/"
    path_y = path_data + "alarm/"

    data_list = os.listdir(path_x)

    alarm_record = pd.read_excel(path_y + "★ 설비 알람 Test (6-7월)_20200910" + ".xlsx")
    line_info = pd.read_excel(line_info_path, header=16).loc[:, ['Line', 'Mach_ID']]
    target_list = get_file_list(data_list)
    target_machine_list = get_machine_list(target_list)

    base_folder = result_path + 'result ' + datetime.now().strftime('%m-%d %H_%M') + '/'
    ###
    # base_folder = '//163.152.174.30/NewShare/1. 프로젝트/2020_현대자동차/data/preprocessed/result 08-05 14_34/'
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

    # interest = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
    # # done_machine_id = get_done_files(base_folder)
    # done_machine_id = [124, 136]

    # 전처리작업 시작
    for mach_id in target_machine_list:
        # if str(mach_id) not in done_machine_id and int(mach_id) in interest:
        if 1:
            # 파일 정보 획득
            line, matched_file, matched_file_name = get_file_info(mach_id, target_list, line_info)
            print('processing file name: ', matched_file_name)

            # 타겟파일에 매칭되는 알람 기록 참조
            alarm, period = get_matched_alarm_record(matched_file, mach_id, alarm_record)

            # # 알람파일에서 정상구간 분리
            # normal, alarm = split_alarm_normal(alarm)

            # 로드 및 설비ID에 매칭된 파일 하나로 합치기
            dat_t = read_concat_file(path_x, matched_file_name)

            # 시간 단위 통합
            u_dat_t, jump_idx_t, idx_log_t = unify_time_unit_before_labeling(dat_t, unify_sec=unify_sec, idx_logging=False,
                                                                          verbose=True)

            del dat_t

            # 알람 레이블링
            u_dat_t = alarm_labeling(u_dat_t, alarm, None, hours_normal, hours_alarm1, hours_alarm2)

            u_dat_t_x = u_dat_t.iloc[:, :52]
            u_dat_t_y = u_dat_t.iloc[:, 52:]

            u_multilabel = make_multilable(u_dat_t_y)

            # with gzip.open(matched_file_name[0]+"_ji", 'wb') as f:
            #     pickle.dump(jump_idx_t, f, pickle.HIGHEST_PROTOCOL)

            # 윈도윙 전 저장 (/raw)
            save_pickle(line_folder_list, period, matched_file_name, line, True, u_dat_t_x, u_multilabel)

            # 윈도윙
            X_t, y_t = windowing(u_dat_t_x, u_dat_t_y, jump_idx_t, window_size=window_size, shift_size=shift_size)

            w_multilabel = make_multilable(y_t)

            # 윈도윙 후 저장 (/window)
            save_pickle(line_folder_list, period, matched_file_name, line, False, X_t, w_multilabel)

if __name__ == "__main__":
    main()

