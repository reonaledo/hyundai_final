import os
import numpy as np
import pandas as pd
import sys
import pickle
import gzip
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tqdm


def get_file_list(data_list):
    """
    target 파일명 리스트 분리
    (공정id가 'xx_x'로 되어있는거 처리하고 target파일 별로 alarm파일에서 매칭되는 alarm 내역 찾기 위함)

    :param list data_list: 데이터 파일명들.
    :return: target_list
    """

    split = [name.split(".")[0].split("_") for name in data_list]
    return pd.DataFrame(split)


def get_machine_list(target_list):
    """
    타겟파일들에 해당하는 unique한 설비 아이디 찾기
    :param target_list:
    :return:
    """

    mach_id_list = []

    for mach_id in target_list[2]:
        mach_id = mach_id[1:]
        if mach_id[0] == "0":
            mach_id = mach_id[1:]
        else:
            pass
        mach_id_list.append(mach_id)

    mach_id_list = list(set(mach_id_list))
    mach_id_list.sort()
    return mach_id_list


def get_done_files(base_folder):
    done = []
    for (path, dir, files) in os.walk(base_folder):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '':
                done.append(filename)

    done_file_list = get_file_list(done)
    done_machine_id = get_machine_list(done_file_list)
    return done_machine_id


def get_file_info(mach_id, target_list, line_info):
    """
    설비에 해당하는 타겟파일명 도출

    :param int mach_id: 설비아이디 리스트
    :param list target_list: 타겟파일 리스트
    :param pd.DataFrame line_info: line과 mach_id 매핑 테이블
    :return: 라인 이름, 설비id, 타겟파일명
    """
    def get_file_name(splited_file):
        file_name = splited_file[0] + "_" + splited_file[1] + "_" + \
                     splited_file[2] + "_" + splited_file[3] + "_" + \
                     splited_file[4]
        return file_name

    matched_file = []
    matched_file_name = []
    for i, target_file in enumerate(target_list[2]):
        if str(target_file[1:]) == mach_id:
            matched_file.append(list(target_list.iloc[i,:]))
            matched_file_name.append(get_file_name(target_list.iloc[i,:]))

    line = line_info[line_info["Mach_ID"] == int(mach_id)]["Line"].values[0].lower()
    matched_file.sort()
    matched_file_name.sort()

    return line, matched_file, matched_file_name


def get_matched_alarm_record(matched_file, mach_id, alarm_record, start_year=2020, end_year=2020):
    """
    mach_id와 시간(일자)정보로 타겟파일에 해당하는 알람 기록 찾기

    :param list matched_file: 타겟파일 list
    :param string mach_id: 설비id
    :param pd.DataFrame alarm_record: 전체 알람기록파일
    :param int start_year: 타겟파일의 기록 시작 연도
    :param int start_year: 타겟파일의 기록 종료 연도
    :return: alarm
    """

    periods_start = [file[3] for file in matched_file]
    periods_end = [file[4] for file in matched_file]
    start_year, end_year = str(start_year), str(end_year)

    periods_start_dt = pd.Series([pd.to_datetime(start_year + period) for period in periods_start])
    periods_end_dt = pd.Series([pd.to_datetime(end_year + period) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) for period in periods_end])

    period_dt = [periods_start_dt.min(), periods_end_dt.max()]

    period_mask = (alarm_record['START_TIME'] > period_dt[0]) & (alarm_record['START_TIME'] <= period_dt[1])

    alarm = alarm_record[(alarm_record["MACH_ID"] == int(mach_id)) & period_mask]

    period_dt = [str(periods_start_dt.min()), str(periods_end_dt.max())]
    return alarm, period_dt


def split_alarm_normal(alarm):
    """
    안씀
    :param alarm:
    :return:
    """
    normal_idx = alarm['ALARM_ID']=='NORMAL'
    alarm_idx = alarm['ALARM_ID']!='NORMAL'
    normal = alarm[normal_idx]
    alarm = alarm[alarm_idx]

    return normal, alarm


def read_concat_file(path_x, filename_list):
    """
    read data files and concat.
    set "COLLECT_TIME" as pd.datatime

    :param string path_x: 데이터폴더 주소
    :param string filename_list: 파일명의 list
    :return: pd.DataFrame
    """

    dat = pd.DataFrame()
    for filename in filename_list:
        dat = pd.concat([dat, pd.read_csv(path_x + filename + ".csv")])


    # 중간에 칼럼명 껴있는것 제거
    dat = dat.drop(dat[dat["COLLECT_TIME"] == 'COLLECT_TIME'].index).reset_index(drop=True)

    dat["COLLECT_TIME"] = pd.to_datetime(dat["COLLECT_TIME"])
    dat.iloc[:, 1:] = dat.iloc[:, 1:].astype('float')

    return dat


def alarm_labeling(dat_t, alarm, normal,
                   hours_normal,
                   hours_alarm1,
                   hours_alarm2,
                   steps_before_alarm=0, steps_after_alarm=0):
    """
    target 데이터에 알람기록 레이블링.
    0: 정상구간
    1: 알람1
    2: 알람2
    3: 알람3
    -1: 알람중

    dat_t = read_file(path_x, target_filename):param pd.DataFrame dat_t: 타겟파일
    :param pd.DataFrame alarm:
    :param float hours_before_alarm: 시간단위 전조구간 정의
    :param float hours_after_alarm: 시간단위 이상구간 정의
    :param int steps_before_alarm: 타임스텝단위 전조구간 정의
    :param int steps_after_alarm: 타임스텝단위 이상구간 정의
    :return:
    """
    alarm_start_times = alarm["START_TIME"].values
    alarm_end_times = alarm["END_TIME"].values

    alarm_num = len(alarm_start_times)
    alarm.ALARM_ID.apply(str)
    alarm_IDs = list(alarm["ALARM_ID"].values)
    alarm_IDs = [str(id_)+'_' for id_ in alarm_IDs]
    alarm_IDs_unique = list(set(alarm_IDs))
    alarm_num_unique = len(alarm_IDs)

    dat_alarm = [pd.DataFrame(0, index=range(0, len(dat_t)), columns=[str(alarm_IDs[i])]) for i in range(alarm_num)]

    # 한 타겟파일에 알람 여러번 생겼을 수 있는거 고려하였음
    for idx in range(alarm_num):
        alarm_ID = alarm_IDs[idx]
        alarm_start_time = pd.to_datetime(alarm_start_times[idx])
        alarm_end_time = pd.to_datetime(alarm_end_times[idx])

        if steps_before_alarm == 0:
            time_normal = alarm_start_time - pd.Timedelta(hours=hours_normal)
            time_alarm1 = alarm_start_time - pd.Timedelta(hours=hours_alarm1)
            time_alarm2 = alarm_start_time - pd.Timedelta(hours=hours_alarm2)

            idxs_alarm1 = (dat_t["COLLECT_TIME"] > time_normal) & (dat_t["COLLECT_TIME"] < time_alarm1) # 1
            idxs_alarm2 = (dat_t["COLLECT_TIME"] > time_alarm1) & (dat_t["COLLECT_TIME"] < time_alarm2) # 2
            idxs_alarm3 = (dat_t["COLLECT_TIME"] > time_alarm2) & (dat_t["COLLECT_TIME"] < alarm_start_time) # 3
            idxs_alarm4 = (dat_t["COLLECT_TIME"] > alarm_start_time) & (dat_t["COLLECT_TIME"] < alarm_end_time) # -1

            dat_alarm[idx][alarm_ID][idxs_alarm1] = 1
            dat_alarm[idx][alarm_ID][idxs_alarm2] = 2
            dat_alarm[idx][alarm_ID][idxs_alarm3] = 3
            dat_alarm[idx][alarm_ID][idxs_alarm4] = -1

    for i in range(alarm_num):
        dat_t = pd.concat((dat_t, dat_alarm[i]), axis=1)

    return dat_t


def unify_time_unit_before_labeling(dat, unify_sec, idx_logging=False, verbose=False):
    """
    동일 시간단위로 통합 (시간 단위 내 값들을 평균취함)
    :param dat:
    :param unify_sec: 몇초로 통합할 것인지
    :param idx_logging: 옵션
    :param verbose: 옵션
    :return:
    """

    def to_micsec(time_gap):
        return (time_gap.days * 24 * 60 * 60 + time_gap.seconds) * 1000000 + time_gap.microseconds

    from_, to_, time_gap_micsec, unified_dat_x_idx = 0, 0, 0, 0
    from_time = dat["COLLECT_TIME"][from_]
    idx_log = []
    jump_idx = []
    unified_dat_x = []

    finished = False
    while to_ < len(dat) - 1:
        to_ += 1

        if to_ == len(dat) - 1:
            finished = True
            time_gap_micsec = 0

        if not finished:
            time_gap = dat["COLLECT_TIME"][to_] - from_time
            time_gap_micsec = to_micsec(time_gap)

        if time_gap_micsec < unify_sec * 1000000:
            pass
        else:
            if idx_logging: idx_log.append([from_, to_])
            if verbose and unified_dat_x_idx % 100 == 0: print(from_ / len(dat))

            dat_range = dat.iloc[from_:to_, :]
            unified_time = dat_range.iloc[:, :2].max()
            unified_mach_info = dat_range.iloc[:, 2:10].median()
            unified_target = dat_range.iloc[:, 10:].mean()
            unified = pd.concat((unified_time, unified_mach_info, unified_target))
            unified_dat_x.append(unified)

            # check time-jump
            if not finished:
                unified_dat_x_idx += 1
                next_unified_gap = dat["COLLECT_TIME"][to_] - dat["COLLECT_TIME"][to_ - 1]
                next_unified_gap_micsec = to_micsec(next_unified_gap)
                if next_unified_gap_micsec > unify_sec * 1000000:
                    jump_idx.append(unified_dat_x_idx)  # 추후에 jump_idx를 포함하는 윈도우는 이를 첫 인덱스로 갖는 경우를 제외하고 제거하면 됨.

                from_ = to_
                from_time = dat["COLLECT_TIME"][from_]

    unified_dat_x = pd.DataFrame(unified_dat_x)

    unified_dat_x.iloc[:, 2:] = unified_dat_x.iloc[:, 2:].astype('float')

    return unified_dat_x, jump_idx, idx_log


def make_multilable(u_dat_y):
    unique_alarm_id = list(set(list(u_dat_y.columns)))
    unique_alarm_id.sort()

    alarm_col_names = [[alarm_id+'1',alarm_id+'2',alarm_id+'3', alarm_id+'-1'] for alarm_id in unique_alarm_id]
    alarm_col_names = sum(alarm_col_names, [])
    multilabel = pd.DataFrame(0, index=range(0, len(u_dat_y)), columns=alarm_col_names)

    for idx in tqdm.tqdm(range(len(u_dat_y))):

        for alarm_id in unique_alarm_id:
            labels = list(set(pd.Series(u_dat_y[alarm_id].iloc[idx]).values))
            if 1 in labels:
                multilabel[alarm_id + '1'][idx] = 1
            if 2 in labels:
                multilabel[alarm_id+'2'][idx] = 1
            if 3 in labels:
                multilabel[alarm_id + '3'][idx] = 1
            if -1 in labels:
                multilabel[alarm_id + '-1'][idx] = 1

    # check error value 72057594037927936
    error_col = multilabel.columns[(multilabel > 3).sum() != 0]
    for i in range(len(error_col)):
        (multilabel[error_col[i]])[(multilabel[error_col[i]]) > 3] = 0

    return multilabel


def windowing(dat_t_x, dat_t_y, jump_idx
              , window_size, shift_size):
    """
    윈도윙함. 수집간격이 통합단위보다 큰 경우 윈도윙을 중단하고 그 다음 시점부터 다시 윈도윙 수행
    윈도우 내 과반수의 레이블을 윈도우의 레이블로 지정

    :param pd.DataFrame dat_t_x:
    :param pd.Series dat_t_y:
    :param list jump_idx: 수집간격 큰 시점
    :param int window_size:
    :param int shift_size:
    :return: X: (n_window, n_sensor, window_size) , y_label: (n_window, )
    """
    alarm_columns = list(dat_t_y.columns)
    dat_t_x = np.array(dat_t_x)
    dat_t_y = np.array(dat_t_y)

    # create windows
    X = []
    y = []

    from_, to_ = 0, 0
    while to_ < len(dat_t_x) - 1:
        to_ = from_ + window_size
        if to_ > len(dat_t_x): break;

        # 수집간격 큰 시점 고려
        window_range = set(range(from_ + 1, to_))
        if len(window_range & set(jump_idx)) != 0:
            from_ = list(window_range & set(jump_idx))[-1]
        else:
            X.append(np.array(dat_t_x[from_:to_].transpose()))
            y.append(np.array(dat_t_y[from_:to_].transpose()))
            from_ = from_ + shift_size

    # to 3D array
    X = np.array(X)  # .reshape(-1, n_sensors, window_size)

    # set window label
    y_label = []

    for each_window in y:
        labels = []
        for alarm_idx in range(each_window.shape[0]):
            count_00 = sum(each_window[alarm_idx] == -1)
            count_0 = sum(each_window[alarm_idx] == 0)
            count_1 = sum(each_window[alarm_idx] == 1)
            count_2 = sum(each_window[alarm_idx] == 2)
            count_3 = sum(each_window[alarm_idx] == 3)
            count_list = [count_00, count_0, count_1, count_2, count_3]
            label = count_list.index(max(count_list))-1

            labels.append(label)
        y_label.append(labels)

    y_label = pd.DataFrame(y_label, columns=alarm_columns)
    return X, y_label


def save_pickle(line_folder_list, period, matched_file_name, line, is_raw, X, y):
    """
    save as pickle

    :param path:
    :param matched_file_name:
    :param X:
    :param y:
    :return:
    """

    filename = matched_file_name[0][:-10] + '_' + period[0][:10] + '_' + period[1][:10]
    try:
        if line == "block":
            path = line_folder_list[0]
        elif line == "head":
            path = line_folder_list[1]
        elif line == "crank":
            path = line_folder_list[2]
        else:
            raise Exception(line)
    except Exception as e:
        print('존재하지 않는 라인입니다. 입력받은 라인 명: ', e)

    if is_raw:
        path = path + 'unified_raw/'
    else:
        path = path + 'window/'

    with gzip.open(path + filename + "_x", 'wb') as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
    with gzip.open(path + filename + "_y", 'wb') as f:
        pickle.dump(y, f, pickle.HIGHEST_PROTOCOL)

    return
