from src.utils.PreprocessingUtils import *
from src.utils.InferenceUtils import *
from src.utils.model import *

import configparser

def test():
    config = configparser.ConfigParser()
    config.read('C:/Users/Dabin/PycharmProjects/hyundai_final/config.ini', encoding='utf-8')

    data_path = config['TEST_PATH']['data_path']
    line_info_path = config['TRAIN_PATH']['line_info_path']
    model_path = config['TRAIN_PATH']['model_path']
    save_path = config['TRAIN_PATH']['save_path']
    result_path = config['TEST_PATH']['result_path']

    unify_sec = int(config['PREPROCESSING']['unify_sec'])

    window_size = int(config['PREPROCESSING']['window_size'])
    shift_size = int(config['PREPROCESSING']['shift_size'])

    scenario = config['MODEL']['scenario']
    model_name = config['MODEL']['model_name']
    TEST_BATCH_SIZE = int(config['MODEL']['TEST_BATCH_SIZE'])

    data_list = os.listdir(data_path)
    line_info = pd.read_excel(line_info_path, header=16).loc[:, ['Line', 'Mach_ID']]
    target_list = get_file_list(data_list)
    target_machine_list = get_machine_list(target_list)

    # 전처리작업 시작
    for mach_id in target_machine_list:

        # 파일 정보 획득
        line, matched_file, matched_file_name = get_file_info(mach_id, target_list, line_info)
        print('processing file name: ', matched_file_name)

        # 로드 및 설비ID에 매칭된 파일 하나로 합치기
        dat_t = read_concat_file(data_path, matched_file_name)

        # 시간 단위 통합
        u_dat_t, jump_idx_t, idx_log_t = unify_time_unit(dat_t, unify_sec=unify_sec, idx_logging=False,
                                                                      verbose=True)

        del dat_t

        # 윈도윙
        X = windowing_test(u_dat_t, jump_idx_t, window_size=window_size, shift_size=shift_size)

        del u_dat_t

        # get train data information
        train_infor = np.load(os.path.join(save_path, "{}_info_train.npz".format(scenario)))
        x_shape = train_infor['x_shape']
        y_shape = train_infor['y_shape']
        threshold = train_infor['threshold']
        class_names = train_infor['class_names']
        class_names.extend(['Unknown'])

        with open(os.path.join(model_path, '{}.pkl'.format(scenario)), 'rb') as file:
            scaler = joblib.load(file)

        X = X.transpose(0, 2, 1)
        X = X.reshape(-1, 42)
        X = scaler.transform(X)
        X = X.reshape(-1, 60, 42)
        X = X.transpose(0, 2, 1)

        ##predict 코드
        # Load CNN Model
        if model_name == 'densenet':
            CNN = densenet_model(x_shape, y_shape, False, False)
        elif model_name == 'resnet':
            CNN = resnet_model(x_shape, y_shape, False, False)
        elif model_name == 'vggnet':
            CNN = vggnet_model(x_shape, y_shape, False, False)

        CNN.load_weights(os.path.join(model_path, "{}.h5".format(scenario)))

        inference_data = tf.data.Dataset.from_tensor_slices(X).batch(TEST_BATCH_SIZE)
        inference_pred_scores = get_class_prob(CNN, inference_data)
        inference_pred_labels = make_prediction(inference_pred_scores, threshold)

        ##output 출력
        pd.DataFrame(inference_pred_labels, columns=class_names).to_csv(
            os.path.join(result_path, "{}_pred.csv".format(scenario)), index=False)
