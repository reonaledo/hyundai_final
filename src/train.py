from src.utils.PreprocessingUtils import *
from src.utils.InferenceUtils import *
import configparser

import os

def train():

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    dev = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(dev[0], True)

    # Path Setting #
    config = configparser.ConfigParser()
    config.read('C:/Users/Dabin/PycharmProjects/hyundai_final/config.ini', encoding='utf-8')

    # set data path
    data_path = config['TRAIN_PATH']['result_path']

    # set model path
    model_path = config['TRAIN_PATH']['model_path']

    # set save path
    save_path = config['TRAIN_PATH']['save_path']


    # hyper parameter setting


    scenario = config['MODEL']['scenario']
    model_name = config['MODEL']['model_name']
    EPOCH = int(config['MODEL']['EPOCH'])
    BATCH_SIZE = int(config['MODEL']['BATCH_SIZE'])
    TEST_BATCH_SIZE = int(config['MODEL']['TEST_BATCH_SIZE'])
    BUFFER_SIZE = int(config['MODEL']['BUFFER_SIZE'])
    SEED = int(config['MODEL']['SEED'])
    init_lr = float(config['MODEL']['init_lr'])
    patience = int(config['MODEL']['patience'])
    decay_rate = float(config['MODEL']['decay_rate'])
    alpha = int(config['MODEL']['alpha'])

    # 1. Train step
    train_x, valid_x, test_x, train_y, valid_y, test_y, class_names = preprocessing_train(data_path = data_path,
                                                                                    main_alarm = None,
                                                                                    model_path = model_path,
                                                                                    scenario = scenario)

    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(BUFFER_SIZE, SEED, True).batch(BATCH_SIZE)
    valid_data = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(BATCH_SIZE)

    # Training CNN Model & Save Weights
    if model_name == 'densenet':
        CNN = densenet_model(train_x.shape[1:], train_y.shape[1], False, False)
    elif model_name == 'resnet':
        CNN = resnet_model(train_x.shape[1:], train_y.shape[1], False, False)
    elif model_name == 'vggnet':
        CNN = vggnet_model(train_x.shape[1:], train_y.shape[1], False, False)

    network_opt = tf.optimizers.Adam(init_lr)
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * decay_rate

    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                tf.keras.callbacks.LearningRateScheduler(scheduler)]

    CNN.compile(loss="binary_crossentropy", optimizer=network_opt, metrics=["accuracy"])
    CNN.fit(x=train_data, validation_data=valid_data, epochs=EPOCH, verbose=1, callbacks = callback)
    CNN.save_weights(os.path.join(model_path,"{}.h5".format(scenario)))

    # Calculate Sigmoid score Threshold for each Class
    train_data = tf.data.Dataset.from_tensor_slices(train_x).batch(TEST_BATCH_SIZE)
    train_pred_scores = get_class_prob(CNN, train_data)
    threshold = get_threshold(train_pred_scores, train_y, CNN, alpha)

    # 2. Inference step
    inference(test_x, test_y, TEST_BATCH_SIZE, class_names, CNN, threshold, save_path, scenario)

    # 3. save train data information
    np.savez(os.path.join(save_path, "{}_info_train.npz".format(scenario)),
             x_shape = np.array(train_x.shape[1:]),
             y_shape = np.array(train_y.shape[1]),
             threshold = threshold,
             class_names = class_names)



