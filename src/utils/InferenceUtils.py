import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import jaccard_score, f1_score, multilabel_confusion_matrix

def get_class_prob(CNN, dataset):
    '''
    get model's sigmoid output from input data
    :param dataset: input data for inference
    :return: Sigmoid score from each node of network's last layer : range(0,1)
    '''
    pred_scores = []
    for x in dataset:
        class_prob = CNN(x, training=False)
        pred_scores.append(class_prob.numpy())
    pred_scores = np.concatenate(pred_scores, axis=0)
    return pred_scores

def make_prediction(_scores, _T):
    '''
    get predictive label Y from using sigmoid output and threshold
    :param _scores: Sigmoid score
    :param _T: Threshold of sigmoid score
    :return: Predicted Label Y
    '''
    _labels = _scores - _T > 0
    _labels = np.column_stack([_labels,_labels.sum(1)==0])
    return _labels

def get_threshold(train_pred_scores, train_y, CNN, alpha):
    '''
    get each class classification threshold from true train data
    :param train_pred_scores: sigmoid score from train x data
    :param train_y: true train y
    :param CNN: model to predict value
    :param alpha: sigma-level (train data standard deviation level)
    :return:
    '''
    threshold = []
    for c in range(train_y.shape[1]):
        class_scores = train_pred_scores[train_y[:, c] == 1, c]
        class_scores = np.concatenate([class_scores, 2 - class_scores])
        class_std = np.std(class_scores)
        threshold.append(np.max([0.5, 1 - alpha * class_std]))
    threshold = np.array(threshold)
    return threshold

def inference(test_x, test_y, TEST_BATCH_SIZE, class_names, CNN, threshold, save_path, scenario):
    '''
    :param test_x, test_y: test data
    :param TEST_BATCH_SIZE: test batch size for model
    :param class_names: target alarm class
    :param CNN: model
    :param threshold: class sigmoid score threshold
    :param save_path: save path for result
    :param scenario: scenario name to identify the model
    :return:
    '''
    # append unknown class label
    test_y_extended = np.hstack((test_y, np.zeros(len(test_y)).reshape(-1, 1)))

    # Calculate Test sigmoid score & Predicted label
    test_data = tf.data.Dataset.from_tensor_slices(test_x).batch(TEST_BATCH_SIZE)
    test_pred_scores = get_class_prob(CNN, test_data)
    test_pred_labels = make_prediction(test_pred_scores, threshold)

    # Calculate Classification Performace
    f = open(os.path.join(save_path, "{}.txt".format(scenario)), 'w')
    f.write('\n--------------Test--------------\n')
    f.write('Jaccard Score : {}\n'.format(jaccard_score(test_y_extended,
                                                        test_pred_labels,
                                                        average='samples')))
    f.write('F1 Score : {}\n'.format(f1_score(test_y_extended,
                                              test_pred_labels,
                                              average='samples')))

    f.write('Multi-label Confusion Matrix\n')
    for i in range(test_y_extended.shape[1]):
        if i == test_y_extended.shape[1]-1:
            f.write('Class : Unknown\n')
            f.write(str(multilabel_confusion_matrix(test_y_extended,
                                                    test_pred_labels)[i]) + '\n')
        else:
            f.write('Class : {}\n'.format(class_names[i]))
            f.write(str(multilabel_confusion_matrix(test_y_extended,
                                                    test_pred_labels)[i]) + '\n')

    # Calculate Classification Performace for each Multi-label Pattern
    test_y_df = pd.DataFrame(test_y, columns=class_names)
    target_pattern = test_y_df.drop_duplicates().reset_index()
    target_index = []
    for i in range(target_pattern.shape[0]):
        target_index.append((test_y_df == target_pattern.iloc[i]).sum(1) == test_y_df.shape[1])
    for i, a in enumerate(target_index):
        target_test_data = tf.data.Dataset.from_tensor_slices(test_x[a]).batch(TEST_BATCH_SIZE)
        target_test_pred_scores = get_class_prob(CNN, target_test_data)
        target_test_pred_labels = make_prediction(target_test_pred_scores, threshold)
        f.write('\n--------------Score for each Pattern--------------\n')
        f.write('\n Pattern : {}\n'.format(target_pattern.columns[target_pattern.iloc[i] == 1].values))
        f.write('# of this pattern : {}\n'.format(len(test_x[a])))
        f.write(
            'Jaccard Score : {}\n'.format(jaccard_score(test_y_extended[a],
                                                        target_test_pred_labels,
                                                        average='samples')))
        f.write('F1 Score : {}\n'.format(f1_score(test_y_extended[a],
                                                  target_test_pred_labels,
                                                  average='samples')))
    f.close()

    class_names_new = list(class_names)
    class_names_new.extend(['Unknown'])

    # save predicted value and true test data
    pd.DataFrame(test_y_extended, columns=class_names_new).to_csv(
        os.path.join(save_path, "{}_true.csv".format(scenario)), index=False)
    pd.DataFrame(test_pred_labels, columns=class_names_new).to_csv(
        os.path.join(save_path, "{}_pred.csv".format(scenario)), index=False)