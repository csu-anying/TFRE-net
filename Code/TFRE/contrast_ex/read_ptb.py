import numpy as np
import wfdb
import os
from wfdb import processing

tw12_lead = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
fs = 1e3
fs_resampled = 500
samples_num = 600


def resampled(sig):
    sig_resampled = []
    for lead in sig.T:
        lead_resampled, _ = wfdb.processing.resample_sig(lead, fs, fs_resampled)
        sig_resampled.append(np.array(lead_resampled))
    return sig_resampled


def segmented(sig):
    segment = []
    sig = np.array(sig)
    for j in range(int(sig.shape[1] / samples_num)):
        segment_lead = sig[:, samples_num * j:samples_num * j + samples_num]
        segment.append(segment_lead)
    return segment


def get_segments(path, lead_num):
    sig, fields = wfdb.rdsamp(path, channels=lead_num)
    sig_resampled = resampled(sig)
    sig_segmented = segmented(sig_resampled)
    sig_processed = np.array(sig_segmented)
    sig_processed = np.swapaxes(sig_processed, 1, 2)
    return sig_processed


def load_ptb_7(filepaths):
    label = []
    data = []
    for i, path in enumerate(filepaths):
        _, fields = wfdb.rdsamp(path)
        if 'Healthy control' in fields['comments'][4]:
            segments = get_segments(path, tw12_lead)
            data.extend(segments)
            for i in range(segments.shape[0]):
                label.append(6)
        else:
            if 'Myocardial infarction' in fields['comments'][4]:
                if 'anterior' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(0)
                elif 'antero-septal' in fields['comments'][5] or 'antero-septo-lateral' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(1)
                elif 'antero-lateral' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(2)
                elif 'inferior' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(3)
                elif 'infero-latera' in fields['comments'][5] or 'infero-lateral' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(4)
                elif 'infero-postero-lateral' in fields['comments'][5] or 'infero-posterior' in fields['comments'][
                    5] or 'lateral' in fields['comments'][5] or 'posterior' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(5)

    return data, label


def load_ptb_5(filepaths):
    label = []
    data = []
    for i, path in enumerate(filepaths):
        _, fields = wfdb.rdsamp(path)
        if 'Healthy control' in fields['comments'][4]:
            segments = get_segments(path, tw12_lead)
            data.extend(segments)
            for i in range(segments.shape[0]):
                label.append(4)
        else:
            if 'Myocardial infarction' in fields['comments'][4]:
                if 'anterior' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(0)
                elif 'antero-septal' in fields['comments'][5] or 'antero-septo-lateral' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(1)
                elif 'antero-lateral' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(2)
                elif 'inferior' in fields['comments'][5] or 'infero-latera' in fields['comments'][
                    5] or 'infero-lateral' in fields['comments'][5] or 'infero-postero-lateral' in fields['comments'][
                    5] or 'infero-posterior' in fields['comments'][5] or 'lateral' in fields['comments'][
                    5] or 'posterior' in fields['comments'][5]:
                    segments = get_segments(path, tw12_lead)
                    data.extend(segments)
                    for i in range(segments.shape[0]):
                        label.append(3)

    return data, label


# 输入是数据文件路径组成的列表，功能是获取十二导联的数据和二分类标签，数据文件的comments属性写着标签
def load_ptb_2(filepaths):
    label = []
    data = []
    for i, path in enumerate(filepaths):
        _, fields = wfdb.rdsamp(path)
        if 'Healthy control' in fields['comments'][4]:
            segments = get_segments(path, tw12_lead)
            data.extend(segments)
            for i in range(segments.shape[0]):
                label.append(0)
        else:
            if 'Myocardial infarction' in fields['comments'][4]:
                segments = get_segments(path, tw12_lead)
                data.extend(segments)
                for i in range(segments.shape[0]):
                    label.append(1)

    return data, label


def read_ptb(num_class, fold_num):
    files_subjects = []
    with open('../../ptbdb/Fold') as fp:  # 包含需要打开的数据文件夹名称
        # with open('../MCA-net/ptbdb/Fold') as fp:
        lines = fp.readlines()
    # 读取数据文件夹路径
    for file in lines:
        files_subjects_path = "../../ptbdb/" + file[:10] + "/"
        files_subjects.append(files_subjects_path)
    # 挑选用于验证的折数
    train, val = [], []
    if (int(fold_num) == 0):
        train = files_subjects
        val = files_subjects
    elif (int(fold_num) == 1):
        train = files_subjects[33:]
        val = files_subjects[0:33]
    elif (int(fold_num) == 2):
        train = files_subjects[0:33] + files_subjects[68:]
        val = files_subjects[33:68]
    elif (int(fold_num) == 3):
        train = files_subjects[0:68] + files_subjects[103:]
        val = files_subjects[68:103]
    elif (int(fold_num) == 4):
        train = files_subjects[0:103] + files_subjects[141:]
        val = files_subjects[103:141]
    elif (int(fold_num) == 5):
        train = files_subjects[0:141]
        val = files_subjects[141:]
    # 添加文件路径
    fold_train = []
    fold_val = []
    for file in train:
        for root, dirs, files in os.walk(file):
            for i in files:
                if os.path.splitext(i)[1] == ".hea":
                    file_path = str(file) + str(i[0:8])
                    fold_train.append(file_path)

    for file in val:
        for root, dirs, files in os.walk(file):
            for i in files:
                if os.path.splitext(i)[1] == ".hea":
                    file_path = str(file) + str(i[0:8])
                    fold_val.append(file_path)

    X_train, Y_train, X_val, Y_val = [], [], [], []

    if (int(num_class) == 2):
        X_train, Y_train = load_ptb_2(fold_train)
        X_val, Y_val = load_ptb_2(fold_val)
    elif (int(num_class) == 5):
        X_train, Y_train = load_ptb_5(fold_train)
        X_val, Y_val = load_ptb_5(fold_val)
    elif (int(num_class) == 7):
        X_train, Y_train = load_ptb_7(fold_train)
        X_val, Y_val = load_ptb_7(fold_val)

    X_train = np.array(X_train)
    y_train = np.array(Y_train)

    X_test = np.array(X_val)
    y_test = np.array(Y_val)

    def re_format(X):
        X = X.transpose(2, 0, 1).tolist()
        # print('1'+"\n")
        X_te = []
        for i, a in enumerate(X):
            c = []
            for j, b in enumerate(a):
                c.append(np.array(b))
            X_te.append(np.array(c))
        return X_te

    return re_format(X_train), y_train, re_format(X_test), y_test
