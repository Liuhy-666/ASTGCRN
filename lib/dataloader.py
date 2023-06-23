import pickle

import torch
import numpy as np
import torch.utils.data
from .add_window import Add_Window_Horizon
from .load_dataset import load_st_dataset
from .normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    """
    PEMS Series
    """
    # 1.load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    # 2.normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    # 3.spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    # 4.add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler

# def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
#     """
#     DND-US
#     """
#     # 1.load raw st dataset
#     data = load_st_dataset(args.dataset)        # B, N, D
#     # 2.normalize st data
#     data, scaler = normalize_dataset(data, normalizer, args.column_wise)
#     # 3.add time window
#     x_data, y_data = Add_Window_Horizon(data, args.lag, args.horizon, single)
#     # 4.spilit dataset by days or by ratio
#     if args.test_ratio > 1:
#         x_tra, x_val, x_test = split_data_by_days(x_data, args.val_ratio, args.test_ratio)
#         y_tra, y_val, y_test = split_data_by_days(y_data, args.val_ratio, args.test_ratio)
#     else:
#         x_tra, x_val, x_test = split_data_by_ratio(x_data, args.val_ratio, args.test_ratio)
#         y_tra, y_val, y_test = split_data_by_ratio(y_data, args.val_ratio, args.test_ratio)
#     print('Train: ', x_tra.shape, y_tra.shape)
#     print('Val: ', x_val.shape, y_val.shape)
#     print('Test: ', x_test.shape, y_test.shape)
#     ##############get dataloader######################
#     train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
#     if args.val_ratio == 0:
#         val_dataloader = None
#     else:
#         val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
#     test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
#     return train_dataloader, val_dataloader, test_dataloader, scaler


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
