import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMS03':
        data_path = os.path.join('data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  # one dimensions, traffic flow data
        print(data.shape)  # (26208, 358)
    elif dataset == 'PEMS04':
        data_path = os.path.join('data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  # onley the first dimension, traffic flow data
        print(data.shape)  # (16992, 307)
    elif dataset == 'PEMS07':
        data_path = os.path.join('data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # one dimensions, traffic flow data
        print(data.shape)  # (28224, 883)
    elif dataset == 'PEMS07M':
        data_path = os.path.join('data/PEMS07M/PEMS07M.npz')
        data = np.load(data_path)['data'][:, :, 0]  # one dimensions, traffic speed data
        print(data.shape)  # (12671, 228)
    elif dataset == 'PEMS08':
        data_path = os.path.join('data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  # onley the first dimension, traffic flow data
        print(data.shape)  # (17856, 170)
    elif dataset == 'DND-US':
        data_path = os.path.join('data/DND-US/DND-US.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  # onley the first dimension, traffic flow data
        print(data.shape)  # (313, 53)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    # Load PEMS03 Dataset shaped:  (26208, 358, 1) 1852.0 0.0 179.26077680660433 136.0
    # Load PEMS04 Dataset shaped:  (16992, 307, 1) 919.0 0.0 211.7007794815878 180.0
    # Load PEMS07 Dataset shaped:  (28224, 883, 1) 1498.0 0.0 308.52346223738647 304.0
    # Load PEMS08 Dataset shaped:  (17856, 170, 1) 1147.0 0.0 230.68069424678473 215.0
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


if __name__ == '__main__':
    pemss03 = load_st_dataset('PEMS03')
    pems = pemss03[..., 0]
    print("pems03-1.shape:", pems.shape)  # pems03-1.shape: (26208, 358)
    pems = pemss03[..., :1]
    print("pems03-2.shape:", pems.shape)  # pems03-2.shape: (26208, 358, 1)
    pemss04 = load_st_dataset('PEMS04')
    pemss04 = load_st_dataset('PEMS07')
    pemss08 = load_st_dataset('PEMS08')
