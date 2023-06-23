import argparse
import csv
import pickle

import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import numpy as np


def get_adj_matrix1(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None, normalized_k=0.1):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename and id_filename!='None':
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                if type_ == 'connectivity':
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
                elif type_ == 'distance':
                    A[id_dict[i], id_dict[j]] = 1 / distance
                    A[id_dict[j], id_dict[i]] = 1 / distance
                else:
                    raise ValueError("type_ error, must be connectivity or distance!")
        return A

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be connectivity or distance!")
    return A

def get_adj_matrix2(distance_df_filename, num_of_vertices, id_filename=None, normalized_k=0.1):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    A[:] = np.inf  # 初始化无穷大（表示无限远）
    if id_filename and id_filename!='None':
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = distance
                A[id_dict[j], id_dict[i]] = distance
        distances = A[~np.isinf(A)].flatten()
        std = distances.std()
        A = np.exp(-np.square(A / std))
        A[A < normalized_k] = 0
        return A+np.identity(num_of_vertices)

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            A[i, j] = distance
            A[j, i] = distance
    distances = A[~np.isinf(A)].flatten()
    std = distances.std()
    A = np.exp(-np.square(A / std))
    A[A < normalized_k] = 0
    return A+np.identity(num_of_vertices)

def scaled_Laplacian(W):
    """
    compute \tilde{L}
    :param W: adj_mx
    :return: scaled laplacian matrix
    """
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real  # k largest real part of eigenvalues

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T{K-1}
    :param L_tilde: scaled laplacian matrix
    :param K: the maximum order of chebyshev polynomials
    :return: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * np.dot(L_tilde, cheb_polynomials[i-1]) - cheb_polynomials[i-2])

    return cheb_polynomials


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_filename', type=str, default='data/PEMS03/PEMS03.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='data/PEMS08/PEMS08.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--num_nodes', type=int, default=170, help='')
    # parser.add_argument('--adj_type', type=str, default='distance_exp',
    #                     help='type of matrix')
    # parser.add_argument('--pkl_filename', type=str, default='data/PEMS03/adj_con.pkl',
    #                     help='Path of the output file.')

    args = parser.parse_args()

    # adj_mx = get_adj_matrix1(args.distances_filename, args.num_nodes, type_=args.adj_type, id_filename=args.id_filename)
    # adj_mx = get_adj_matrix2(args.distances_filename, args.num_nodes, id_filename=args.id_filename)
    adj_mx = get_adj_matrix2(args.distances_filename, args.num_nodes)
    print(adj_mx)
    print(adj_mx.shape)
    # print(adj_mx.sum(axis=1))
    print(asym_adj(adj_mx))
    # # Save to pickle file.
    # with open(args.output_pkl_filename, 'wb') as f:
    #     pickle.dump(adj_mx, f, protocol=2)
    print(len(adj_mx[adj_mx==0.]))
    # print(np.identity(10))

