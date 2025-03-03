# %%
import csv
import torch
import random
import numpy as np
import math

# %%
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        data = []
        data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(data)

def get_gaussian(adj):
    Gaussian = np.zeros((adj.shape[0], adj.shape[0]), dtype=np.float32)
    gamaa = 1
    sumnorm = 0
    for i in range(adj.shape[0]):
        norm = np.linalg.norm(adj[i]) ** 2
        sumnorm = sumnorm + norm
    gama = gamaa / (sumnorm / adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            Gaussian[i, j] = math.exp(-gama * (np.linalg.norm(adj[i] - adj[j]) ** 2))
    return torch.Tensor(Gaussian) # @Q


def make_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

# %%
# %%
def gen_dataset(args, md_p, ix, loop_i):
    dataset = dict()
    dataset['md_p'] = torch.tensor(md_p, dtype=torch.float)  # QT
    zero_index = []
    one_index = []
    for i in range(dataset['md_p'].shape[0]):
        for j in range(dataset['md_p'].shape[1]):
            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            else:
                one_index.append([i, j])
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = torch.LongTensor(zero_index)
    one_tensor = torch.LongTensor(one_index)

    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]  # Q for luu, shuffle nhug k anh huog ndug

    ##-------------------------------------------------------
    mi_func_matrix = read_csv(args.fi_ori_feature + 'Mi_Func.csv')
    mi_fam_matrix = read_csv(args.fi_ori_feature + 'Mi_Fam.csv')
    mi_gip_matrix = get_gaussian(dataset['md_p'])

    mi_fea_temp = (mi_fam_matrix + mi_func_matrix) / 2

    MW = torch.tensor(mi_fea_temp > 0, dtype=torch.int)
    SM = MW * mi_fea_temp + (1 - MW) * mi_gip_matrix

    ##-------------------------------------------------------
    dis_sema1_matrix = read_csv(args.fi_ori_feature + 'Dis_Sema1.csv')
    dis_sema2_matrix = read_csv(args.fi_ori_feature + 'Dis_Sema2.csv')
    dis_func_matrix = read_csv(args.fi_ori_feature + 'Dis_Func.csv')
    dis_gip_matrix = get_gaussian(dataset['md_p'].T)

    dis_sema_matrix = (dis_sema1_matrix + dis_sema2_matrix) / 2
    dis_fea_temp = (dis_sema_matrix + dis_func_matrix) / 2  # ten chua doi for general

    DW = torch.tensor(dis_fea_temp > 0, dtype=torch.int)
    SD = DW * dis_fea_temp + (1 - DW) * dis_gip_matrix
    ##-------------------------------------------------------
    ## CHUA DOI TEN
    mi_func_edge_index = get_edge_index(SM)
    dataset['mm_func'] = {'data_matrix': SM, 'edges': mi_func_edge_index}  # Dung chung ten func
    dis_fea_temp_index = get_edge_index(SD)
    dataset['dd_sema'] = {'data_matrix': SD, 'edges': dis_fea_temp_index}  # Dung chung ten sema

    return dataset


# %%
# %%
def split_kfold_MCB(path_in, path_proc, file_MD_name, temp_mcb, type_test, loop_i): #Q muon gop lay DFSA CB
    from sklearn.model_selection import KFold
    from params import args

    MD = np.genfromtxt(path_in + file_MD_name, delimiter=',').astype(int)
    idx_pair1 = np.argwhere(MD == 1)
    idx_pair0 = np.argwhere(MD == 0)

    kf = KFold(args.nfold, random_state = 123, shuffle = True)
    kf_idx_pair1 = [train_test for train_test in kf.split(idx_pair1)]
    kf_idx_pair0 = [train_test for train_test in kf.split(idx_pair0)]

    idx_pair_train_set = []
    idx_pair_test_set = []
    y_train_set = []
    y_test_set = []
    train_adj_set = []

    for k in range(args.nfold): #Q bat buoc fai 5
        idx_pair_train = np.concatenate((idx_pair1[kf_idx_pair1[k][0]], idx_pair0[kf_idx_pair0[k][0]]), axis = 0)
        idx_pair_test = np.concatenate((idx_pair1[kf_idx_pair1[k][1]], idx_pair0[kf_idx_pair0[k][1]]), axis = 0)
        y_train = np.concatenate((np.ones(len(kf_idx_pair1[k][0])), np.zeros(len(kf_idx_pair0[k][0]))), axis = 0)
        y_test = np.concatenate((np.ones(len(kf_idx_pair1[k][1])), np.zeros(len(kf_idx_pair0[k][1]))), axis = 0)
        train_adj = make_adj(idx_pair1[kf_idx_pair1[k][0], :], (MD.shape[0], MD.shape[1]))

        #Q
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/train_pairs' + temp_mcb + \
        #            str(k + 1) + '.csv', idx_pair_train, delimiter=',')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/test_pairs' + type_test + \
        #            str(k + 1) + '.csv', idx_pair_test, delimiter=',')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/ytrain' + temp_mcb + str(k + 1) + '.csv', y_train, fmt = '%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/ytest' + type_test + str(k + 1) + '.csv', y_test, fmt='%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/md_p' + str(k + 1) + '.csv', train_adj, delimiter = ',', fmt='%d')

        idx_pair_train_set.append(idx_pair_train)
        idx_pair_test_set.append(idx_pair_test)
        y_train_set.append(y_train)
        y_test_set.append(y_test)
        train_adj_set.append(train_adj)
    return idx_pair_train_set, idx_pair_test_set, y_train_set, y_test_set, train_adj_set

def split_kfold_CBTr(path_in, path_proc, file_MD_name, temp_mcb, temp_test, neg_rate, loop_i):
    from sklearn.model_selection import KFold
    from params import args

    MD = np.genfromtxt(path_in + file_MD_name, delimiter=',').astype(int)
    idx_pair1 = np.argwhere(MD == 1)
    idx_pair0 = np.argwhere(MD == 0)
    np.random.shuffle(idx_pair1) #Q cuc QT
    np.random.shuffle(idx_pair0)
    kf = KFold(args.nfold, random_state = 123, shuffle = True)
    kf_idx_pair1 = [train_test for train_test in kf.split(idx_pair1)]
    kf_idx_pair0 = [train_test for train_test in kf.split(idx_pair0)]

    idx_pair_train_set = []
    idx_pair_test_set = []
    y_train_set = []
    y_test_set = []
    train_adj_set = []

    for k in range(args.nfold): #Q bat buoc fai 5
        idx_pair_train = np.concatenate((idx_pair1[kf_idx_pair1[k][0]], \
                        idx_pair0[kf_idx_pair0[k][0]][: neg_rate * len(idx_pair1[kf_idx_pair1[k][0]])]), axis = 0)
        idx_pair_test = np.concatenate((idx_pair1[kf_idx_pair1[k][1]], idx_pair0[kf_idx_pair0[k][1]]), axis = 0)
        y_train = np.concatenate((np.ones(len(kf_idx_pair1[k][0])), \
                        np.zeros(len(kf_idx_pair0[k][0]))[: neg_rate * len(kf_idx_pair1[k][0])]), axis = 0)
        y_test = np.concatenate((np.ones(len(kf_idx_pair1[k][1])), np.zeros(len(kf_idx_pair0[k][1]))), axis = 0)
        train_adj = make_adj(idx_pair1[kf_idx_pair1[k][0], :], (MD.shape[0], MD.shape[1]))

        #Q
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/train_pairs' + temp_mcb + \
        #            str(k + 1) + '.csv', idx_pair_train, delimiter=',', fmt = '%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/test_pairs' + temp_test + \
        #            str(k + 1) + '.csv', idx_pair_test, delimiter=',', fmt = '%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/ytrain' + temp_mcb + str(k + 1) + '.csv', y_train, fmt = '%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/ytest' + temp_test + str(k + 1) + '.csv', y_test, fmt='%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/md_p' + str(k + 1) + '.csv', train_adj, delimiter = ',', fmt='%d')

        idx_pair_train_set.append(idx_pair_train)
        idx_pair_test_set.append(idx_pair_test)
        y_train_set.append(y_train)
        y_test_set.append(y_test)
        train_adj_set.append(train_adj)
    return idx_pair_train_set, idx_pair_test_set, y_train_set, y_test_set, train_adj_set

def split_kfold_CBTe(path_in, path_proc, file_MD_name, temp_mcb, temp_test, neg_rate, loop_i):
    from sklearn.model_selection import KFold
    from params import args

    MD = np.genfromtxt(path_in + file_MD_name, delimiter=',').astype(int)
    idx_pair1 = np.argwhere(MD == 1)
    idx_pair0 = np.argwhere(MD == 0)
    np.random.shuffle(idx_pair1)
    np.random.shuffle(idx_pair0)
    kf = KFold(args.nfold, random_state = 123, shuffle = True)
    kf_idx_pair1 = [train_test for train_test in kf.split(idx_pair1)]
    kf_idx_pair0 = [train_test for train_test in kf.split(idx_pair0)]

    idx_pair_train_set = []
    idx_pair_test_set = []
    y_train_set = []
    y_test_set = []
    train_adj_set = []

    for k in range(args.nfold): #Q bat buoc fai 5
        idx_pair_train = np.concatenate((idx_pair1[kf_idx_pair1[k][0]], \
                        idx_pair0[kf_idx_pair0[k][0]][: neg_rate * len(idx_pair1[kf_idx_pair1[k][0]])]), axis = 0)
        idx_pair_test = np.concatenate((idx_pair1[kf_idx_pair1[k][1]], \
                        idx_pair0[kf_idx_pair0[k][1]][: neg_rate * len(idx_pair1[kf_idx_pair1[k][1]])]), axis = 0)
        y_train = np.concatenate((np.ones(len(kf_idx_pair1[k][0])), \
                        np.zeros(len(kf_idx_pair0[k][0]))[: neg_rate * len(kf_idx_pair1[k][0])]), axis = 0)
        y_test = np.concatenate((np.ones(len(kf_idx_pair1[k][1])), \
                        np.zeros(len(kf_idx_pair0[k][1]))[: neg_rate * len(kf_idx_pair1[k][1])]), axis = 0)
        train_adj = make_adj(idx_pair1[kf_idx_pair1[k][0], :], (MD.shape[0], MD.shape[1]))

        #Q
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/train_pairs' + temp_mcb + \
        #            str(k + 1) + '.csv', idx_pair_train, delimiter=',', fmt = '%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/test_pairs' + temp_test + \
        #            str(k + 1) + '.csv', idx_pair_test, delimiter=',', fmt = '%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/ytrain' + temp_mcb + str(k + 1) + '.csv', y_train, fmt = '%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/ytest' + temp_test + str(k + 1) + '.csv', y_test, fmt='%d')
        # np.savetxt(path_proc + 'L' + str(loop_i) + '/md_p' + str(k + 1) + '.csv', train_adj, delimiter = ',', fmt='%d')

        idx_pair_train_set.append(idx_pair_train)
        idx_pair_test_set.append(idx_pair_test)
        y_train_set.append(y_train)
        y_test_set.append(y_test)
        train_adj_set.append(train_adj)
    return idx_pair_train_set, idx_pair_test_set, y_train_set, y_test_set, train_adj_set

def read_train_test_adj(path, file_mdp_name, temp_mcb, type_test, ix, loop_i):
    idx_pair_train = np.genfromtxt(path + 'L' + str(loop_i) + '/train_pairs' + temp_mcb + \
               str(ix) + '.csv', delimiter=',').astype(int)
    idx_pair_test = np.genfromtxt(path + 'L' + str(loop_i) + '/test_pairs' + type_test + \
               str(ix) + '.csv', delimiter=',').astype(int)
    y_train = np.genfromtxt(path + 'L' + str(loop_i) + '/ytrain' + temp_mcb + \
                            str(ix) + '.csv').astype(int).reshape(-1)
    y_test = np.genfromtxt(path + 'L' + str(loop_i) + '/ytest' + type_test + \
                           str(ix) + '.csv').astype(int).reshape(-1)
    train_adj = np.genfromtxt(path + 'L' + str(loop_i) + file_mdp_name + str(ix) + '.csv', delimiter = ',')
    return idx_pair_train, idx_pair_test, y_train, y_test, train_adj
