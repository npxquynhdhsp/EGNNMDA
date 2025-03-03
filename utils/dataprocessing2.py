# %%
import numpy as np
import pandas as pd
import random

# %%
def split_tr_te_adj(type_eval, path_in, path_proc, file_MD_name, temp_mcb, type_test, neg_rate, ix, loop_i):
    from params import args
    from utils.dataprocessing import make_adj

    if temp_mcb == '_CB':
        print('CB, DOC CMT DATAPROC!!!')

    trainLabel = []
    testLabel = []
    train_pairs = list()
    train_pairs0 = list()
    test_pairs = list()

    count_pos_train = 0

    MD = np.genfromtxt(path_in + file_MD_name, delimiter=',').astype(int)
    
    if type_eval == 'DIS_K':
        MD[:, ix - 1] = MD[:, ix - 1] + 3
    elif type_eval == 'DENO_MI':
        MD[ix - 1, :] = MD[ix - 1, :] + 3

    if temp_mcb == '_CB':
        negative_pool_train = list()
        for rowIndex in range(args.mi_num):
            for colIndex in range(args.dis_num):
                if MD[rowIndex][colIndex] == 0:
                    negative_pool_train.append([rowIndex, colIndex])
                else:
                    if MD[rowIndex][colIndex] == 1:
                        count_pos_train += 1
        print(count_pos_train)
        negative_indexes_train = list(range(len(negative_pool_train)))
        random.seed(loop_i) #Q de kq khac each lan but hih nhu qua may dif lai same nhau, HOAC DONG LAI
        random.shuffle(negative_indexes_train)
        train_pairs0.append([[negative_pool_train[pind][0], negative_pool_train[pind][1]] for pind in
                               negative_indexes_train[:neg_rate * count_pos_train]])

    for rowIndex in range(args.mi_num):
        for colIndex in range(args.dis_num):
            if  temp_mcb == '_MCB':  #Q MCB, train_label lộn xộn
                if MD[rowIndex][colIndex] <= 1:
                    trainLabel.append(MD[rowIndex][colIndex])
                    train_pairs.append([rowIndex, colIndex])
            elif MD[rowIndex][colIndex] == 1:  #Q CB, train1
                trainLabel.append(MD[rowIndex][colIndex])
                train_pairs.append([rowIndex, colIndex])

            if (MD[rowIndex][colIndex] == 3) or (MD[rowIndex][colIndex] == 4):
                #Q test, trong cả CB và MCB, =4, =3. Viet dai vi dung cho ca y_5loai
                if MD[rowIndex][colIndex] == 4:
                    testLabel.append(1)
                else:
                    if MD[rowIndex][colIndex] == 3:
                        testLabel.append(0)
                test_pairs.append([rowIndex, colIndex])

    if  temp_mcb == '_CB':  #Q CB, train_label pos -> neg
        trainLabel = trainLabel + [0] * neg_rate * count_pos_train

        tem = np.array(train_pairs0)
        tem = tem.reshape(tem.shape[1], tem.shape[2])  #Q 3 dim
        train_pairs = np.vstack((np.array(train_pairs), tem))
        # print(train_pairs.shape)

    train_adj = make_adj(np.argwhere(MD == 1), (MD.shape[0], MD.shape[1]))

    # Q
    print('Ko luu proc')
    # np.savetxt(path_proc + 'L' + str(loop_i) + '/train_pairs' + temp_mcb + \
    #            str(ix) + '.csv', train_pairs, delimiter=',')
    # np.savetxt(path_proc + 'L' + str(loop_i) + '/test_pairs' + \
    #            str(ix) + type_test + '.csv', test_pairs, delimiter=',')
    # np.savetxt(path_proc + 'L' + str(loop_i) + '/ytrain' + temp_mcb + str(ix) + '.csv', trainLabel, fmt = '%d')
    # np.savetxt(path_proc + 'L' + str(loop_i) + '/ytest' + str(ix) + type_test + '.csv', testLabel, fmt='%d')
    # np.savetxt(path_proc + 'L' + str(loop_i) + '/md_p' + str(ix) + '.csv', train_adj, delimiter = ',', fmt='%d')

    return np.array(train_pairs), np.array(test_pairs), np.array(trainLabel), np.array(testLabel), train_adj



