# %%
from params import args
from gen_feature import gen_feature
from utils.dataprocessing import *
from utils.dataprocessing2 import *
from utils.utils import *
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import numpy as np
import time

# %%
def models_eval(method_set_name, X_train_enc, X_test_enc, y_train, y_test, ix, loop_i, model_i):
    if method_set_name == 'RF':
        print('Random Forest')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=args.rf_ne, max_depth=None, n_jobs=-1)
    elif method_set_name == 'ETR':
        print('Extra trees regression')
        from sklearn.ensemble import ExtraTreesRegressor
        clf = ExtraTreesRegressor(n_estimators=args.etr_ne, n_jobs=-1)
    elif method_set_name == 'LR':
        print('Linear regression')
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()
    else:
        print('XGBoost')
        from xgboost import XGBClassifier
        clf = XGBClassifier(booster='gbtree', n_jobs=2, learning_rate=args.xg_lrr, n_estimators=args.xg_ne)

    clf.fit(X_train_enc, y_train)

    if (method_set_name == 'ETR') or (method_set_name == 'LR'):
        y_prob = clf.predict(X_test_enc)
    else:
        y_prob = clf.predict_proba(X_test_enc)[:,1]

    np.savetxt(args.fi_out + 'L' + str(loop_i) + '_M' + str(model_i) + '_yprob_' + method_set_name.lower() + str(ix) + '.csv', y_prob)
    calculate_score([y_test], [y_prob])
    return y_prob


def save_eval(method_set_name, true_set, prob_set):
    print(method_set_name,':')
    calculate_score(true_set, prob_set) # cal mean
    prob_set_join = np.concatenate(prob_set, axis = 0) # join
    np.savetxt(args.fi_out + 'prob_set_' + method_set_name.lower() + '.csv', prob_set_join)

    if method_set_name == 'XG':
        true_set_join = np.concatenate(true_set, axis = 0)
        np.savetxt(args.fi_out + 'true_set.csv', true_set_join, fmt='%d')
    return

def combine2(mi_em1, dis_em1, mi_em2, dis_em2, idx_pair_train, idx_pair_test, loop_i, ix):
    from sklearn.preprocessing import MinMaxScaler
    def cosSim2(interaction1, interaction2):
        rows, columns = interaction1.shape
        sim_matrix = np.zeros(rows)

        # Calculate cosine similarity for each row
        for i in range(rows):
            vec1 = interaction1[i, :]
            vec2 = interaction2[i, :]
            norm_row = np.linalg.norm(vec1)
            norm_col = np.linalg.norm(vec2)

            if norm_row == 0 or norm_col == 0:
                sim_matrix[i] = 0
            else:
                sim_matrix[i] = np.dot(vec1, vec2) / (norm_row * norm_col)

        return sim_matrix

    scaler = MinMaxScaler()
    mi_em1T = scaler.fit_transform(mi_em1)
    mi_em2T = scaler.fit_transform(mi_em2)
    dis_em1T = scaler.fit_transform(dis_em1)
    dis_em2T = scaler.fit_transform(dis_em2)

    # Combine
    mi_em1 = mi_em1 * cosSim2(mi_em1T, mi_em2T)[:, np.newaxis] + mi_em2 * (1-cosSim2(mi_em1T, mi_em2T)[:, np.newaxis])
    mi_em2 = mi_em2 * cosSim2(mi_em1T, mi_em2T)[:, np.newaxis] + mi_em1 * (1-cosSim2(mi_em1T, mi_em2T)[:, np.newaxis])
    dis_em1 = dis_em1 * cosSim2(dis_em1T, dis_em2T)[:, np.newaxis] + dis_em2 * (1-cosSim2(dis_em1T, dis_em2T)[:, np.newaxis])
    dis_em2 = dis_em2 * cosSim2(dis_em1T, dis_em2T)[:, np.newaxis] + dis_em1 * (1-cosSim2(dis_em1T, dis_em2T)[:, np.newaxis])

    X_train1T = np.hstack((mi_em1[idx_pair_train[:, 0]].tolist(), dis_em1[idx_pair_train[:, 1]].tolist()))
    X_test1 = np.hstack((mi_em1[idx_pair_test[:, 0]].tolist(), dis_em1[idx_pair_test[:, 1]].tolist()))
    X_train2T = np.hstack((mi_em2[idx_pair_train[:, 0]].tolist(), dis_em2[idx_pair_train[:, 1]].tolist()))
    X_test2 = np.hstack((mi_em2[idx_pair_test[:, 0]].tolist(), dis_em2[idx_pair_test[:, 1]].tolist()))

    return X_train1T, X_test1, X_train2T, X_test2

def balance_data(X, y, neg_rate):
    if (neg_rate == -1) or ((args.db == 'HMDD v3.2') and (args.type_eval == 'DIS_K')):
        print(X.shape)
        return X, y
    else:
        x_pos = X[y == 1]
        x_neg = X[y == 0]

        npos = x_pos.shape[0]

        x_neg_new = shuffle(x_neg, random_state=2022)
        x_neg_new = x_neg_new[:npos * neg_rate]

        X_new = np.vstack([x_pos, x_neg_new])
        y_new = np.hstack([np.ones(npos), np.zeros(npos * neg_rate)])

        X_balanced, y_balanced = shuffle(X_new, y_new, random_state=2022)

        return X_balanced, y_balanced

def main():
    if args.db != 'INDE_TEST':
        print(args.db)
        if args.type_eval == 'KFOLD':  # KFOLD/DIS_K/DENO_MI
            set_ix = np.arange(args.bgf, args.nfold + 1) #Q khac cac bai khac
            temp = 'FOLD '
            print('read_tr_te_adj', args.read_tr_te_adj)
        elif args.type_eval == 'DIS_K':
            set_ix = args.dis_set
            temp = 'DIS '
        else:
            mi_set = np.genfromtxt(args.fi_A + 'mi_setT.csv').astype(int).T #Q
            set_ix = mi_set
            temp = 'MIRNA '
    else:
        print('INDEPENDENT TEST')
        set_ix = [1]

    prob_set1, prob_set2, prob_set = [], [], []
    true_set = []

    for loop_i in range(args.bgl, args.nloop + 1):
        print('.......................................... LOOP ', loop_i,'.........................................')
        if (args.db != 'INDE_TEST') and (args.type_eval == 'KFOLD'):
            idx_pair_train_set, idx_pair_test_set, y_trainT_set, y_test_set, train_adj_set = \
                split_kfold_MCB(args.fi_A, args.fi_proc, 'adj_MD.csv', \
                             '_MCB', args.type_test, loop_i)
            print('.')
        for ix in set_ix:
            ###-----------------------
            if (args.db == 'INDE_TEST'):
                idx_pair_train, idx_pair_test, y_trainT, y_test, train_adj = \
                    split_tr_te_adj(args.type_eval, args.fi_A, args.fi_proc, 'adj_4loai.csv', \
                                    '_MCB', args.type_test, -1, ix, loop_i)
            else:
                if (args.type_eval == 'KFOLD'):
                    if args.read_tr_te_adj == 1:
                        idx_pair_train, idx_pair_test, y_trainT, y_test, train_adj = \
                            read_train_test_adj(args.fi_proc, '/md_p', \
                                                '_MCB', args.type_test, ix, loop_i)
                    else:
                        idx_pair_train, idx_pair_test, y_trainT, y_test, train_adj = \
                            idx_pair_train_set[ix-1], idx_pair_test_set[ix-1], y_trainT_set[ix-1], \
                                y_test_set[ix-1], train_adj_set[ix-1]
                else:
                    idx_pair_train, idx_pair_test, y_trainT, y_test, train_adj = \
                        split_tr_te_adj(args.type_eval, args.fi_A, args.fi_proc, 'adj_MD.csv', \
                                        '_MCB', args.type_test, -1, ix, loop_i)

            ###-----------------------
            if args.db != 'INDE_TEST':
                print(temp, ix, '*' * 50)

            print('GEN FEATURE MODEL 1')
            mi_em1, dis_em1 = gen_feature(idx_pair_train, idx_pair_test, \
                                train_adj, ix, loop_i, 1)
            print('GEN FEATURE MODEL 2')
            mi_em2, dis_em2 = gen_feature(idx_pair_train, idx_pair_test, \
                                train_adj, ix, loop_i, 2)

            #------------------------------------------------------
            method_set = ['XG']

            print('Combine cosi')
            X_train1T, X_test1, X_train2T, X_test2 = combine2(mi_em1, dis_em1, mi_em2, dis_em2, idx_pair_train, idx_pair_test, loop_i, ix)

            #--NEGATIVE SAMPLES SELECTION
            print('Negative samples selection')
            from xgboost import XGBClassifier
            from sklearn.model_selection import KFold
            def nega_sample_selection(XtrainT, ytrainT):
                kf = KFold(n_splits = 3, shuffle = True, random_state = 2022)
                X_positive = XtrainT[ytrainT == 1]  # Mẫu dương
                y_positive = np.ones(X_positive.shape[0])
                X_unlabeled = XtrainT[ytrainT == 0]  # Mẫu âm
                y_unlabeled = np.zeros(X_unlabeled.shape[0])
                U_splits = list(kf.split(X_unlabeled))

                Upredictions = np.zeros_like(y_unlabeled, dtype=float)

                for i, (test_idx, train_idx) in enumerate(U_splits):  # chu y co doi vi tri train test cho dung y
                    # Train from P + U_i
                    UX_train = np.vstack([X_positive, X_unlabeled[train_idx]])
                    Uy_train = np.hstack([y_positive, y_unlabeled[train_idx]])

                    UX_test = X_unlabeled[test_idx]

                    Umodel = XGBClassifier(booster='gbtree', n_jobs=2, learning_rate=args.xg_lrr, n_estimators=args.xg_ne)

                    if i == 1:
                        from sklearn.linear_model import LogisticRegression
                        Umodel = LogisticRegression()
                    elif i == 2:
                        from sklearn.ensemble import RandomForestClassifier
                        Umodel = RandomForestClassifier(n_estimators=args.rf_ne, max_depth=None, n_jobs=-1)  # tam

                    Umodel.fit(UX_train, Uy_train)

                    Uy_prob = Umodel.predict_proba(UX_test)[:, 1]
                    Upredictions[test_idx] += Uy_prob  # Cộng dồn kết quả dự đoán

                Upredictions /= 2

                # All neg
                negative_indices = np.argwhere(np.array(Upredictions) < 0.5).reshape(-1)

                # Select neg
                selected_indices = np.random.choice(negative_indices, size=y_positive.shape[0], replace=False)
                X_final_neg = X_unlabeled[selected_indices]

                Xtrain_AN_2x4344 = np.vstack([X_positive, X_final_neg])
                ytrain_AN_2x4344 = np.hstack(
                    [np.ones(y_positive.shape[0]), np.zeros(y_positive.shape[0])])
                return Xtrain_AN_2x4344, ytrain_AN_2x4344, selected_indices
            #-----------------------------------------------------

            #--Balance test
            X_test1C, y_testC = balance_data(X_test1, y_test, 1)
            X_test2C, y_testC = balance_data(X_test2, y_test, 1)
            true_set.append(y_testC)
            #--

            #--EVALUATION
            print('EVALUATION:')
            print('MODEL 1')
            ##--random
            X_train1, y_train = balance_data(X_train1T, y_trainT, 1)
            y_prob1 = models_eval(method_set[0] + '_RandomMCB', X_train1, X_test1C, y_train, y_testC, ix, loop_i, 1)
            X_train2, y_trainR = balance_data(X_train2T, y_trainT, 1)
            y_prob2 = models_eval(method_set[0] + '_RandomMCB', X_train2, X_test2C, y_train, y_testC, ix, loop_i, 2)

            #--Reliable negative selection
            # X_train1, y_train, selected_indices1 = nega_sample_selection(X_train1T, y_trainT)
            # y_prob1 = models_eval(method_set[0], X_train1, X_test1C, y_train, y_testC, ix, loop_i, 1)

            # print('MODEL 2')
            # X_train2, y_train, selected_indices2 = nega_sample_selection(X_train2T, y_trainT)
            # y_prob2 = models_eval(method_set[0], X_train2, X_test2C, y_train, y_testC, ix, loop_i, 2)

            print('BOTH:')
            log_loss_1 = log_loss(y_testC, y_prob1)
            log_loss_2 = log_loss(y_testC, y_prob2)

            # Tính xác suất hậu nghiệm (dựa trên log-loss)
            exp_loss_1 = np.exp(-log_loss_1)
            exp_loss_2 = np.exp(-log_loss_2)

            w_1 = exp_loss_1 / (exp_loss_1 + exp_loss_2)
            w_2 = exp_loss_2 / (exp_loss_1 + exp_loss_2)

            y_prob = w_1 * y_prob1 + w_2 * y_prob2

            np.savetxt(
                args.fi_out + 'L' + str(loop_i) + '_yprob_' + method_set[0].lower() + str(
                    ix) + '.csv', y_prob)
            np.savetxt(
                args.fi_out + 'L' + str(loop_i) + '_ytrue' + str(
                    ix) + '.csv', y_testC, fmt = '%d')
            calculate_score([y_testC], [y_prob])

            prob_set1.append(y_prob1)
            prob_set2.append(y_prob2)
            prob_set.append(y_prob)

    print('--------------------------------FINAL MEAN ALL:-------------------------------')
    save_eval(method_set[0], true_set, prob_set)

# %%
if __name__ == "__main__":
    print('fi_ori_feature', args.fi_ori_feature)
    main()



