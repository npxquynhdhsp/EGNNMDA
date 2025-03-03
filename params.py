# %%
import argparse
import numpy as np

db = 'HMDD v2.0' ## HMDD v2.0/HMDD v3.2/INDE_TEST
type_eval = 'KFOLD' ## KFOLD/DIS_K/DENO_MI
type_test = '' ## '_nho'

# %%
nloop = 1 # ori 5
bgl = 1
nfold = 5 # ori 5
bgf = 1 # begin fold = 1
eofmi = 50 # ori 50
dis_set = [3, 5, 9]
read_tr_te_adj = 0 # from disk, for kfold

if db == 'INDE_TEST':
    type_eval = ''

# fi_in = './DATA/PROC/' + db + '/'         #Q '../IN/Q18/'
fi_A = './DATA/IN/' + db + '/'              #Q fi_in + type_eval + '/'
fi_ori_feature = './DATA/IN/' + db + '/'    #Q '../IN/' + temp_db + '/' + type_eval + '/'
fi_proc = './DATA/PROC/' + db + '/' + type_eval + '/'
fi_out = './OUT/' + db + '/' + type_eval + '/'

# %%
def parameter_parser():
    parser = argparse.ArgumentParser(description = "Q_EGNNMDA.")
    parser.add_argument("--db",
                       default = db,
                        help = "HMDD v2.0/HMDD v3.2/INDE_TEST.")
    parser.add_argument("--fi_A",
                        nargs = "?",
                        default = fi_A,
                        help = "adj_MD path")
    parser.add_argument("--fi_ori_feature",
                        nargs = "?",
                        default = fi_ori_feature,
                        help = "original feature path.")
    parser.add_argument("--fi_proc",
                        nargs = "?",
                        default = fi_proc,
                        help = "processing path.")
    parser.add_argument("--fi_out",
                        nargs = "?",
                        default=fi_out,
                        help = "out path.")
    parser.add_argument("--type_eval",
                        nargs = "?",
                        default = type_eval,
                        help = "KFOLD/DIS_K/DENO_MI.")
    parser.add_argument("--nloop",
                        type = int,
                        default = nloop)
    parser.add_argument("--bgl",
                        type = int,
                        default = bgl)
    parser.add_argument("--nfold",
                        type = int,
                        default = nfold,
                        help = "n cross-validation.")
    parser.add_argument("--bgf",
                        type = int,
                        default = bgf,
                        help = "begin number of fold.") #Q = 1
    parser.add_argument("--eofmi",
                        type = int,
                        default = eofmi,
                        help = "number of denovo miRNAs.")
    parser.add_argument("--dis_set",
                        default = dis_set,
                        help = "[3, 5, 9].")
    parser.add_argument("--read_tr_te_adj",
                        default = read_tr_te_adj,
                        help = "[1/0].")
    parser.add_argument("--type_test",
                        default = type_test,
                        help = " `` or `_nho`.")
    ##############################
    #--Phase 1 of Gen_fea
    parser.add_argument("--em_mi",
                        type = int,
                        default = 256)
    parser.add_argument("--em_dis",
                        type = int,
                        default = 256)
    parser.add_argument("--out_mi_dim",
                        type = int,
                        default = 256)
    parser.add_argument("--out_dis_dim",
                        type = int,
                        default = 256)
    parser.add_argument("--ne_feature",
                        type = int,
                        default = 100,
                        help = "100.")
    # --Phase 2 of Gen_fea
    parser.add_argument("--rf_ne",
                        type = int,
                        default = 100,
                        help="100.")
    # parser.add_argument("--etr_ne",
    #                     type = int,
    #                     default = 100,
    #                     help="100.")
    parser.add_argument("--xg_ne",
                        type = int,
                        default = 500,
                        help="500.")
    parser.add_argument("--xg_lrr",
                        default = 0.2)
    ##############################
    return parser.parse_args()

args = parameter_parser()
if args.db == 'HMDD v2.0':
    args.mi_num, args.dis_num = 495, 383
else:
    args.mi_num, args.dis_num = 788, 374

