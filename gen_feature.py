# %%
from torch import nn
from torch_geometric.nn import GCNConv, conv
from params import args
from utils.dataprocessing import gen_dataset, make_adj
import torch.nn.functional as F
import torch
import numpy as np

# %%
torch.backends.cudnn.enabled = False

# %%
class model_feature1(nn.Module):
    def __init__(self, args):
        super(model_feature1, self).__init__()
        self.kernel_len = 3

        self.conv1_mi_func = GCNConv(args.em_mi, args.em_mi)
        self.conv2_mi_func = GCNConv(args.em_mi, args.em_mi)
        self.conv3_mi_func = conv.GATConv(args.em_mi, args.out_mi_dim)

        # -----------------------------------------------------------------
        self.conv1_dis_sem = GCNConv(args.em_dis, args.em_dis)
        self.conv2_dis_sem = GCNConv(args.em_dis, args.em_dis)
        self.conv3_dis_sem = conv.GATConv(args.em_dis, args.out_dis_dim)

        self.mirna_mmf = []
        self.disease_dds = []

    def forward(self, data):
        torch.manual_seed(123)

        mirna_kernels_mmf = []
        disease_kernels_dds = []

        mm_f0 = torch.randn((args.mi_num, args.em_mi))
        dd_s0 = torch.randn((args.dis_num, args.em_dis))

        mm_f1 = F.elu(self.conv1_mi_func(mm_f0, data['mm_func']['edges'], data['mm_func']['data_matrix'][
            data['mm_func']['edges'][0], data['mm_func']['edges'][1]]))
        mirna_kernels_mmf.append(mm_f1)
        mm_f2 = F.elu(self.conv2_mi_func(mm_f1, data['mm_func']['edges']))
        mirna_kernels_mmf.append(mm_f2)
        mmf = F.elu(self.conv3_mi_func(mm_f2, data['mm_func']['edges']))
        mirna_kernels_mmf.append(mmf)

        dd_s1 = F.elu(self.conv1_mi_func(dd_s0, data['dd_sema']['edges'], data['dd_sema']['data_matrix'][
            data['dd_sema']['edges'][0], data['dd_sema']['edges'][1]]))
        disease_kernels_dds.append(dd_s1)
        dd_s2 = F.elu(self.conv2_dis_sem(dd_s1, data['dd_sema']['edges']))
        disease_kernels_dds.append(dd_s2)
        dds = F.elu(self.conv3_dis_sem(dd_s2, data['dd_sema']['edges']))
        disease_kernels_dds.append(dds)

        mi_fea = sum([mirna_kernels_mmf[i] for i in range(self.kernel_len)]) / 3
        dis_fea = sum([disease_kernels_dds[i] for i in range(self.kernel_len)]) / 3

        return mi_fea.mm(dis_fea.t()), mi_fea, dis_fea


class model_feature2(nn.Module):
    def __init__(self, args):
        super(model_feature2, self).__init__()
        self.kernel_len = 3

        self.conv1_mi_func = GCNConv(args.em_mi, args.em_mi)
        self.conv2_mi_func = conv.GATConv(args.em_mi, args.em_mi)
        self.conv3_mi_func = conv.GATConv(args.em_mi, args.out_mi_dim)

        # -----------------------------------------------------------------
        self.conv1_dis_sem = GCNConv(args.em_dis, args.em_dis)
        self.conv2_dis_sem = conv.GATConv(args.em_dis, args.em_dis)
        self.conv3_dis_sem = conv.GATConv(args.em_dis, args.out_dis_dim)

        self.mirna_mmf = []
        self.disease_dds = []

    def forward(self, data):
        torch.manual_seed(123)

        mirna_kernels_mmf = []
        disease_kernels_dds = []

        mm_f0 = torch.randn((args.mi_num, args.em_mi))
        dd_s0 = torch.randn((args.dis_num, args.em_dis))

        mm_f1 = F.elu(self.conv1_mi_func(mm_f0, data['mm_func']['edges'], data['mm_func']['data_matrix'][
            data['mm_func']['edges'][0], data['mm_func']['edges'][1]]))
        mirna_kernels_mmf.append(mm_f1)
        mm_f2 = F.elu(self.conv2_mi_func(mm_f1, data['mm_func']['edges']))
        mirna_kernels_mmf.append(mm_f2)
        mmf = F.elu(self.conv3_mi_func(mm_f2, data['mm_func']['edges']))
        mirna_kernels_mmf.append(mmf)

        dd_s1 = F.elu(self.conv1_mi_func(dd_s0, data['dd_sema']['edges'], data['dd_sema']['data_matrix'][
            data['dd_sema']['edges'][0], data['dd_sema']['edges'][1]]))
        disease_kernels_dds.append(dd_s1)
        dd_s2 = F.elu(self.conv2_dis_sem(dd_s1, data['dd_sema']['edges']))
        disease_kernels_dds.append(dd_s2)
        dds = F.elu(self.conv3_dis_sem(dd_s2, data['dd_sema']['edges']))
        disease_kernels_dds.append(dds)

        mi_fea = sum([mirna_kernels_mmf[i] for i in range(self.kernel_len)]) / 3
        dis_fea = sum([disease_kernels_dds[i] for i in range(self.kernel_len)]) / 3

        return mi_fea.mm(dis_fea.t()), mi_fea, dis_fea


# %%
def train(model, train_data, optimizer, opt):
    model.train()
    for epoch in range(0, opt.ne_feature):
        model.zero_grad()
        score, mi_fea, dis_fea = model(train_data)
        print('epoch ', epoch)
        loss = torch.nn.MSELoss(reduction='mean')
        loss = loss(score, train_data['md_p'])
        loss.backward()
        optimizer.step()
        # print(loss.item())
    score = score.detach().cpu().numpy()
    scoremin, scoremax = score.min(), score.max()
    score = (score - scoremin) / (scoremax - scoremin)
    return score, mi_fea, dis_fea


# %%
def gen_feature(train_pair_list, test_pair_list, train_adj, ix, loop_i, model_i):
    dataset = gen_dataset(args, train_adj, ix, loop_i)
    train_data = dataset

    ###---or (1,2)------------
    if model_i == 1:
        model = model_feature1(args)
    else:
        model = model_feature2(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    score, mi_em, dis_em = train(model, train_data, optimizer, args)

    return mi_em.detach().numpy(), dis_em.detach().numpy()
