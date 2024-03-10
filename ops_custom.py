
import torch
import torch.nn as nn
import numpy as np
import torch_geometric

# import sys

# orig_stdout = sys.stdout
# f = open('cora_results.txt', 'w')
# sys.stdout = f

# from colorama import Fore, Back, Style

from run_GNN import Print

from GNN_early import GNNEarly

import warnings
warnings.filterwarnings("ignore")
root = './data/CORA'

# The Answer to the Ultimate Question of Life, the Universe, and Everything is 42
seed = 42
torch.manual_seed(seed)
import random 
random.seed(seed)
import numpy
numpy.random.seed(seed)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--segment')
args = parser.parse_args() 

# Print([0, :].sum())


class GraphUnet(nn.Module):

    def __init__(self, ks, drop_p, dataset, sml_dict):
        num_features = dataset.data.num_features
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.dataset = dataset
        self.sml_dict = sml_dict
        self.bottom_gcn = GNNEarly(opt=sml_dict, dataset=dataset, num_features=self.sml_dict['hidden_dim'])

        # print("debug dataset.x.shape = ", dataset.x.shape)
        dim = dataset.x.shape[-1]

        self.dropout = torch.nn.Dropout(p=0.1)
        

        # self.bottom_gcn = GCN(dim, dim, act, drop_p)
        
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        dim = self.sml_dict['hidden_dim']
        lst = [dataset.x.shape[-1], dim, dim]
        up_lst = [dim, dim, dim]
        self.clf = torch.nn.Linear(lst[-1], 7)
        for i in range(self.l_n):
            gcn = GNNEarly(opt=sml_dict, dataset=self.dataset, num_features=lst[i])
            # self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.down_gcns.append(gcn)
            gcn = GNNEarly(opt=sml_dict, dataset=self.dataset, num_features=up_lst[-i - 1])
            # self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(gcn)
            
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))
        # Print(self.unpools)
        # Print(self.pools)

    def to_adj(self, edge_index):
        
        adj = torch_geometric.utils.to_dense_adj(edge_index)
        out = torch.squeeze(adj, 0)
        
        
        return out
    
    def to_edge_index(self, adj, device='cuda'):
        
        
        edge_index = torch_geometric.utils.dense_to_sparse(adj)[0]
        # Print(torch_geometric.utils.to_edge_index(edge_index))
        
        
        return edge_index.to(device)

    def forward(self, g, h):
        # h is dataset.x[:batch_size, :]
        # g is edge_index
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h


        curr_edge_index = g
        for i in range(self.l_n):
            # if i == 0:
            #     edge_index = self.to_edge_index(g)
            import time
            b = time.time()
            h = self.down_gcns[i](h, new_edge_index=curr_edge_index, return_hidden=True)
            #Print(time.time() - b, 'Down GCN')
            # adj = self.to_adj(g)
            
            b = time.time()
            adj = self.to_adj(curr_edge_index)  
            
            
            # Print(adj)
            
                   
            adj_ms.append(adj)
            down_outs.append(h)
            
            adj, h, idx = self.pools[i](adj, h)
            
            # Print(adj)
            curr_edge_index = self.to_edge_index(adj) # previous
            indices_list.append(idx)
            #Print(time.time() - b, 'Pooling')
        
        
        h = self.bottom_gcn(h, new_edge_index=curr_edge_index, return_hidden=True)
        
        
        for i in range(self.l_n ):
            
            up_idx = self.l_n - i - 1
            adj, idx = adj_ms[up_idx], indices_list[up_idx]
            # g = adj_ms[up_idx]
            b = time.time()
            adj, h = self.unpools[i](adj, h, down_outs[up_idx], idx)
            
            curr_edge_index = self.to_edge_index(adj) # previous
            #Print(time.time() - b, 'Unpooling')
            b = time.time()
            h = self.up_gcns[i](h, curr_edge_index, return_hidden=True)

            
            h = h.add(down_outs[up_idx])
            #Print(time.time() - b, 'Up GCN and add')
            hs.append(h)
        
        # return hs
        h = self.clf(hs[-1])
        h = self.dropout(h)
        return torch.nn.LogSoftmax()(h)


# def custom_to_edge_index(idx, curr_edge_index):
#     new_edge_index_src = []
#     new_edge_index_dest = []
#     for i in range(len(curr_edge_index[0])):
#         src = curr_edge_index[0][i]
#         dest = curr_edge_index[1][i]
#         if src in idx and dest in idx:
#             new_edge_index_src.append(int(src))
#             new_edge_index_dest.append(int(dest))
#     return torch.tensor((new_edge_index_src, new_edge_index_dest))

class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        print(h.shape, g.shape, self.proj.weight.shape)
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        weights = self.proj(h).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h





def top_k_graph(scores, g, h, k, rw='self-loop', device='cuda'):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]

    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    if rw == 'self-loop':
        un_g += torch.diag(torch.ones(un_g.shape[0]).to(device))
    elif rw == 'a2':
        un_g = torch.matmul(un_g, un_g).bool().float()
    # Print(un_g.shape)
    
    # Print(un_g)
    # Print('--------------')
    # Print(to_edge_index(un_g).shape)
    un_g = un_g[idx, :]
    
    # Print(to_edge_index(un_g).shape)
    un_g = un_g[:, idx]
    
    # Print(to_edge_index(un_g).shape)
    # g = norm_g(un_g)

    return un_g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)


def train(model, edge_index, x, supervision,  y, lf, optmz, device='cuda'):

    model.train()
    optmz.zero_grad()
    
    out = model(edge_index.to(device), x.to(device))[supervision]
    loss = lf(out, y)
    loss.backward()
    optmz.step()

    for grand in model.down_gcns + model.up_gcns + [model.bottom_gcn]:
        grand.fm.update(grand.getNFE())
        grand.resetNFE()
        grand.bm.update(grand.getNFE())
        grand.resetNFE()

    return loss.item()

@torch.no_grad()
def eval(model, edge_index, x, supervision,  y):
    model.eval()
    out = model(edge_index.to(device), x.to(device))[supervision]

    return (out.argmax(-1) == y).float().mean()


def train_pipeline(fold, model, train_mask, dev_mask, test_mask, dataset, lf, optmz, batch_size, num_epochs):
    # X = dataset.x[rand_index, :]
    y = dataset.y.to('cuda')

    # train_x_index = rand_index[:train_size]
    train_y = y[train_mask]

    # dev_x_index = rand_index[train_size: train_size + dev_size]
    dev_y = y[dev_mask]

    # test_x_index = rand_index[train_size + dev_size:]
    test_y = y[test_mask]

    dev_scores = []
    test_scores = []
    index = torch.where(train_mask)[0]
    dev_index = torch.where(dev_mask)[0]
    test_index = torch.where(test_mask)[0]
    
    
    # lf = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_iter = 0
        
        for index_batch, y_batch in zip(torch.split(index, batch_size), torch.split(train_y, batch_size)):
            epoch_loss += train(model, dataset.edge_index, dataset.x, index_batch, y_batch, lf, optmz)
            num_iter += 1
        dev_score = eval(model, dataset.edge_index, dataset.x, dev_index, dev_y)
        
        test_score = eval(model, dataset.edge_index, dataset.x, test_index, test_y)
        dev_scores.append(dev_score)
        test_scores.append(test_score)

        m = max(dev_scores)
        if m == dev_score:
            from colorama import Fore, Back, Style
            style = Fore.YELLOW
            reporting_test = test_score
        else:
            style = Fore.BLUE

        Print(f'Fold {fold + 1}')
        Print(f'Epoch {epoch + 1}')
        Print(f'Loss: {epoch_loss / num_iter:.4f}')
        
        Print(f'Dev Accuracy: {style}{ dev_score * 100 :.3f}%')
        Style.RESET_ALL
    
        Print(f'Test Accuracy: {test_score * 100 :.3f}%, Reporting Test Score:{Fore.MAGENTA}{reporting_test * 100 :.3f}%')
        Style.RESET_ALL
        Print('======================&&&&&&&&&=======================')

    return dev_scores, test_scores, m, reporting_test


if __name__ == '__main__':
    



    sml_dict = {'use_cora_defaults': False,
             'dataset': 'Cora',
               'data_norm': 'rw',
                 'self_loop_weight': 1.0,
                   'use_labels': False,
                     'geom_gcn_splits': False,
                       'num_splits': 2,
                         'label_rate': 0.5,
                           'planetoid_split': False,
                             'hidden_dim': 80,
                               'fc_out': False, 
                               'input_dropout': 0.5, 
                               'dropout': 0.046878964627763316, 
                               'batch_norm': False, 
                               'optimizer': 'adamax', 
                               'lr': 0.022924849756740397, 
                               'decay': 0.00507685443154266, 
                               'epoch': 100, 
                               'alpha': 1.0, 
                               'alpha_dim': 'sc', 
                               'no_alpha_sigmoid': False, 
                               'beta_dim': 'sc', 
                               'block': 'constant', 
                               'function': 'laplacian', 
                               'use_mlp': False, 
                               'add_source': True, 
                               'cgnn': False, 
                               'time': 18.294754260552843, 
                               'augment': False, 
                               'method': 'euler', 
                               'step_size': 1, 
                               'max_iters': 100, 
                               'adjoint_method': 'adaptive_heun', 
                               'adjoint': False, 
                               'adjoint_step_size': 1, 
                               'tol_scale': 821.9773048827274, 
                               'tol_scale_adjoint': 1.0, 
                               'ode_blocks': 1, 
                               'max_nfe': 2000, 
                               'no_early': False, 
                               'earlystopxT': 3, 
                               'max_test_steps': 100, 
                               'leaky_relu_slope': 0.2, 
                               'attention_dropout': 0.0, 
                               'heads': 8, 
                               'attention_norm_idx': 1, 
                               'attention_dim': 128, 
                               'mix_features': False, 
                               'reweight_attention': False, 
                               'attention_type': 'scaled_dot', 
                               'square_plus': True, 
                               'jacobian_norm2': None, 
                               'total_deriv': None, 
                               'kinetic_energy': None, 
                               'directional_penalty': None, 
                               'not_lcc': True, 
                               'rewiring': None, 
                               'gdc_method': 'ppr', 
                               'gdc_sparsification': 'topk', 
                               'gdc_k': 64, 
                               'gdc_threshold': 0.01, 
                               'gdc_avg_degree': 64, 
                               'ppr_alpha': 0.05, 
                               'heat_time': 3.0, 
                               'att_samp_pct': 1, 
                               'use_flux': False, 
                               'exact': True, 
                               'M_nodes': 64, 
                               'new_edges': 'random', 
                               'sparsify': 'S_hat', 
                               'threshold_type': 'addD_rvR', 
                               'rw_addD': 0.02, 
                               'rw_rmvR': 0.02, 
                               'rewire_KNN': False, 
                               'rewire_KNN_T': 'T0', 
                               'rewire_KNN_epoch': 10, 
                               'rewire_KNN_k': 64, 
                               'rewire_KNN_sym': False, 
                               'KNN_online': False, 
                               'KNN_online_reps': 4, 
                               'KNN_space': 'pos_distance', 
                               'beltrami': False, 
                               'fa_layer': False, 
                               'pos_enc_type': 'GDC', 
                               'pos_enc_orientation': 'row', 
                               'feat_hidden_dim': 64, 
                               'pos_enc_hidden_dim': 16, 
                               'edge_sampling': False, 
                               'edge_sampling_T': 'T0', 
                               'edge_sampling_epoch': 5, 
                               'edge_sampling_add': 0.64, 
                               'edge_sampling_add_type': 'importance', 
                               'edge_sampling_rmv': 0.32, 
                               'edge_sampling_sym': False, 
                               'edge_sampling_online': False, 
                               'edge_sampling_online_reps': 4, 
                               'edge_sampling_space': 'attention', 
                               'symmetric_attention': False, 
                               'fa_layer_edge_sampling_rmv': 0.8, 
                               'gpu': 0, 
                               'pos_enc_csv': False, 
                               'pos_dist_quantile': 0.001, 
                               'adaptive': False, 
                               'attention_rewiring': False, 
                               'baseline': False, 
                               'cpus': 1, 
                               'dt': 0.001, 
                               'dt_min': 1e-05, 
                               'gpus': 0.5, 
                               'grace_period': 20, 
                               'max_epochs': 1000, 
                               'metric': 'accuracy', 
                               'name': 'cora_beltrami_splits', 
                               'num_init': 1, 
                               'num_samples': 1000, 
                               'patience': 100, 
                               'reduction_factor': 10, 
                               'regularise': False, 
                               'use_lcc': True}
    sml_dict['method'] = 'rk4'
    # The Answer to the Ultimate Question of Life, the Universe, and Everything is 42
    sml_dict['time'] = 42 / 10
    sml_dict['hidden_dim'] = 512
    # sml_dict['max_nfe'] = 1e5
    batch_size = 32
    num_epochs = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Print(device)
    
    dataset = torch_geometric.datasets.Planetoid(root, name=args.dataset, split='public')
    
    sml_dict['dataset'] = args.dataset
    #i = int(args.segment)

    dataset.x.to(device)
    dataset.edge_index.to(device)
    dataset.y.to(device)

        
    
    sml_dict['dataset'] = args.dataset
    #with open(f'what_matters_{sml_dict["dataset"]}', 'w'):
    #    pass
    
    
    train_folds = dataset.train_mask.reshape(-1, 1).to(device)
    dev_folds = dataset.val_mask.reshape(-1, 1).to(device)
    test_folds = dataset.test_mask.reshape(-1, 1).to(device)

    list_dims = [64, 80, 256, 384, 512]
    drop_list = [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.4]
    time_list = [2, 4, 6, 10, 18.2]
    lr_list = [1e-4, 3e-4, 1e-3]
    # method_list = ['dopri5', 'euler', 'rk4]
    from itertools import product
    space = list(product(list_dims, drop_list, time_list, lr_list))
    
    size = len(space) // 6
    i = int(args.segment) * size
    while i < len(space):
        for dim, drop, time, lr in space[i: i+size]:
            Print(dim, drop, time, lr)
            sml_dict['time'] = time 
            sml_dict['hidden_dim'] = dim
            devs = []
            tests = []

            for fold in range(1):
                train_mask = train_folds[:, fold]
                
                dev_mask = dev_folds[:, fold]
                test_mask = test_folds[:, fold]


                g_unet = GraphUnet([0.9, 0.8, 0.7], drop, dataset, sml_dict).to(device)
                optmz = torch.optim.Adam(g_unet.parameters(), lr=lr)
                lf = torch.nn.NLLLoss()

                dev, test, max_dev, test_at_max_dev = train_pipeline(
                    fold, 
                    g_unet, 
                    train_mask, 
                    dev_mask, 
                    test_mask, 
                    dataset, 
                    lf, 
                    optmz, 
                    batch_size, 
                    num_epochs
                )
                tests.append(test)
                devs.append(dev)

                with open(f'what_matters_{sml_dict["dataset"]}', 'a') as f:
                    f.write(f'{args.dataset}, {dim}, {drop}, {time}, {lr}\n')                                                                                                     
                    f.write(f'{max_dev}, {test_at_max_dev}\n===========================\n')
        break

        
    
    
