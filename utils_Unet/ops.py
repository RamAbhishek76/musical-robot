import torch
import torch.nn as nn
import numpy as np
import torch_geometric

from colorama import Fore, Back, Style
def Print(text):
    print(Fore.RED + str(text))
    print(Style.RESET_ALL)

from GNN_early import GNNEarly

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
                               'method': 'dopri5', 
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
root = './data/CORA'


# Print([0, :].sum())


class GraphUnet(nn.Module):

    def __init__(self, ks, drop_p, dataset):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GNNEarly(opt=sml_dict, dataset=dataset)
        self.dataset = dataset

        dim = dataset.x.shape[-1]
        

        # self.bottom_gcn = GCN(dim, dim, act, drop_p)
        
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            gcn = GNNEarly(opt=sml_dict, dataset=dataset)
            # self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.down_gcns.append(gcn)
            gcn = GNNEarly(opt=sml_dict, dataset=dataset)
            # self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(gcn)
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))
        

    def to_adj(self, edge_index):
        adj = torch_geometric.utils.to_dense_adj(edge_index)
        return torch.squeeze(adj, 0)

    def forward(self, g, h):
        # h is dataset.x[:batch_size, :]
        # g is edge_index
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        
        for i in range(self.l_n):
            h = self.down_gcns[i](h)
            adj = self.to_adj(g)
            adj_ms.append(adj)
            down_outs.append(h)
            g, h, idx = self.pools[i](adj, h)
            indices_list.append(idx)
        h = self.bottom_gcn(g, h)
        
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs


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
        Print("pool")
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


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


if __name__ == '__main__':
    dataset = torch_geometric.datasets.Planetoid(root, name='Cora')
    g_unet = GraphUnet([0.9, 0.8, 0.7], 0.3)
    Print(g_unet(dataset.edge_index, dataset.x))
    