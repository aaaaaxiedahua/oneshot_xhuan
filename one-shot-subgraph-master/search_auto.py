import os
import argparse
import json
import ast
import torch
import time
import numpy as np
import pickle as pkl
from load_data import DataLoader
from base_model import BaseModel
from utils import *
from base_HPO import RF_HPO
from PPR_sampler import pprSampler

HPO_search_space = {
        # discrete
        'lr':                    ('choice', [1e-2, 1e-3, 1e-4, 1e-5]),
        'hidden_dim':            ('choice', [16, 32, 48, 64, 128, 256]),
        'attn_dim':              ('choice', [2, 4, 8, 16, 32, 64]),
        'n_layer':               ('choice', [4, 6, 8, 10]),
        'act':                   ('choice', ['relu', 'idd', 'tanh']),
        'initializer':           ('choice', ['binary', 'relation']),
        'concatHidden':          ('choice', [True, False]),
        'shortcut':              ('choice', [True, False]),
        'readout':               ('choice', ['linear', 'multiply', 'pair_mlp']),

        # continuous
        'decay_rate':            ('uniform', (0.8, 1)),
        'lamb':                  ('uniform', (1e-5, 1e-3)),
        'dropout':               ('uniform', (0, 0.2)),
    }

# ========== QTAR search space ==========
HPO_search_space_QTAR = {
        'qtar_ratio_end':        ('choice', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'qtar_router_hidden':    ('choice', [32, 64, 128]),
    }

HPO_search_space_LQCD = {
        'lqcd_coarse_ratio':     ('choice', [1.25, 1.5, 2.0]),
        'lqcd_fuse_lambda':      ('uniform', (0.3, 0.9)),
        'lqcd_topl':             ('choice', [1, 2, 3]),
    }

# # ========== Module 1: Relation-Path Conditioned Sampling search space ==========
# HPO_search_space_M1 = {
#         'path_lambda':           ('uniform', (0.1, 2.0)),
#         'rel_path_topk':         ('choice', [5, 10, 20, 50]),
#         'fusion_mode':           ('choice', ['add', 'multiply']),
#     }
#
# # ========== Module 2: Relation Composition Augmentation search space ==========
# HPO_search_space_M2 = {
#         'compose_dim':           ('choice', [16, 32, 64, 128]),
#         'max_virtual':           ('choice', [20, 50, 100, 200]),
#         'compose_aware':         ('choice', [True, False]),
#         'rca_dropout':           ('uniform', (0, 0.9)),
#         'rca_mode':              ('choice', ['shared', 'per_layer']),
#         'compose_max_hop':       ('choice', [2, 3]),
#     }

parser = argparse.ArgumentParser(description="Parser")
parser.add_argument('--data_path', type=str, default='data/WN18RR/')
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--topk', type=float, default=0.1) # number of sampled nodes (for a subgraph)
parser.add_argument('--topm', type=float, default=-1) # number of sampled edges (for a subgraph)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--fact_ratio', type=float, default=0.75)
parser.add_argument('--val_num', type=int, default=-1)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--add_manual_edges', action='store_true')
parser.add_argument('--remove_1hop_edges', action='store_true')
parser.add_argument('--only_eval', action='store_true')
parser.add_argument('--useSearchLog', action='store_true')
parser.add_argument('--search', action='store_true')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--finetune_config', type=str, default='')
parser.add_argument('--start_config', type=str, default='')  # optional: JSON string or file path for seeded trial config
parser.add_argument('--not_shuffle_train', action='store_true')
parser.add_argument('--use_readout_refine', action='store_true')
# ========== LQCD args ==========
parser.add_argument('--use_lqcd', action='store_true')
parser.add_argument('--lqcd_coarse_ratio', type=float, default=1.5)
parser.add_argument('--lqcd_fuse_lambda', type=float, default=0.7)
parser.add_argument('--lqcd_topl', type=int, default=2)
# ========== QTAR args ==========
parser.add_argument('--use_qtar', action='store_true')
parser.add_argument('--qtar_ratio_start', type=float, default=1.0)
parser.add_argument('--qtar_ratio_end', type=float, default=0.8)
parser.add_argument('--qtar_warmup', type=int, default=5)
parser.add_argument('--qtar_router_hidden', type=int, default=64)
parser.add_argument('--qtar_min_edges', type=int, default=64)
# # ========== Module 1: Relation-Path Conditioned Sampling args ==========
# parser.add_argument('--use_rel_prior', action='store_true')         # enable path-based relation prior
# parser.add_argument('--path_lambda', type=float, default=0.5)       # weight for path prior in fusion
# parser.add_argument('--rel_path_topk', type=int, default=10)        # top-K relation path patterns per relation
# parser.add_argument('--fusion_mode', type=str, default='add')       # fusion: add / multiply
# # ========== Module 2: Relation Composition Augmentation args ==========
# parser.add_argument('--use_rca', action='store_true')               # enable RCA virtual edges
# parser.add_argument('--compose_dim', type=int, default=32)          # embedding dim for composition
# parser.add_argument('--max_virtual', type=int, default=50)          # max virtual edges per subgraph
# parser.add_argument('--compose_aware', action='store_true')         # composition-aware virtual edge embedding
# parser.add_argument('--rca_dropout', type=float, default=0.1)       # dropout in composition scorer
# parser.add_argument('--rca_mode', type=str, default='shared')       # shared / per_layer
# parser.add_argument('--compose_max_hop', type=int, default=2)       # max composition hops: 2 or 3
args = parser.parse_args()


def _sample_one_from_space(space):
    cfg = {}
    for hp_name, info in space.items():
        hp_type, hp_range = info[0], info[1]
        if hp_type == 'uniform':
            cfg[hp_name] = float(np.random.uniform(hp_range[0], hp_range[1]))
        else:
            cfg[hp_name] = np.random.choice(hp_range, 1)[0]
    return cfg


def _load_start_config(raw_value):
    if raw_value is None or str(raw_value).strip() == '':
        return None

    raw_value = str(raw_value).strip()
    data = None

    if os.path.exists(raw_value):
        if raw_value.endswith('.pkl'):
            data = pkl.load(open(raw_value, 'rb'))
        else:
            text = open(raw_value, 'r', encoding='utf-8').read()
            try:
                data = json.loads(text)
            except Exception:
                data = ast.literal_eval(text)
    else:
        try:
            data = json.loads(raw_value)
        except Exception:
            data = ast.literal_eval(raw_value)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        if len(data) == 0:
            return None
        if not all(isinstance(x, dict) for x in data):
            raise ValueError('--start_config list must contain dict items.')
        return data
    raise ValueError('--start_config must be a dict or list[dict].')


def _coerce_choice_value(value, choices):
    if value in choices:
        return value

    if isinstance(value, str):
        for caster in (int, float):
            try:
                casted = caster(value)
                if casted in choices:
                    return casted
            except Exception:
                pass

    if isinstance(value, (int, float)):
        for c in choices:
            if isinstance(c, (int, float)) and abs(float(c) - float(value)) < 1e-12:
                return c

    return None


def _build_start_candidates(raw_start_configs, hp_space):
    if raw_start_configs is None:
        return None

    start_candidates = []
    hp_keys = set(hp_space.keys())
    for idx, user_cfg in enumerate(raw_start_configs):
        seed_cfg = _sample_one_from_space(hp_space)
        unknown_keys = [k for k in user_cfg.keys() if k not in hp_keys]
        if len(unknown_keys) > 0:
            print(f'==> warning: ignored unknown start_config keys at index {idx}: {unknown_keys}')

        for hp_name, info in hp_space.items():
            if hp_name not in user_cfg:
                continue

            hp_type, hp_range = info[0], info[1]
            user_val = user_cfg[hp_name]

            if hp_type == 'uniform':
                user_val = float(user_val)
                lo, hi = float(hp_range[0]), float(hp_range[1])
                if user_val < lo or user_val > hi:
                    raise ValueError(f'start_config[{idx}]["{hp_name}"]={user_val} out of range [{lo}, {hi}]')
                seed_cfg[hp_name] = user_val
            else:
                coerced = _coerce_choice_value(user_val, hp_range)
                if coerced is None:
                    raise ValueError(f'start_config[{idx}]["{hp_name}"]={user_val} not in choices {hp_range}')
                seed_cfg[hp_name] = coerced

        start_candidates.append(seed_cfg)

    return start_candidates

if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    args.dataset = dataset
    
    # check all output paths
    checkPath('./results/')
    checkPath(f'./results/{dataset}/')
    checkPath(f'{args.data_path}/saveModel/')
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(os.path.join(results_dir, dataset)):
        os.makedirs(os.path.join(results_dir, dataset))
            
    time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    args.perf_file = os.path.join(results_dir,  dataset + '/' + time + '.txt')
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    print('==> gpu:', gpu)
    args.n_batch = args.n_tbatch = int(args.batchsize)
    
    assert args.search or args.finetune

    if args.use_qtar:
        HPO_search_space.update(HPO_search_space_QTAR)
        print('==> HPO: added QTAR search space')

    if args.use_lqcd:
        HPO_search_space.update(HPO_search_space_LQCD)
        print('==> HPO: added LQCD search space')

    # # conditionally extend search space based on enabled modules
    # if args.use_rel_prior:
    #     HPO_search_space.update(HPO_search_space_M1)
    #     print('==> HPO: added Module 1 (Relation-Aware Sampling) search space')
    # if args.use_rca:
    #     HPO_search_space.update(HPO_search_space_M2)
    #     print('==> HPO: added Module 2 (RCA) search space')

    with open(args.perf_file, 'a+') as f:
        f.write(str(args))
    
    loader = DataLoader(args, mode='train')
    val_loader = DataLoader(args, mode='valid')
    test_loader = DataLoader(args, mode='test')
    args.n_ent = loader.n_ent
    args.n_rel = loader.n_rel
    args.n_samp_ent = int(args.topk * loader.n_ent)
    args.n_samp_edge = int(args.topm * len(loader.fact_data)) if args.topm > 0  else -1
    
    # sampler for testing
    test_data = loader.double_triple(loader.all_triple)
    test_homo_edges = list(set([(h,t) for (h,r,t) in test_data]))
    test_data = np.concatenate([np.array(test_data), loader.idd_data], 0)
    test_sampler = pprSampler(loader.n_ent, loader.n_rel, args.n_samp_ent, args.n_samp_edge,
        test_homo_edges, test_data, args.data_path, split='test', args=args)

    del test_homo_edges
        
    # sampler for training
    fact_homo_edges = list(set([(h,t) for (h,r,t) in loader.fact_data]))
    fact_data = np.concatenate([np.array(loader.fact_data), loader.idd_data], 0)
    train_sampler = pprSampler(loader.n_ent, loader.n_rel, args.n_samp_ent, args.n_samp_edge,
        fact_homo_edges, fact_data, args.data_path, split='train', args=args)
        
    del fact_homo_edges

    # add sampler to the data loaders
    loader.addSampler(train_sampler)
    val_loader.addSampler(test_sampler)
    test_loader.addSampler(test_sampler)
    HPO_save_path = f'./results/{dataset}/search_log.pkl'
    
    def loadSearchLog(file):
        assert os.path.exists(file)
        data = pkl.load(open(file, 'rb'))
        config_list, mrr_list = [], []
        for HP_key, HP_values in data.items():
            (best_mrr, best_test_mrr, params, opts) = HP_values
            config_list.append(params)
            mrr_list.append(best_mrr)

        print(f'==> load {len(config_list)} trials from file: {file}')
        return config_list, mrr_list
    
    def run_model(params, save_path=HPO_save_path, finetune_idx=-1):       
        print(params)
        args.lr = params['lr']
        args.decay_rate = params['decay_rate']
        args.lamb = params['lamb']
        args.hidden_dim = int(params['hidden_dim'])
        args.attn_dim = int(params['attn_dim'])
        args.n_layer = args.layer = int(params['n_layer'])
        args.dropout = params['dropout']
        args.act = params['act']
        args.initializer = params['initializer']
        args.concatHidden = params['concatHidden']
        args.shortcut = params['shortcut']
        args.readout = params['readout']

        if args.use_qtar:
            args.qtar_ratio_end = params['qtar_ratio_end']
            args.qtar_router_hidden = int(params['qtar_router_hidden'])

        if args.use_lqcd:
            args.lqcd_coarse_ratio = params['lqcd_coarse_ratio']
            args.lqcd_fuse_lambda = params['lqcd_fuse_lambda']
            args.lqcd_topl = int(params['lqcd_topl'])
            for sampler in (train_sampler, test_sampler):
                sampler.lqcd_coarse_ratio = float(args.lqcd_coarse_ratio)
                sampler.lqcd_fuse_lambda = float(args.lqcd_fuse_lambda)
                sampler.lqcd_topl = int(args.lqcd_topl)
                sampler.lqcd_logged = False

        # # Module 1: Relation-Path Conditioned Sampling params
        # if args.use_rel_prior:
        #     args.path_lambda = params['path_lambda']
        #     args.rel_path_topk = int(params['rel_path_topk'])
        #     args.fusion_mode = params['fusion_mode']
        #
        # # Module 2: RCA params
        # if args.use_rca:
        #     args.compose_dim = int(params['compose_dim'])
        #     args.max_virtual = int(params['max_virtual'])
        #     args.compose_aware = params['compose_aware']
        #     args.rca_dropout = params['rca_dropout']
        #     args.rca_mode = params['rca_mode']
        #     args.compose_max_hop = int(params['compose_max_hop'])

        # build model
        model = BaseModel(args, loaders=(loader, val_loader, test_loader), samplers=(train_sampler, test_sampler))

        # load pretrained weight (only for first trial)
        if args.weight != '':
            model.loadModel(args.weight)
            args.weight = ''

        # training
        best_mrr, best_test_mrr, bearing = 0, 0, 0
        for epoch in range(args.epoch):
            # v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, out_str = model.train_batch()
            v_mrr, out_str = model.train_batch()
            
            with open(args.perf_file, 'a+') as f:
                f.write(out_str)
                
            if v_mrr > best_mrr:
                best_mrr = v_mrr
                best_str = out_str
                print(str(epoch) + '\t' + best_str)
                bearing = 0
                
                # save model weight
                BestMetricStr = f'ValMRR_{str(v_mrr)[:5]}'
                model.saveModelToFiles(args, BestMetricStr, deleteLastFile=False)
            else:
                bearing += 1
                
            # early stop (3 as the threshould to boost searching)
            if bearing >= 3: 
                print(f'early stopping at {epoch+1} epoch.')
                break
        
        # save to local file
        if args.search:
            if not os.path.exists(save_path):
                HPO_records = {}
            else:
                HPO_records = pkl.load(open(save_path, 'rb'))
            HPO_records[str(args)] = (best_mrr, best_test_mrr, params, args)
            pkl.dump(HPO_records, open(save_path, 'wb'))
        elif args.finetune:
            assert finetune_idx != -1
            data = pkl.load(open(args.finetune_config, 'rb'))
            data[finetune_idx]['status'] = 'done'
            data[finetune_idx]['val_mrr'] = best_mrr
            data[finetune_idx]['test_mrr'] = best_test_mrr
            pkl.dump(data, open(args.finetune_config, 'wb'))
            
        return best_mrr

    # standard HPO pipeline (no best_configs start, random exploration for new model)
    if args.search:
        print('==> HPO search mode (random start)')
        HPO_instance = RF_HPO(kgeModelName='redgnn', obj_function=run_model, dataset_name=args.dataset, HP_info=HPO_search_space, acq='EI')
        start_candidates = None

        if args.start_config != '':
            user_start_cfg = _load_start_config(args.start_config)
            start_candidates = _build_start_candidates(user_start_cfg, HPO_search_space)
            print(f'==> HPO: loaded {len(start_candidates)} start candidate(s) from --start_config')

        if args.useSearchLog and os.path.exists(HPO_save_path):
            config_list, mrr_list = loadSearchLog(HPO_save_path)
            dataset_names = [args.dataset for i in range(len(config_list))]
            HPO_instance.pretrain(config_list, mrr_list, dataset_names=dataset_names)

        max_trials, sample_num = 1e10, 1e4
        HPO_instance.runTrials(max_trials, sample_num, explore_trials=1e10, start_candidate=start_candidates)
        
    elif args.finetune:
        print('==> HPO finetune mode')
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        def getNextConfig():
            data = pkl.load(open(args.finetune_config, 'rb'))
            for idx in range(len(data)):
                if data[idx]['status'] == 'none':
                    data[idx]['status'] = 'running'
                    pkl.dump(data, open(args.finetune_config, 'wb'))
                    return idx, data[idx]['param']
            return -1, None
        
        while True:
            idx, param = getNextConfig()
            print(idx, param)
            if idx == -1: break
            run_model(param, finetune_idx=idx)
