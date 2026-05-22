import json
import os
import argparse
import torch
import time
import numpy as np
import pickle as pkl
from load_data import DataLoader
from base_model import BaseModel
from utils import *
from base_HPO import RF_HPO
from optuna_hpo import OptunaTPEHyperbandHPO
from PPR_sampler import pprSampler

HPO_search_space = {
    'lr': ('choice', [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]),
    'topk': ('choice', [0.09, 0.1, 0.11]),
    'hidden_dim': ('choice', [32, 48, 64, 128, 256]),
    'attn_dim': ('choice', [8, 16, 32, 64]),
    'n_layer': ('choice', [4, 6, 8]),
    'act': ('choice', ['relu', 'idd', 'tanh']),
    'initializer': ('choice', ['binary', 'relation']),
    'concatHidden': ('choice', [True, False]),
    'shortcut': ('choice', [True, False]),
    'readout': ('choice', ['linear', 'multiply']),
    'decay_rate': ('uniform', (0.8, 1)),
    'lamb': ('uniform', (1e-5, 1e-3)),
    'dropout': ('uniform', (0, 0.2)),
}

parser = argparse.ArgumentParser(description="Parser")
parser.add_argument('--data_path', type=str, default='data/WN18RR/')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--topk', type=float, default=0.1)
parser.add_argument('--topm', type=float, default=-1)
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
parser.add_argument('--not_shuffle_train', action='store_true')
parser.add_argument('--hpo_backend', type=str, choices=['legacy', 'optuna'], default='legacy')
parser.add_argument('--max_trials', type=int, default=10000000000)
parser.add_argument('--optuna_study_name', type=str, default='')
parser.add_argument('--optuna_storage', type=str, default='')
parser.add_argument('--optuna_startup_trials', type=int, default=3)
parser.add_argument('--optuna_ei_candidates', type=int, default=128)
parser.add_argument(
    '--start_config',
    type=str,
    default='',
    help='Optional Optuna start config. Accepts either a JSON string or a path to a JSON file.',
)
parser.add_argument('--use_qmgf', action='store_true')
parser.add_argument('--qmgf_hidden_dim', type=int, default=None)
parser.add_argument('--qmgf_temperature', type=float, default=None)
parser.add_argument('--use_ltsb', action='store_true')
parser.add_argument('--type_bias_weight', type=float, default=None)
args = parser.parse_args()


def load_manual_start_configs(start_config_arg):
    if start_config_arg == '':
        return []

    if os.path.exists(start_config_arg):
        with open(start_config_arg, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    else:
        payload = json.loads(start_config_arg)

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload

    raise ValueError('--start_config must be a JSON object, a JSON array of objects, or a path to a JSON file.')


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, int(args.cpu)))
    torch.multiprocessing.set_sharing_strategy('file_system')

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    args.dataset = dataset

    checkPath('./results/')
    checkPath(f'./results/{dataset}/')
    checkPath(f'{args.data_path}/saveModel/')

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(os.path.join(results_dir, dataset)):
        os.makedirs(os.path.join(results_dir, dataset))

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    args.perf_file = os.path.join(results_dir, dataset, timestamp + '.txt')
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    print('==> gpu:', gpu)
    args.n_batch = args.n_tbatch = int(args.batchsize)

    assert args.search or args.finetune

    with open(args.perf_file, 'a+') as f:
        f.write(str(args))

    loader = DataLoader(args, mode='train')
    val_loader = DataLoader(args, mode='valid')
    test_loader = DataLoader(args, mode='test')
    args.n_ent = loader.n_ent
    args.n_rel = loader.n_rel
    args.n_samp_ent = int(args.topk * loader.n_ent)
    args.n_samp_edge = int(args.topm * len(loader.fact_data)) if args.topm > 0 else -1

    test_data = loader.double_triple(loader.all_triple)
    test_homo_edges = list(set([(h, t) for (h, r, t) in test_data]))
    test_data = np.concatenate([np.array(test_data), loader.idd_data], 0)
    test_sampler = pprSampler(
        loader.n_ent,
        loader.n_rel,
        args.n_samp_ent,
        args.n_samp_edge,
        test_homo_edges,
        test_data,
        args.data_path,
        split='test',
        args=args,
    )
    del test_homo_edges

    fact_homo_edges = list(set([(h, t) for (h, r, t) in loader.fact_data]))
    fact_data = np.concatenate([np.array(loader.fact_data), loader.idd_data], 0)
    train_sampler = pprSampler(
        loader.n_ent,
        loader.n_rel,
        args.n_samp_ent,
        args.n_samp_edge,
        fact_homo_edges,
        fact_data,
        args.data_path,
        split='train',
        args=args,
    )
    del fact_homo_edges

    loader.addSampler(train_sampler)
    val_loader.addSampler(test_sampler)
    test_loader.addSampler(test_sampler)
    HPO_save_path = f'./results/{dataset}/search_log.pkl'

    if args.use_qmgf:
        HPO_search_space['concatHidden'] = ('choice', [False])
        HPO_search_space['qmgf_hidden_dim'] = ('choice', [16, 32, 64, 128])
        HPO_search_space['qmgf_temperature'] = ('choice', [0.5, 1.0, 1.5, 2.0])
        print('==> HPO: added query-adaptive multi-granularity fusion search space')
    if args.use_ltsb:
        HPO_search_space['type_bias_weight'] = ('choice', [0.05, 0.1, 0.2, 0.3])
        print('==> HPO: added latent type-aware score bias search space')

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

    def run_model(params, save_path=HPO_save_path, finetune_idx=-1, reporter=None):
        print(params)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        args.lr = params['lr']
        args.topk = float(params.get('topk', args.topk))
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
        args.use_qmgf = bool(params.get('use_qmgf', args.use_qmgf))
        if args.use_qmgf:
            args.concatHidden = False
        args.qmgf_hidden_dim = params.get('qmgf_hidden_dim', args.qmgf_hidden_dim)
        args.qmgf_temperature = params.get('qmgf_temperature', args.qmgf_temperature)
        args.use_ltsb = bool(params.get('use_ltsb', args.use_ltsb))
        args.type_bias_weight = params.get('type_bias_weight', args.type_bias_weight)

        args.n_samp_ent = max(1, int(args.topk * loader.n_ent))
        train_sampler.topk = args.n_samp_ent
        test_sampler.topk = args.n_samp_ent
        train_sampler.n_samp_ent = args.n_samp_ent
        test_sampler.n_samp_ent = args.n_samp_ent

        model = BaseModel(args, loaders=(loader, val_loader, test_loader), samplers=(train_sampler, test_sampler))

        best_mrr, best_test_mrr, bearing = 0, 0, 0
        for epoch in range(args.epoch):
            v_mrr, out_str = model.train_batch()
            if reporter is not None:
                reporter.report(epoch, v_mrr)

            with open(args.perf_file, 'a+') as f:
                f.write(out_str)

            if v_mrr > best_mrr:
                best_mrr = v_mrr
                best_str = out_str
                print(str(epoch) + '\t' + best_str)
                bearing = 0

                BestMetricStr = f'ValMRR_{str(v_mrr)[:5]}'
                model.saveModelToFiles(args, BestMetricStr, deleteLastFile=False)
            else:
                bearing += 1

            if bearing >= 3:
                print(f'early stopping at {epoch + 1} epoch.')
                break

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

    if args.search:
        print(f'==> HPO search mode ({args.hpo_backend} backend)')
        if args.hpo_backend == 'legacy':
            HPO_instance = RF_HPO(
                kgeModelName='redgnn',
                obj_function=run_model,
                dataset_name=args.dataset,
                HP_info=HPO_search_space,
                acq='EI',
            )

            if args.useSearchLog and os.path.exists(HPO_save_path):
                config_list, mrr_list = loadSearchLog(HPO_save_path)
                dataset_names = [args.dataset for _ in range(len(config_list))]
                HPO_instance.pretrain(config_list, mrr_list, dataset_names=dataset_names)

            sample_num = 1e4
            HPO_instance.runTrials(args.max_trials, sample_num, explore_trials=1e10)
        else:
            study_name = args.optuna_study_name if args.optuna_study_name != '' else f'redgnn-{args.dataset}'
            storage = args.optuna_storage if args.optuna_storage != '' else None
            HPO_instance = OptunaTPEHyperbandHPO(
                HPO_search_space,
                study_name=study_name,
                direction='maximize',
                seed=int(args.seed),
                metric_name='valid_mrr',
                n_startup_trials=args.optuna_startup_trials,
                n_ei_candidates=args.optuna_ei_candidates,
                storage=storage,
                enable_pruner=False,
            )
            start_configs = []
            if args.useSearchLog and os.path.exists(HPO_save_path):
                config_list, _ = loadSearchLog(HPO_save_path)
                start_configs.extend(config_list)
                print(f'==> Optuna: enqueue {len(config_list)} start config(s) from search log')
            if args.start_config != '':
                manual_start_configs = load_manual_start_configs(args.start_config)
                start_configs.extend(manual_start_configs)
                print(f'==> Optuna: enqueue {len(manual_start_configs)} manual start config(s)')
            if len(start_configs) == 0:
                start_configs = None

            def objective_fn(trial, config, reporter):
                return run_model(config, reporter=reporter)

            HPO_instance.optimize(
                objective_fn,
                n_trials=args.max_trials,
                start_configs=start_configs,
            )
            print(f'==> Optuna best valid_mrr={HPO_instance.best_value:.6f}')
            print(HPO_instance.best_config)

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
            if idx == -1:
                break
            run_model(param, finetune_idx=idx)
