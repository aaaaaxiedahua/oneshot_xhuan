import os
import torch
import numpy as np
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy

class BaseModel(object):
    def __init__(self, args, loaders, samplers):
        self.args = args
        loader, val_loader, test_loader = loaders
        self.loader = loader
        self.model = GNN_auto(args, loader)
        self.model.cuda()
        self.n_ent = loader.n_ent
        self.n_samp_ent = args.n_samp_ent
        self.n_rel = loader.n_rel
        self.train_sampler, self.test_sampler = samplers
        self.trainLoader = DataLoader(loader, batch_size=args.n_batch, num_workers=args.cpu, collate_fn=loader.collate_fn, shuffle=False, prefetch_factor=args.cpu, pin_memory=True)
        self.valLoader = DataLoader(val_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=val_loader.collate_fn, shuffle=False, prefetch_factor=args.cpu, pin_memory=True)
        self.testLoader = DataLoader(test_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=test_loader.collate_fn, shuffle=False, prefetch_factor=args.cpu, pin_memory=True)
        self.optimizer = self._build_optimizer()
        min_group_lr = min(group['lr'] for group in self.optimizer.param_groups)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, min_lr=min_group_lr/20, verbose=True)
        self.smooth = 1e-5
        self.t_time = 0
        self.epoch_train_time = 0
        self.mean_rank_dict = {}
        self.epoch_idx = 0
        self.best_valid_mrr = 0.0

    def _build_optimizer(self):
        default_lr = float(self.args.lr)
        default_wd = float(self.args.lamb)
        param_groups = []
        special_param_ids = set()

        if getattr(self.args, 'use_input_refine', False) and hasattr(self.model, 'input_refine'):
            input_params = [p for p in self.model.input_refine.parameters() if p.requires_grad]
            if input_params:
                input_lr = float(self.args.input_lr) if float(getattr(self.args, 'input_lr', -1.0)) > 0 else default_lr
                input_wd = float(self.args.input_weight_decay) if float(getattr(self.args, 'input_weight_decay', -1.0)) >= 0 else default_wd
                param_groups.append({'params': input_params, 'lr': input_lr, 'weight_decay': input_wd})
                special_param_ids.update(id(p) for p in input_params)

        if getattr(self.args, 'use_layer_refine', False) and hasattr(self.model, 'layer_refine'):
            layer_params = [p for p in self.model.layer_refine.parameters() if p.requires_grad]
            if layer_params:
                layer_lr = float(self.args.layer_lr) if float(getattr(self.args, 'layer_lr', -1.0)) > 0 else default_lr
                layer_wd = float(self.args.layer_weight_decay) if float(getattr(self.args, 'layer_weight_decay', -1.0)) >= 0 else default_wd
                param_groups.append({'params': layer_params, 'lr': layer_lr, 'weight_decay': layer_wd})
                special_param_ids.update(id(p) for p in layer_params)

        default_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in special_param_ids]
        param_groups.insert(0, {'params': default_params, 'lr': default_lr, 'weight_decay': default_wd})
        return Adam(param_groups)
        
    def saveModelToFiles(self, args, best_metric, deleteLastFile=True):
        budget_tag = f'topk_{self.args.topk}'
        if getattr(self.args, 'use_relation_refine', False):
            final_topk = float(getattr(self.args, 'final_topk', -1.0))
            if final_topk > 0:
                budget_tag = f'ctopk_{self.args.topk}_ftopk_{final_topk}'
        if args.val_num == -1:
            savePath = f'{self.args.data_path}/saveModel/{budget_tag}_layer_{self.args.layer}_{best_metric}.pt'
        else:
            savePath = f'{self.args.data_path}/saveModel/{budget_tag}_layer_{self.args.layer}_valNum_{self.args.val_num}_{best_metric}.pt'
            
        print(f'Save checkpoint to : {savePath}')
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_mrr':best_metric,
                }, savePath)
        
    def loadModel(self, filePath):
        print(f'Load weight from {filePath}')
        assert os.path.exists(filePath)
        checkpoint = torch.load(filePath, map_location=torch.device(f'cuda:{self.args.gpu}'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # re-build optimizter
        self.optimizer = self._build_optimizer()
        min_group_lr = min(group['lr'] for group in self.optimizer.param_groups)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, min_lr=min_group_lr/20, verbose=True)

    def _new_relation_refine_tracker(self):
        return {
            'query_count': 0.0,
            'active_queries': 0.0,
            'changed_queries': 0.0,
            'coarse_nodes': 0.0,
            'refined_nodes': 0.0,
            'coarse_edges': 0.0,
            'refined_edges': 0.0,
        }

    def _new_path_memory_tracker(self):
        return {
            'query_count': 0.0,
            'real_node_count': 0.0,
            'active_node_count': 0.0,
            'state_norm_sum': 0.0,
            'abs_score_sum': 0.0,
        }

    def _update_module_trackers(self, relation_refine_tracker, path_memory_tracker, batch_stats):
        if batch_stats is None:
            return

        relation_refine_stats = batch_stats.get('relation_refine')
        if relation_refine_stats is not None and relation_refine_stats.get('enabled', 0.0) > 0.5:
            for key in relation_refine_tracker.keys():
                relation_refine_tracker[key] += float(relation_refine_stats.get(key, 0.0))

        path_memory_stats = batch_stats.get('path_memory')
        if path_memory_stats is not None and path_memory_stats.get('enabled', 0.0) > 0.5:
            for key in path_memory_tracker.keys():
                path_memory_tracker[key] += float(path_memory_stats.get(key, 0.0))

    def _format_relation_refine_tracker(self, phase, tracker):
        query_count = float(tracker['query_count'])
        if query_count <= 0:
            return None
        node_keep = tracker['refined_nodes'] / max(1.0, tracker['coarse_nodes'])
        edge_keep = tracker['refined_edges'] / max(1.0, tracker['coarse_edges'])
        return (
            f'==> RelationRefine[{phase}][epoch={self.epoch_idx}]: '
            f'queries={int(query_count)} '
            f'active={tracker["active_queries"] / query_count:.1%} '
            f'changed={tracker["changed_queries"] / query_count:.1%} '
            f'nodes={tracker["coarse_nodes"] / query_count:.1f}->{tracker["refined_nodes"] / query_count:.1f} '
            f'node_keep={node_keep:.1%} '
            f'edges={tracker["coarse_edges"] / query_count:.1f}->{tracker["refined_edges"] / query_count:.1f} '
            f'edge_keep={edge_keep:.1%}'
        )

    def _format_path_memory_tracker(self, phase, tracker):
        query_count = float(tracker['query_count'])
        real_node_count = float(tracker['real_node_count'])
        if query_count <= 0 or real_node_count <= 0:
            return None
        active_ratio = tracker['active_node_count'] / max(real_node_count, 1.0)
        mean_norm = tracker['state_norm_sum'] / max(real_node_count, 1.0)
        mean_abs_score = tracker['abs_score_sum'] / max(real_node_count, 1.0)
        return (
            f'==> PathMemory[{phase}][epoch={self.epoch_idx}]: '
            f'queries={int(query_count)} '
            f'active_nodes={active_ratio:.1%} '
            f'state_norm={mean_norm:.4f} '
            f'abs_score={mean_abs_score:.4f} '
            f'lambda={float(getattr(self.args, "path_lambda", 0.0)):.3f}'
        )

    def _format_epoch_summary(self, eval_info):
        current_lr = self.optimizer.param_groups[0]['lr']
        return (
            f'==> Epoch {self.epoch_idx:03d}: '
            f'lr={current_lr:.4e} '
            f'train_epoch={self.epoch_train_time:.2f}s '
            f'train_total={self.t_time:.2f}s '
            f'eval={eval_info["eval_time"]:.2f}s '
            f'best_valid={self.best_valid_mrr:.4f} '
            f'valid={eval_info["valid_mrr"]:.4f} '
            f'test={eval_info["test_mrr"]:.4f}'
        )

    def prepareData(self, batch_data):
        subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = batch_data
        subgraph_data = [batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs.cuda(), batch_sampled_edges.cuda()]
        subs = subs.cuda().flatten()
        rels = rels.cuda().flatten()
        objs = objs.cuda()
        return subs, rels, objs, subgraph_data
        
    def train_batch(self,):        
        epoch_loss = 0
        reach_tails_list = []
        train_relation_refine_tracker = self._new_relation_refine_tracker()
        train_path_memory_tracker = self._new_path_memory_tracker()
        t_time = time.time()
        self.model.train()
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.epoch_idx)
        
        for batch_data in tqdm(self.trainLoader, ncols=50, leave=False):                      
            # prepare data    
            subs, rels, objs, subgraph_data = self.prepareData(batch_data)
            
            # forward
            self.model.zero_grad()
            scores = self.model(subs, rels, subgraph_data)
            self._update_module_trackers(
                train_relation_refine_tracker,
                train_path_memory_tracker,
                self.model.pop_module_stats(),
            )
            
            # loss calculation
            pos_scores = scores[[torch.arange(len(scores)).cuda(), objs.flatten()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))

            # loss backward
            loss.backward()
            self.optimizer.step()

            # avoid NaN
            # for p in self.model.parameters():
            #     X = p.data.clone()
            #     flag = X != X
            #     X[flag] = np.random.random()
            #     p.data.copy_(X)

            # cover tail entity or not
            reach_tails = (pos_scores == 0).detach().int().reshape(-1).cpu().tolist()
            reach_tails_list += reach_tails
            epoch_loss += loss.item()
            
        self.epoch_train_time = time.time() - t_time
        self.t_time += self.epoch_train_time
        
        # evaluate on val/test set
        valid_mrr, out_str, eval_info = self.evaluate()
        self.best_valid_mrr = max(self.best_valid_mrr, valid_mrr)
        self.scheduler.step(valid_mrr)
        print(self._format_epoch_summary(eval_info))
        train_relation_refine_log = self._format_relation_refine_tracker('train', train_relation_refine_tracker)
        if train_relation_refine_log is not None:
            print(train_relation_refine_log)
        train_path_memory_log = self._format_path_memory_tracker('train', train_path_memory_tracker)
        if train_path_memory_log is not None:
            print(train_path_memory_log)
        for phase_log in eval_info['relation_refine_logs']:
            if phase_log is not None:
                print(phase_log)
        for phase_log in eval_info['path_memory_logs']:
            if phase_log is not None:
                print(phase_log)
        
        # shuffle train set
        if self.args.not_shuffle_train:
            pass
        else:
            self.loader.shuffle_train()
            fact_data = np.concatenate([np.array(self.loader.fact_data), self.loader.idd_data], 0)
            self.train_sampler.updateEdges(fact_data)
            # # ========== Module 1: Update adjacency lists after shuffle ==========
            # # Rebuild adj_by_rel from the new fact partition (patterns stay the same)
            # if hasattr(self.args, 'use_rel_prior') and self.args.use_rel_prior:
            #     self.train_sampler.updateRelationPatterns(fact_data)
            # # ========== End Module 1 ==========

        self.epoch_idx += 1
        return valid_mrr, out_str
    
    @torch.no_grad()
    def evaluate(self, eval_val=True, eval_test=True, verbose=False, rank_CR=False, mean_rank=False):
        ranking = []
        self.model.eval()
        i_time = time.time()
        valid_relation_refine_tracker = self._new_relation_refine_tracker()
        test_relation_refine_tracker = self._new_relation_refine_tracker()
        valid_path_memory_tracker = self._new_path_memory_tracker()
        test_path_memory_tracker = self._new_path_memory_tracker()
        
        # eval on val set
        if eval_val:
            val_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.valLoader, ncols=50, leave=False):      
                # prepare data            
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                # forward
                scores = self.model(subs, rels, subgraph_data, mode='valid').data.cpu().numpy()
                self._update_module_trackers(
                    valid_relation_refine_tracker,
                    valid_path_memory_tracker,
                    self.model.pop_module_stats(),
                )

                # calculate rank
                subs = subs.cpu().numpy()
                rels = rels.cpu().numpy()
                objs = objs.cpu().numpy()
                filters = []
                for i in range(len(subs)):
                    filt = self.loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((self.n_ent, ))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)
                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks
                
                if mean_rank: 
                    mean_ranks = cal_ranks_mean(scores, objs, filters)
                    mean_rank_list += mean_ranks

                # cover tails or not
                ans = np.nonzero(objs)
                ans_score = scores[ans].reshape(-1)
                reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                val_reach_tails_list += reach_tails

            ranking = np.array(ranking)
            v_mrr, v_h1, v_h10 = cal_performance(ranking)
            # print(f'[val]  covering tail ratio: {len(val_reach_tails_list)}, {1 - sum(val_reach_tails_list) / len(val_reach_tails_list)}')
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Val set:\n', rank_CR)
                
            # save mean rank
            if mean_rank: self.mean_rank_dict['val'] = copy.deepcopy(mean_rank_list)
                
        else:
            v_mrr, v_h1, v_h10 = -1, -1, -1
        
        # eval on test set
        if eval_test:
            ranking = []
            test_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.testLoader, ncols=50, leave=False):        
                # prepare data            
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                # forward
                scores = self.model(subs, rels, subgraph_data, mode='test').data.cpu().numpy()
                self._update_module_trackers(
                    test_relation_refine_tracker,
                    test_path_memory_tracker,
                    self.model.pop_module_stats(),
                )

                # calculate rank
                subs = subs.cpu().numpy()
                rels = rels.cpu().numpy()
                objs = objs.cpu().numpy()
                filters = []
                for i in range(len(subs)):
                    filt = self.loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((self.n_ent, ))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)
                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks

                if mean_rank: 
                    mean_ranks = cal_ranks_mean(scores, objs, filters)
                    mean_rank_list += mean_ranks
                    
                # cover tails or not
                ans = np.nonzero(objs)
                ans_score = scores[ans].reshape(-1)
                reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                test_reach_tails_list += reach_tails

            ranking = np.array(ranking)
            t_mrr, t_h1, t_h10 = cal_performance(ranking)
            # print(f'[test] covering tail ratio: {len(test_reach_tails_list)}, {1 - sum(test_reach_tails_list) / len(test_reach_tails_list)}')
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Test set:\n', rank_CR)
                
            # save mean rank
            if mean_rank: self.mean_rank_dict['test'] = copy.deepcopy(mean_rank_list)
            
        else:
            t_mrr, t_h1, t_h10 = -1, -1, -1
            
        i_time = time.time() - i_time
        current_lr = self.optimizer.param_groups[0]['lr']
        out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train_epoch:%.4f train_total:%.4f inference:%.4f \t[LR] %.4e\n'%(v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.epoch_train_time, self.t_time, i_time, current_lr)
        eval_info = {
            'valid_mrr': v_mrr,
            'test_mrr': t_mrr,
            'eval_time': i_time,
            'relation_refine_logs': [
                self._format_relation_refine_tracker('valid', valid_relation_refine_tracker),
                self._format_relation_refine_tracker('test', test_relation_refine_tracker),
            ],
            'path_memory_logs': [
                self._format_path_memory_tracker('valid', valid_path_memory_tracker),
                self._format_path_memory_tracker('test', test_path_memory_tracker),
            ],
        }
        return v_mrr, out_str, eval_info
