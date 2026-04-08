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
        default_params = [p for p in self.model.parameters() if p.requires_grad]
        return Adam([{'params': default_params, 'lr': default_lr, 'weight_decay': default_wd}])
        
    def saveModelToFiles(self, args, best_metric, deleteLastFile=True):
        budget_tag = f'topk_{self.args.topk}'
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
        t_time = time.time()
        self.model.train()
        
        for batch_data in tqdm(self.trainLoader, ncols=50, leave=False):                      
            # prepare data    
            subs, rels, objs, subgraph_data = self.prepareData(batch_data)
            
            # forward
            self.model.zero_grad()
            scores = self.model(subs, rels, subgraph_data)
            
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
        
        # eval on val set
        if eval_val:
            val_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.valLoader, ncols=50, leave=False):      
                # prepare data            
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                # forward
                scores = self.model(subs, rels, subgraph_data, mode='valid').data.cpu().numpy()

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
        }
        return v_mrr, out_str, eval_info
