# Third party import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel

class GeneralizedSupervisedNTXenLoss(nn.Module):
    def __init__(self, config, temperature, return_logits, sigma): # ADNI
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)

        :param temperature:
        :param return_logits:
        """

        # sigma = prior over the label's range
        super().__init__()
        self.config = config # ADNI
        self.sigma = sigma
        '''
        if self.kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        else:
            assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        '''
        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

    def forward(self, z_i, z_j, labels): # ADNI
        N = len(z_i)
        assert N == len(labels), "Unexpected labels length: %i"%len(labels)
        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        ### ADNI
        weights_list = []
        for i in range(len(self.config.label_name)):
            labels_tmp = labels[:, [i]].flatten()
            all_labels = labels_tmp.view(N, -1).repeat(2, 1).detach().cpu().numpy() # [2N, *]
            kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma[i]**2))
            weights = kernel(all_labels, all_labels) # [2N, 2N]
            if self.config.label_type[i] == 'cat':
                #constant = kernel(np.array([[0]]), np.array([[1]]))
                weights[weights != 1.] = self.config.cat_similarity[i]

            weights = weights * (1 - np.eye(2*N)) # puts 0 on the diagonal
            weights /= weights.sum(axis=1)
            weights_list.append(weights)
        ###

        # if 'rbf' kernel and sigma->0, we retrieve the classical NTXenLoss (without labels)
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1), torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0) # [2N, 2N]
        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        ### ADNI
        loss_list = []
        alpha_list = self.config.alpha_list
        for i in range(len(self.config.label_name)):
            loss = -1./N * (torch.from_numpy(weights_list[i]).to(z_i.device) * log_sim_Z).sum() * torch.tensor(alpha_list[i])
            loss_list.append(loss)
        loss = torch.stack(loss_list, dim=0).sum()
        ###

        correct_pairs = torch.arange(N, device=z_i.device).long()

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss

    def __str__(self):
        return "{}(temp={}, sigma={}, see losses.py for kernel info)".format(type(self).__name__, self.temperature, self.sigma) # ADNI


class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations, arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
