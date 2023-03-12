import torch
import numpy as np

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', direction = "max"):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
            direction : whether to try to maximize/minimize the criteria 
                * if using `val_score` : use "min"
                * if using `val_auroc` : use "max"
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if direction == "min" :
            self.raw_best_score = np.Inf
        elif direction == "max" : 
            self.raw_best_score = -np.Inf
        self.delta = delta
        self.path = path
        
        self.direction = direction

    def __call__(self, val_score, model):
        #direction에 따라 score을 줄지 말지 고르기 
        if self.direction == "max": 
            score = val_score
        elif self.direction == "min":
            score = -val_score
        else : 
            raise ValueError("use either max/min for direction")
        ####
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: #i.e. score is higher or SAME than the best score #즉,same score이면 early stoping counter 을 안씀
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.raw_best_score:.6f} --> {val_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.raw_best_score = val_score