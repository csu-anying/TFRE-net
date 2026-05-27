import numpy as np
import torch
import wandb
#��ǰ����ѵ���Ĵ����ļ�
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./data/model.pth', trace_func=print,auc_score_max = 0,val_loss_min = np.Inf,f1_score = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = val_loss_min
        self.early_stop = False
        self.val_loss_min = val_loss_min
        self.auc_score_max = auc_score_max
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model = None
        self.f1_auc = 0

    def __call__(self, val_loss, model,auc_score,f1_socre):

        score = val_loss
        # score = auc_score#׼ȷ����Ϊɸѡ
        auc_f1_score = auc_score + f1_socre
        if self.best_score is None or score < self.best_score + self.delta or auc_f1_score>self.f1_auc:
        # if self.best_score is None or score < self.best_score + self.delta:

            if self.best_score is None or score < self.best_score + self.delta:
                self.best_score = score
            self.save_checkpoint(val_loss, model,auc_f1_score)
            self.f1_auc = auc_f1_score
            self.counter = 0
        # if self.best_score is None or score > self.best_score + self.delta:
        #     self.best_score = score
        #     self.save_checkpoint(val_loss, model,auc_score)
        #     self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.best_score
        # elif score > self.best_score + self.delta:
        #     self.counter += 1
        #     self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True

    def save_checkpoint(self, val_loss, model,auc_f1_score):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).({self.f1_auc:.6f} --> {auc_f1_score:.6f}). Saving model...')
            # self.trace_func(f'Validation loss decreased ({self.auc_score_max:.6f} --> {auc_score:.6f}). Saving model...')
        # self.auc_score_max = auc_score
        self.val_loss_min = val_loss
        self.best_model = model.state_dict()  # Save the best model state_dict
        # Save the model file to disk
        torch.save(model.state_dict(), self.path)

        # Log the model as an artifact to W&B
        # artifact = wandb.Artifact('model', type='model')
        # artifact.add_file(self.path)
        # wandb.log_artifact(artifact)

    def load_best_model(self, model):
        '''Load the best model from saved state_dict'''
        if self.best_model:
            model.load_state_dict(self.best_model)

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#             path (str): Path for the checkpoint to be saved to.
#                             Default: 'checkpoint.pt'
#             trace_func (function): trace print function.
#                             Default: print
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.auroc_min = 0
#         self.delta = delta
#         self.path = path
#         self.trace_func = trace_func
#     def __call__(self, val_loss, model,auroc_score):
#
#         # score = -val_loss
#         score = auroc_score#׼ȷ����Ϊɸѡ
#
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model,auroc_score)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         # elif score > self.best_score + self.delta:
#         #     self.counter += 1
#         #     self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#         #     if self.counter >= self.patience:
#         #         self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model ,auroc_score)
#             self.counter = 0
#
#     def save_checkpoint(self, val_loss, model,auroc_score):
#         if self.verbose:
#         #     self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
#         # self.val_loss_min = val_loss
#             self.trace_func(f'Validation loss decreased ({self.auroc_min:.6f} --> {auroc_score:.6f}).')
#         self.auroc_min = auroc_score