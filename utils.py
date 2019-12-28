import os
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py

class SalEval:
    def __init__(self, nthresh=49):
        self.nthresh = nthresh
        self.thresh = torch.from_numpy(np.linspace(1./(nthresh + 1), 1. - 1./(nthresh + 1), nthresh)).float().cuda()
        self.EPSILON = 1e-8
        
        self.recall = torch.zeros(nthresh).cuda()
        self.precision = torch.zeros(nthresh).cuda()
        self.mae = 0
        self.num = 0
    
    @torch.no_grad()
    def addBatch(self, predict, gth):
        assert len(predict.shape) == 3 and len(gth.shape) == 3
        for t in range(self.nthresh):
            bi_res = predict > self.thresh[t]
            intersection = torch.sum(torch.sum(bi_res & gth.bool(), dim=1), dim=1).float()
            all_gth = torch.sum(torch.sum(gth, dim=1), dim=1).float()
            all_pred = torch.sum(torch.sum(bi_res, dim=1), dim=1).float()
            self.recall[t] += torch.sum(intersection / (all_gth + self.EPSILON))
            self.precision[t] += torch.sum(intersection / (all_pred + self.EPSILON))
            
        self.mae += torch.sum(torch.abs(gth.float() - predict)) / (gth.shape[1] * gth.shape[2])
        self.num += gth.shape[0]
    
    @torch.no_grad()
    def getMetric(self):
        tr = self.recall / self.num
        tp = self.precision / self.num
        MAE = self.mae / self.num
        F_beta = (1 + 0.3) * tp * tr / (0.3 * tp + tr + self.EPSILON)

        return torch.max(F_beta).cpu().item(), MAE.cpu().item()

class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger("Logger")
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.setLevel(logging.INFO)
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()
 
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, maxlen=100):
        self.reset()
        self.maxlen = maxlen

    def reset(self):
        self.memory = []
        self.avg = 0
        self.val = 0
        self.count = 0

    def update(self, val):
        if self.count >= self.maxlen:
            self.memory.pop(0)
            self.count -= 1
        self.memory.append(val)
        self.val = val
        self.sum = sum(self.memory)
        self.count += 1
        self.avg = self.sum / self.count

def plot_training_process(record, save_dir, bests):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].plot(record[0], linewidth=1.)
    axes[0, 0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0, 0].legend(["training loss scale1"], loc="upper right")
    axes[0, 0].set_xlabel("Iter")
    axes[0, 0].set_ylabel("training loss scale1")
        
    axes[1, 0].plot(record[1], linewidth=1.)
    axes[1, 0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1, 0].legend(["training loss scale2"], loc="upper right")
    axes[1, 0].set_xlabel("Iter")
    axes[1, 0].set_ylabel("training loss scale2")
        
    axes[0, 1].plot(record[2], linewidth=1.)
    axes[0, 1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0, 1].legend(["training loss scale3"], loc="upper right")
    axes[0, 1].set_xlabel("Iter")
    axes[0, 1].set_ylabel("training loss scale3")
    
    axes[1, 1].plot(record[3], linewidth=1.)
    axes[1, 1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1, 1].legend(["training loss scale4"], loc="upper right")
    axes[1, 1].set_xlabel("Iter")
    axes[1, 1].set_ylabel("training loss scale4")
    
    axes[0, 2].plot(record[4], linewidth=1.)
    axes[0, 2].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0, 2].legend(["training loss main"], loc="upper right")
    axes[0, 2].set_xlabel("Iter")
    axes[0, 2].set_ylabel("training loss main")
    
    axes[1, 2].plot(record['lr'], linewidth=1.)
    axes[1, 2].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1, 2].legend(["learning rate"], loc="upper right")
    axes[1, 2].set_xlabel("epoch")
    axes[1, 2].set_ylabel("learning rate")
   
    axes[0, 3].plot(record["val"]["F_beta"], linewidth=1., color="blue")
    axes[0, 3].plot(record["train"]["F_beta"], linewidth=1., color="orange")
    axes[0, 3].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0, 3].legend(["F_beta_val (Best: %.4f)" % bests["F_beta_val"], "F_beta_tr (Best: %.4f)" % bests["F_beta_tr"]], loc="lower right")
    axes[0, 3].set_xlabel("Epoch")
    axes[0, 3].set_ylabel("F_beta")
    
    axes[1, 3].plot(record["val"]["MAE"], linewidth=1., color="blue")
    axes[1, 3].plot(record["train"]["MAE"], linewidth=1., color="orange")
    axes[1, 3].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1, 3].legend(["MAE_val (Best: %.4f)" % bests["MAE_val"], "MAE_tr (Best: %.4f)" % bests["MAE_tr"]], loc="upper right")
    axes[1, 3].set_xlabel("Epoch")
    axes[1, 3].set_ylabel("MAE")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'record.pdf'))
    plt.close(fig)

def load_vgg16pretrain(model, vggmodel='vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = "module.backbone." + name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)

def kernel_function(x, sigma):
    K = torch.sum((x[:, None, :, :] - x[:, :, None, :]) ** 2, dim=-1)
    K = torch.exp(-K / sigma)
    return K

def hsic_single(x1, x2, sigma):
    """
    param::x1, x2: of size (c, n)
    """
    c, n = x1.size()
    Ks = []
    for x in [x1, x2]:
        d = F.pdist(x)
        idx = torch.triu_indices(c, c, 1)
        dist = torch.zeros((c, c)).to(x.device)
        dist[idx[0], idx[1]] = d
        dist = dist + dist.t()
        K = torch.exp(-dist ** 2 / (sigma * n))
        Ks.append(K)
    K12 = torch.matmul(Ks[0], Ks[1])
    hsic = torch.trace(K12) / (c - 1) ** 2 + \
           torch.mean(Ks[0]) * torch.mean(Ks[1]) * c ** 2 / (c - 1) ** 2 - \
           2 * torch.mean(K12) * c / (c - 1) ** 2
    return hsic

def hsic(feat1, feat2, sigma):
    """
    param::feat1, feat2: of size (b, c, h, w)
    """
    bs, c, _, _ = feat1.size()
    feat1 = feat1.view(bs, c, -1) # b x c x hw
    feat2 = feat2.view(bs, c, -1) # b x c x hw
    
    loss = sum(hsic_single(feat1[i, ...], feat2[i, ...], sigma) for i in range(bs))
    return loss        

if __name__ == "__main__":
    dim = [112, 112]
    c = 128
    x = torch.randn(5, c, *dim)
    y = torch.randn(5, c, *dim)
    z = x + 0.1 * torch.randn(5, c, *dim)
    for i in range(-10, 10):
        print(1.1**i, 
              hsic(x, y, 1.1**i), 
              hsic(x, z, 1.1**i), 
              hsic(x, x, 1.1**i)
              )
