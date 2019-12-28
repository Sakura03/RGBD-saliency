import sys, os, torch, random ,pickle, time
sys.path.insert(0, '.')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import loadData as ld
import numpy as np
import torch.backends.cudnn as cudnn
import Transforms as myTransforms
import DataSet as myDataLoader
from Models.baseline_with_depth_share_dense_att import BiSalNet
from parallel import DataParallelModel, DataParallelCriterion
from argparse import ArgumentParser
from utils import SalEval, AverageMeter, Logger
from torch.nn.parallel.scatter_gather import gather
import torch.nn as nn
import torch.nn.functional as F

parser = ArgumentParser()
parser.add_argument('--data_dir', default="/media/data/zxy/saliency/RGBD-data", help='Data directory')
parser.add_argument('--inWidth', type=int, default=224, help='Width of RGB image')
parser.add_argument('--inHeight', type=int, default=224, help='Height of RGB image')
parser.add_argument('--max_epochs', type=int, default=50, help='Max. number of epochs')
parser.add_argument('--num_workers', type=int, default=5, help='No. of parallel threads')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training loader')
parser.add_argument('--test_batch_size', type=int, default=10, help='Batch size for testing loader')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--warmup', type=int, default=0, help='lr warming up epoches')
parser.add_argument('--scheduler', type=str, default="step", choices=["step", "poly", "cos"], help='Lr scheduler (valid: "step", "poly", "cos")')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma for multi-step lr scheduler')
parser.add_argument('--milestones', type=str, default='[]', help='milestones for multi-step lr scheduler')
parser.add_argument('--print_freq', default=30, type=int, help='frequency of printing training info')
parser.add_argument('--savedir', default='./results/', help='Directory to save the results')
parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
parser.add_argument('--cached_data_file', default='NJU2K+NLPR_train.p', help='Cached file name')
# parser.add_argument('--import_script', default='from Models.baseline_with_depth_share import BiSalNet',
#                     type=str, help='import script')
parser.add_argument('--seed', default=666, type=int, help='Random seed for reproducibility')
parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='Run on CPU or GPU. If TRUE, then GPU.')

args = parser.parse_args()

# exec(args.import_script)

cudnn.benchmark = False
cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

def adjust_lr(optimizer, epoch):
    if epoch < args.warmup:
        lr = args.lr * (epoch + 1) / args.warmup
    else:
        if args.scheduler == "cos":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / args.max_epochs))
        elif args.scheduler == "poly":
            lr = args.lr * (1 - epoch * 1.0 / args.max_epochs) ** 0.9
        else:
            lr = args.lr
            for milestone in eval(args.milestones):
                if epoch >= milestone: 
                    lr *= args.gamma
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def plot_training_process(record, save_dir, bests):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].plot(record["loss"], linewidth=1.)
    axes[0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0].legend(["training loss"], loc="upper right")
    axes[0].set_xlabel("Iter")
    axes[0].set_ylabel("training loss")
    
    axes[1].plot(record['lr'], linewidth=1.)
    axes[1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[1].legend(["learning rate"], loc="upper right")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("learning rate")
   
    axes[2].plot(record["val"]["F_beta"], linewidth=1., color="blue")
    axes[2].plot(record["train"]["F_beta"], linewidth=1., color="orange")
    axes[2].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[2].legend(["F_beta_val (Best: %.4f)" % bests["F_beta_val"], "F_beta_tr (Best: %.4f)" % bests["F_beta_tr"]], loc="lower right")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F_beta")
    
    axes[3].plot(record["val"]["MAE"], linewidth=1., color="blue")
    axes[3].plot(record["train"]["MAE"], linewidth=1., color="orange")
    axes[3].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[3].legend(["MAE_val (Best: %.4f)" % bests["MAE_val"], "MAE_tr (Best: %.4f)" % bests["MAE_tr"]], loc="upper right")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("MAE")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'record.pdf'))
    plt.close(fig)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, *inputs):
        pred_main, pred_side1, pred_side2, pred_side3, pred_side4, target = tuple(inputs)
        target = target.float()
        loss_main = F.binary_cross_entropy(pred_main.squeeze(1), target)
        loss_side1 = F.binary_cross_entropy(pred_side1.squeeze(1), target)
        loss_side2 = F.binary_cross_entropy(pred_side2.squeeze(1), target)
        loss_side3 = F.binary_cross_entropy(pred_side3.squeeze(1), target)
        loss_side4 = F.binary_cross_entropy(pred_side4.squeeze(1), target)
        return loss_main + 0.4*loss_side1 + 0.4*loss_side2 + 0.4*loss_side3 + 0.4*loss_side4

@torch.no_grad()
def val(epoch):
    # switch to evaluation mode
    model.eval()
    salEvalVal = SalEval()

    for it, (inp, depth, target) in enumerate(val_loader):
        if args.onGPU:
            inp, depth, target = inp.cuda(), depth.cuda(), target.cuda()
          
        start_time = time.time()
        # run the mdoel
        output = model(inp, depth)

        torch.cuda.synchronize()
        val_times.update(time.time() - start_time)

        if not args.onGPU or torch.cuda.device_count() <= 1:
            pred_main, pred_side1, pred_side2, pred_side3, pred_side4 = tuple(output)
            loss = criterion(pred_main, pred_side1, pred_side2, pred_side3, pred_side4, target)
        else:
            loss = criterion(output, target)
        val_losses.update(loss.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)[0]
        else:
            output = output[0]
        salEvalVal.addBatch(output.squeeze(1), target)
        
    F_beta, MAE = salEvalVal.getMetric()
    record["val"]["F_beta"].append(F_beta)
    record["val"]["MAE"].append(MAE)

    return F_beta, MAE

def train(train_loader, epoch):
    # switch to train mode
    model.train()
    salEvalTrain = SalEval()
    
    total_batches = len(train_loader)

    end = time.time()
    for it, (inp, depth, target) in enumerate(train_loader):
        if args.onGPU == True:
            inp, depth, target = inp.cuda(), depth.cuda(), target.cuda()

        start_time = time.time()
        # run the mdoel
        output = model(inp, depth)

        if not args.onGPU or torch.cuda.device_count() <= 1:
            pred_main, pred_side1, pred_side2, pred_side3, pred_side4 = tuple(output)
            loss = criterion(pred_main, pred_side1, pred_side2, pred_side3, pred_side4, target)
        else:
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.update(loss.item())
        train_batch_times.update(time.time() - start_time)
        train_data_times.update(start_time - end)
        record["loss"].append(train_losses.avg)
        
        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(output, target_device=0, dim=0)[0]
        else:
            output = output[0]
        salEvalTrain.addBatch(output.squeeze(1), target)

        if it % args.print_freq == 0:
            logger.info('Epoch [%d/%d] Iter [%d/%d] Batch time: %.3f Data time: %.3f loss: %.3f (avg: %.3f) lr: %.1e' % 
                        (epoch, args.max_epochs, it, total_batches, train_batch_times.avg, 
                         train_data_times.avg, train_losses.val, train_losses.avg, lr))
        end = time.time()
    
    F_beta, MAE = salEvalTrain.getMetric()
    return F_beta, MAE

# create the directory if not exist
if not os.path.exists(args.savedir):
    os.mkdir(args.savedir)
    
logger = Logger(os.path.join(args.savedir, "log.txt"))
logger.info('Called with args:')
for (key, value) in vars(args).items():
    logger.info("{0:16} | {1}".format(key, value))
    
# check if processed data file exists or not
if not os.path.isfile(args.cached_data_file):
    dataLoad = ld.LoadData(args.data_dir, args.cached_data_file)
    data = dataLoad.processData()
    if data is None:
        logger.info('Error while pickling data. Please check.')
        exit(-1)
else:
    data = pickle.load(open(args.cached_data_file, "rb"))

data['mean'] = [0.485 * 255., 0.456 * 255., 0.406 * 255.]
data['std'] = [0.229 * 255., 0.224 * 255., 0.225 * 255.]
# load the model
model = BiSalNet()

if args.onGPU and torch.cuda.device_count() > 1:
    # model = torch.nn.DataParallel(model)
    model = DataParallelModel(model)
if args.onGPU:
    model = model.cuda()
    
logger.info("Model Architecture:\n" + str(model))
total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
logger.info('Total network parameters: ' + str(total_paramters))

criterion = CrossEntropyLoss()
if args.onGPU and torch.cuda.device_count() > 1 :
    criterion = DataParallelCriterion(criterion)
if args.onGPU:
    criterion = criterion.cuda()

train_losses = AverageMeter()
train_batch_times = AverageMeter()
train_data_times = AverageMeter()
val_losses = AverageMeter()
val_times = AverageMeter()

record = {
        "loss": [], "lr": [], "val": {"F_beta": [], "MAE": []},
        "train": {"F_beta": [], "MAE": []}
        }
bests = {"F_beta_tr": 0., "F_beta_val": 0., "MAE_tr": 1., "MAE_val": 1.}

logger.info('Data statistics:')
logger.info("mean: [%.5f, %.5f, %.5f], std: [%.5f, %.5f, %.5f]" % (*data['mean'], *data['std']))

# compose the data with transforms

trainTransform = myTransforms.Compose([
        # myTransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.inWidth, args.inHeight),
        # myTransforms.RandomCropResize(int(7./224.*args.inWidth)),
        # myTransforms.RandomFlip(),
        myTransforms.ToTensor()
        ])
valTransform = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
        ])

train_set = myDataLoader.Dataset(data['trainIm'], data['trainDepth'], data['trainAnnot'], transform=trainTransform)
val_set = myDataLoader.Dataset(data['valIm'], data['valDepth'], data['valAnnot'], transform=valTransform)

train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.test_batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
        )
max_batches = len(train_loader) * 5

optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
logger.info("Optimizer Info:\n" + str(optimizer))

start_epoch = 0
if args.resume is not None:
    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        # args.lr = checkpoint['lr']
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))    

for epoch in range(start_epoch, args.max_epochs):
    # train for one epoch
    lr = adjust_lr(optimizer, epoch)
    length = len(train_loader)
    
    F_beta_tr, MAE_tr = train(train_loader, epoch)
    
    record["train"]["F_beta"].append(F_beta_tr)
    record["train"]["MAE"].append(MAE_tr)
    record["lr"].append(lr)
    # evaluate on validation set
    F_beta_val, MAE_val = val(epoch)
    if F_beta_val > bests["F_beta_val"]: bests["F_beta_val"] = F_beta_val
    if MAE_val < bests["MAE_val"]: bests["MAE_val"] = MAE_val
    if F_beta_tr > bests["F_beta_tr"]: bests["F_beta_tr"] = F_beta_tr
    if MAE_tr < bests["MAE_tr"]: bests["MAE_tr"] = MAE_tr
        
    torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_F_beta': bests["F_beta_val"],
            'best_MAE': bests["MAE_val"]
            }, os.path.join(args.savedir, 'checkpoint.pth'))

    # save the model also
    # model_file_name = os.path.join(args.savedir, 'model', 'model_epoch' + str(epoch + 1) + '.pth')
    # torch.save(model.state_dict(), model_file_name)

    logger.info("Epoch %d: F_beta (tr) %.4f (Best: %.4f) MAE (tr) %.4f (Best: %.4f) F_beta (val) %.4f (Best: %.4f) MAE (val) %.4f (Best: %.4f)" %
                (epoch, F_beta_tr, bests["F_beta_tr"], MAE_tr, bests["MAE_tr"], F_beta_val, bests["F_beta_val"], MAE_val, bests["MAE_val"]))
    plot_training_process(record, args.savedir, bests)

logger.close()
