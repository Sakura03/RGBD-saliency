import sys
sys.path.insert(0, '.')
import loadData as ld
import os
from os.path import join, exists, isfile
import torch, torchvision
import pickle
import random
from Models.BiSalNet_no_spatial import BiSalNet
import numpy as np
import Transforms as myTransforms
import DataSet as myDataLoader
from parallel import DataParallelModel
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir', default="/media/data/zxy/saliency/RGBD-data", help='Data directory')
parser.add_argument('--inWidth', type=int, default=224, help='Width of RGB image')
parser.add_argument('--inHeight', type=int, default=224, help='Height of RGB image')
parser.add_argument('--num_workers', type=int, default=10, help='No. of parallel threads')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--savedir', default='./results/', help='Directory to save the results')
parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
parser.add_argument('--cached_data_file', default='NJU2K+NLPR_train.p', help='Cached file name')
parser.add_argument('--seed', type=int, default=666, help='random seed for reproducibility')
parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='Run on CPU or GPU. If TRUE, then GPU.')

args = parser.parse_args()

# Set random seeds
if args.onGPU and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ['PYTHONHASHSEED'] = str(args.seed)

# create the directory if not exist
if not exists(args.savedir):
    os.mkdir(args.savedir)
    
for (key, value) in vars(args).items():
    print("{0:16} | {1}".format(key, value))
    
# check if processed data file exists or not
if not isfile(args.cached_data_file):
    dataLoad = ld.LoadData(args.data_dir, args.cached_data_file)
    data = dataLoad.processData()
    if data is None:
        print('Error while pickling data. Please check.')
        exit(-1)
else:
    data = pickle.load(open(args.cached_data_file, "rb"))

data['mean'] = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
data['std'] = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)

# load the model
model = BiSalNet()
model.eval()

if args.onGPU and torch.cuda.device_count() > 1:
    # model = torch.nn.DataParallel(model)
    model = DataParallelModel(model)
if args.onGPU:
    model = model.cuda()
    
# compose the data with transforms
valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
        ])
# since we training from scratch, we create data loaders at different scales
# so that we can generate more augmented data and prevent the network from overfitting
valLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=args.onGPU
        )

if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    model.load_state_dict(torch.load(args.resume)["state_dict"])
else:
    raise ValueError("Resuming checkpoint does not exists!")

mean = torch.from_numpy(data['mean']).view(1, -1, 1, 1)
std = torch.from_numpy(data['std']).view(1, -1, 1, 1)

for it, (inp, target) in enumerate(valLoader):
    if args.onGPU:
        inp = inp.cuda()
    
    with torch.no_grad():
        # run the mdoel
        pred_main, pred_side1, pred_side2, pred_side3, pred_side4 = model(inp)
        
        inp = (inp.cpu() * std + mean).clamp(min=0., max=255.).div(255.)
        target = target.float().clamp(min=0., max=1.).unsqueeze(1).repeat(1, 3, 1, 1)
        pred_main = pred_main.cpu().clamp(min=0., max=1.).repeat(1, 3, 1, 1)
        pred_side1 = pred_side1.cpu().clamp(min=0., max=1.).repeat(1, 3, 1, 1)
        pred_side2 = pred_side2.cpu().clamp(min=0., max=1.).repeat(1, 3, 1, 1)
        pred_side3 = pred_side3.cpu().clamp(min=0., max=1.).repeat(1, 3, 1, 1)
        pred_side4 = pred_side4.cpu().clamp(min=0., max=1.).repeat(1, 3, 1, 1)
        
        assembled = torch.cat([inp, target, pred_main, pred_side1, pred_side2, pred_side3, pred_side4], dim=0)
        torchvision.utils.save_image(assembled, join(args.savedir, "batch%02d.png"%it), nrow=args.batch_size, range=(0, 1))
        print("Iter [{}/{}]".format(it, len(valLoader)))
