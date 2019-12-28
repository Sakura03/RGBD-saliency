import sys, os, random, torch, cv2
sys.path.insert(0, '.')
from os.path import join, exists
from PIL import Image
from glob import glob
from Models.BiSalNet_no_spatial import BiSalNet
import numpy as np
from torchvision import transforms
from parallel import DataParallelModel
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir', default="/media/data/zxy/saliency/RGBD-data", help='Data directory')
parser.add_argument('--inWidth', type=int, default=224, help='Width of RGB image')
parser.add_argument('--inHeight', type=int, default=224, help='Height of RGB image')
parser.add_argument('--savedir', type=str, default='./results/', help='Directory to save the results')
parser.add_argument('--datasets', type=str, default='["LFSD", "NJU2K", "NLPR", "RGBD135", "SIP", "SSD100", "STERE"]',
                    help='Datasets for testing (valid: LFSD, NJU2K, NLPR, RGBD135, SIP, SSD100, STERE)')
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
if not exists(join(args.savedir, "pred")):
    os.mkdir(join(args.savedir, "pred"))
    
for (key, value) in vars(args).items():
    print("{0:16} | {1}".format(key, value))
    
# check if processed data file exists or not

data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]

# load the model
model = BiSalNet()
model.eval()

if args.onGPU and torch.cuda.device_count() > 1:
    # model = torch.nn.DataParallel(model)
    model = DataParallelModel(model)
if args.onGPU:
    model = model.cuda()
    
# compose the data with transforms
val_transforms = transforms.Compose([
        transforms.Resize((args.inHeight, args.inWidth)),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)
        ])

if os.path.isfile(join(args.savedir, "checkpoint.pth")):
    print("=> loading checkpoint '{}'".format(join(args.savedir, "checkpoint.pth")))
    checkpoint = torch.load(join(args.savedir, "checkpoint.pth"))["state_dict"]
    if list(checkpoint.keys())[0][:7] == "module." and not isinstance(model, DataParallelModel):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
    elif list(checkpoint.keys())[0][:7] != "module." and isinstance(model, DataParallelModel):
        checkpoint = {"module."+key: value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
else:
    raise ValueError("Resuming checkpoint does not exists!")

for dataset in eval(args.datasets):
    if not exists(join(args.savedir, "pred", dataset)):
        os.mkdir(join(args.savedir, "pred", dataset))
    print("Testing on %s" % dataset)
    source_dir = join(args.data_dir, dataset, "RGB")
    target_fir = join(args.data_dir, dataset, "GT")
    if dataset in ["NJU2K", "NLPR"]:
        with open(join(args.data_dir, dataset, dataset+"_test.txt"), 'r') as f:
            test_paths = [join(source_dir, name.strip()+".jpg") for name in f.readlines()]
            f.close()
    else:
        test_paths = [path.replace("GT", "RGB")[:-3]+"jpg" for path in glob(target_fir + "/*")] # [path[:-3]+"jpg" for path in glob(target_fir + "/*")]
    
    with torch.no_grad():
        for it, path in enumerate(test_paths):
            image = Image.open(path).convert("RGB")
            size = image.size
            image = val_transforms(image).unsqueeze(0)
            if args.onGPU:
                image = image.cuda()
                
            pred, _, _, _ = model(image)
            # pred = model(image)
            pred = torch.squeeze(pred).cpu().numpy()
            pred = cv2.resize(pred, size)
            cv2.imwrite(join(args.savedir, "pred", dataset, path.split("/")[-1].replace(".jpg", ".png")), pred * 255.)
            
            print("%s [%d/%d] written to %s!" % (dataset, it, len(test_paths), join(args.savedir, "pred", dataset, path.split("/")[-1].replace(".jpg", ".png"))))
