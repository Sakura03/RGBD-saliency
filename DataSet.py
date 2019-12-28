# import cv2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

class Dataset(Dataset):
    def __init__(self, imList, depthList, labelList, transform=None):
        self.imList = imList
        self.depthList = depthList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        image = Image.open(self.imList[idx]).convert("RGB")    # cv2.imread(self.imList[idx])
        depth = Image.open(self.depthList[idx]).convert("RGB") # cv2.imread(self.depthList[idx])
        label = Image.open(self.labelList[idx]).convert("L")   # cv2.imread(self.labelList[idx], 0)
        
        if self.transform is not None:
            [image, depth, label] = self.transform(image, depth, label)
        return (image, depth, label)
