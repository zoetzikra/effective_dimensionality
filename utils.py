import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Hugging Face
auth_token = "TODO"

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append((k, correct_k.mul_(100.0 / batch_size).detach().item()))
        return res

preprocess_yolo_raw = transforms.Compose([
    transforms.Resize(224, antialias=False),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
])

# ImageNet pre-processing for YoloV8 raw model
# about a 98% match rate vs using the wrapped model
# (tested in yolo_comparison notebook)
def preprocess_yolo(x):
    x = transforms.functional.to_tensor(x)
    x = transforms.functional.center_crop(x, min(x.shape[-2:]))
    return preprocess_yolo_raw(x)

# ImageNet pre-processing for everything other than Yolo
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----- Datasets -----
MAX_NUM = 2000 # maximum for validation is 50000

from pathlib import Path
from datasets import load_dataset
def get_imagenet_validation_set():
    my_file = Path("imagenet/val.pt")
    if not my_file.is_file():
        dataset = load_dataset('imagenet-1k', split='validation', streaming=True, use_auth_token=auth_token)
        
        print("== Building dataset ==")
        cached_dataset = { 'x': [], 'y': []}
        for xy in dataset:
            x, y = xy['image'], torch.tensor([xy['label']])
            if x.mode == 'L':
                x = x.convert('RGB')
            cached_dataset['x'].append(x)
            cached_dataset['y'].append(y)
        
            if len(cached_dataset['x']) % 250 == 0:
                print(f"* dataset size: {len(cached_dataset['x'])}")
        
            if len(cached_dataset['x']) == MAX_NUM:
                break
        print("== Cached dataset ==")
        torch.save(cached_dataset, f"imagenet/val.pt")
    else:
        print("== Using cached dataset ==")
        cached_dataset = torch.load(f"imagenet/val.pt")
    return cached_dataset

class ImageNet1k(torch.utils.data.Dataset):
    def __init__(self, cached_ds, transform=None, target_transform=None):
        self.img_labels = cached_ds['y']
        self.img_files = cached_ds['x']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_files[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def generate_imagenet_dataloader(is_yolo=False, batch_size=16, num_workers=1, pin_memory=True):
    cached_dataset = get_imagenet_validation_set()
    if is_yolo:
        image_net_dataset = ImageNet1k(cached_dataset, transform=preprocess_yolo)
    else:
        image_net_dataset = ImageNet1k(cached_dataset, transform=preprocess)
    
    image_net_dataloader = DataLoader(
                image_net_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=pin_memory
            )
    return image_net_dataloader