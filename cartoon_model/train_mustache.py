import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import argparse
import os

import time
from torch import nn, optim

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)


root = os.getcwd() + '/'
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)
parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--train_path', type=str, default=root,
                    help="""image dir path default: '../data/cifar10/'.""")
parser.add_argument('--test_path', type=str, default=root,
                    help="""image dir path default: '../data/cifar10/'.""")
parser.add_argument('--epochs', type=int, default=5,
                    help="""Epoch default:50.""")
parser.add_argument('--batch_size', type=int, default=32,
                    help="""Batch_size default:256.""")
parser.add_argument('--lr', type=float, default=0.001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--num_classes', type=int, default=2,
                    help="""num classes""")
parser.add_argument('--model_path', type=str, default='./model_mustache/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='mustache.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch', type=int, default=1)

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

# 准备数据集并预处理
transform_train = transforms.Compose([
    #transforms.RandomCrop(256, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = MyDataset(txt = args.train_path + 'mustache_train.txt', transform = transform_train)
print(len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print(len(train_loader))
test_dataset = MyDataset(txt = args.test_path + 'mustache_test.txt', transform = transform_test)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


def train():
    print(f"Train numbers:{len(train_dataset)}")

    # Load model
    # if torch.cuda.is_available():
    #     model = torch.load(args.model_path + args.model_name).to(device)
    # else:
    #     model = torch.load(args.model_path + args.model_name, map_location='cpu')
    model = torchvision.models.resnet18(pretrained = True)
    model.avgpool = nn.AvgPool2d(1, 1)
    model.fc = nn.Linear(2048 * 64, args.num_classes)
    model = model.to(device)
    # cast
    cast = nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # start time
        start = time.time()
        for i, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)
            print("{}\t{}".format(i, loss))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.display_epoch == 0:
            end = time.time()
            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")

            model.eval()

            correct_prediction = 0.0
            total = 0
            for i, (images, labels) in enumerate(test_loader):
                # to GPU
                images = images.to(device)
                labels = labels.to(device)
                # print prediction
                outputs = model(images)
                # equal prediction and acc
                _, predicted = torch.max(outputs.data, 1)
                # val_loader total
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()

            print(f"Acc: {(correct_prediction / total)}")

    # Save the model checkpoint
    torch.save(model, args.model_path + args.model_name)
    print(f"Model save to {args.model_path + args.model_name}.")


if __name__ == '__main__':
    train()
