import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder


BASE_DIR = 'D:\\Datasets\\good_citrus_dataset_cut'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transformer = torchvision.transforms.Compose(
        [  # Applying Augmentation
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
)


#   Possible test model:
"""
class ImageClassificationModel(nn.Module):
    def __init__(self, num_of_classes):
        super().__init__()
        self.network = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                          stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

                nn.Flatten(),
                nn.Linear(256 * 28 * 28, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, num_of_classes))
    def forward(self, xb):
        return self.network(xb)
"""


def train(model, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        print(f'>>> Epoch #{epoch}')
        batch_count = 0
        img_checked = 0
        for batch in train_loader:
            batch_count += 1
            images, labels = batch
            # 1. feed model
            predictions = model(images)
            # 2. calc loss
            loss = loss_func(predictions, labels)
            # backward propagataion
            optimizer.zero_grad()
            loss.backward()
            # optimizer step()
            optimizer.step()
            img_checked += len(images)
            print(f'Train: complete={round((img_checked/len(train_dataset))*100, 3)}%\timages '
                  f'processed={img_checked}/{len(train_dataset)}\t\tloss={loss}')
        torch.save(model.state_dict(), os.path.join(os.getcwd(), f'model_state_dict{5}.pth'))

def test(model, plot=True):
    with torch.no_grad():
        img_count = 0
        model.eval()
        for batch in test_loader:
            images, labels = batch
            predictions = model(images)
            predictions = pred_to_binary(predictions)

            img_count += len(images)
            img_missed = np.sum(np.logical_xor(predictions.numpy(), labels.numpy()))
            print(f'processed={img_count}/{len(test_dataset)}')
            print(f'predis: {predictions}')
            print(f'labels: {labels}')
            print(f'missed:   {img_missed}/{len(predictions)}')
            if plot:
                for i in range(len(images)):
                    plt.imshow(images[i].permute(1, 2, 0))
                    plt.title(f'label={test_dataset.classes[labels[i]]}\n'
                              f'predict={test_dataset.classes[predictions[i]]}')
                    plt.show()
def pred_to_binary(pred: torch.Tensor):
    dim = len(pred)
    pred = torch.sigmoid(pred).numpy()
    res = torch.tensor([pred[i].tolist().index(np.amax(pred[i])) for i in range(dim)])
    return res


if __name__ == '__main__':
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    CONFIG = {'TRAIN': False,
              'TEST': True,
              'LOAD_PREV': True}

    #   Datasets:
    train_dataset = ImageFolder(TRAIN_DIR, transform=transformer)
    test_dataset = ImageFolder(TEST_DIR, transform=transformer)
    #   Data Loaders:
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True)

    #   Model setup:
    torch.cuda.empty_cache()
    #model = ImageClassificationModel(num_of_classes=2).to(device)
    my_resnet = resnet50()
    model = nn.Sequential(my_resnet, nn.Linear(in_features=1000, out_features=2))
    #   Optimizer and Loss setup:
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss_func = F.cross_entropy

    if CONFIG['LOAD_PREV']:
        print('Loading model state-dict.')
        sd_path = os.path.join(os.getcwd(), f'model_state_dict{5}.pth')
        model.load_state_dict(torch.load(sd_path))
    if CONFIG['TRAIN']:
        print('Training model.')
        train(model, optimizer)
    if CONFIG['TEST']:
        print('Testing model.')
        test(model)
