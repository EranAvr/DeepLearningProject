import numpy as np  # linear algebra
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
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



def train(model, optimizer, data_loader: DataLoader):
    model.train()
    for epoch in range(EPOCHS):
        print(f'>>> Epoch #{epoch}')
        batch_count = 0
        img_checked = 0
        for batch in data_loader:
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


def test(model, data_loader: DataLoader, record=False):
    record_dir = 'test_records/'
    records_df = pd.DataFrame({'processed': [], 'missed': [], 'predictions': [], 'true labels': [], 'differences': []})
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    with torch.no_grad():
        img_count = 0
        model.eval()
        batch_count = 0
        for batch in data_loader:
            images, labels = batch
            predictions = model(images)
            predictions = pred_to_binary(predictions)

            img_count += len(images)
            misses_vector = np.bitwise_xor(predictions.numpy(), labels.numpy())
            img_missed = np.sum(misses_vector)

            print(f'processed: {img_count}/{len(test_dataset)}\t'
                  f'missed: {img_missed}/{len(predictions)}\n'
                  f'predis: {predictions}\nlabels: {labels}')
            if record:
                # build Dataframe records table:
                records_df.loc[len(records_df.index)] = [img_count, img_missed,
                                                         predictions.tolist(),
                                                         labels.tolist(),
                                                         misses_vector.tolist()]
                # plotting image grid:
                dim = int(np.sqrt(BATCH_SIZE))
                fig = plt.figure(figsize=(10., 10.))
                fig.suptitle(f'processed: {len(images)}   missed: {img_missed}')
                grid = ImageGrid(fig, 111,
                                 nrows_ncols=(dim, dim),
                                 axes_pad=0.5)
                idx = 0
                for ax, im in zip(grid, [img.permute(1, 2, 0) for img in images]):
                    ax.imshow(im)
                    ax.set_title(f'true: {test_dataset.classes[labels[idx]]}\n'
                                f'pred: {test_dataset.classes[predictions[idx]]}')

                    idx += 1
                #plt.show()
                plt.savefig(f'{record_dir}test_batch{batch_count}.png')
            batch_count += 1
    records_df.to_csv(f'{record_dir}citrus_dis_test_record.csv')
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
    my_resnet_model = nn.Sequential(resnet50(), nn.Linear(in_features=1000, out_features=2))
    #   Optimizer and Loss setup:
    optimizer = optim.Adam(params=my_resnet_model.parameters(), lr=LEARNING_RATE)
    loss_func = F.cross_entropy

    if CONFIG['LOAD_PREV']:
        print('Loading model state-dict.')
        sd_path = os.path.join(os.getcwd(), f'model_state_dict{5}.pth')
        my_resnet_model.load_state_dict(torch.load(sd_path))
    if CONFIG['TRAIN']:
        print('Training model.')
        train(my_resnet_model, optimizer, data_loader=train_loader)
    if CONFIG['TEST']:
        print('Testing model.')
        test(my_resnet_model, data_loader=test_loader, record=True)
