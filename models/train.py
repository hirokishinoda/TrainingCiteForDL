import copy
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import numpy as np

from models import CNN

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# paramater
batch_size = 128
lr = 0.001
epochs = 50

# pre-processing
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

# data
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# separate dataset
#train_index, valid_index = train_test_split(range(len(train_dataset)), test_size=0.2)
#valid_dataset = Subset(train_dataset, valid_index)
#train_dataset = Subset(train_dataset, train_index)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=-1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

dataset_size  = {}
dataset_size['train'] = len(train_dataset)
#dataset_size['valid'] = len(valid_dataset)
dataset_size['test'] = len(test_dataset)
# model
model = CNN().to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# train
def train(data_loader, model, criterion, optimizer, epochs, dataset_size, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
    
    epoch_acc = 0.0
    epoch_loss = 0.0
    best_acc = 0.0
    for epoch in range(epochs):

        running_loss = 0.0
        running_correct = 0.0
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # predict and calc loss
            predict = model(images)
            _, pred = torch.max(predict, 1)
            loss = criterion(predict, labels)

            # weights
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 

            # loss and correct
            running_loss += loss.item() * images.size(0)
            running_correct += torch.sum(pred == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_correct.double() / dataset_size

        print(f'[train:{is_train}][epoch:{epoch+1}/{epochs}][loss:{epoch_loss:.4f}][acc:{epoch_acc:.4f}]')

        # save best weights
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f'[best accuracy:{best_acc:.4f}]')

    model.load_state_dict(best_model_wts)
    return model

model = train(train_loader, model, criterion, optimizer, epochs, dataset_size['train'], is_train=True)
model_path = '../src/convnet_state.pth'
torch.save(model.to('cpu').state_dict(), model_path)