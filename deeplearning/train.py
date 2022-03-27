import copy
import torch
import torchvision
import torchvision.transforms as transforms

from models import CNN

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')

# paramater
batch_size = 128
lr = 0.001
epochs = 10

# pre-processing
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])

# data
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

dataset_size  = {}
dataset_size['train'] = len(train_dataset)
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

# test
with torch.no_grad():
    n_samples = 0
    n_corrects = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _,predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_corrects += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    print(f'accuracy: {100.0 *n_corrects / n_samples} %')
    
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {i}: {acc} %')

model_path = '../src/convnet_state.pth'
torch.save(model.to('cpu').state_dict(), model_path)