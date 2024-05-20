import torch
import os
import random
from sklearn.model_selection import train_test_split
from shutil import copyfile
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score
import time


# Define a new transform with additional data augmentations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


dataset_dir_train = os.path.join('Training data Path')
dataset_dir_test = os.path.join('testing data path')
dataset_dir_val = os.path.join('evaluation data path')



test_dataset = datasets.ImageFolder(dataset_dir_test, transform=transform)
train_dataset = datasets.ImageFolder(dataset_dir_train, transform=transform)
val_dataset = datasets.ImageFolder(dataset_dir_val, transform=transform)


class_names = os.listdir(dataset_dir_test)
num_classes = len(class_names)




num_epochs = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Uncomment the model you would like to use



###### the alexnet here
# model = models.alexnet(pretrained=False)
# model.classifier[6]=nn.Linear(4096,num_classes)
# Use a pre-trained model

###### the convnext_base here
# model = models.convnext_base(pretrained=True)
# model.classifier[2]=nn.Linear(1024,num_classes)


###### the resnet50 here
# model = models.resnet50(pretrained=False)
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, num_classes)


###### the vgg16 here
# model = models.vgg16(pretrained=False) # pretrained=False just for debug reasons
# model.classifier[6]=nn.Linear(4096,num_classes)

###### the efficientnet_b7 here
# model = models.efficientnet_b7(pretrained=False)
# model.classifier[1]=nn.Linear(2560,num_classes)



###### the mobilenet_v3_large here
# model = models.mobilenet_v3_large(pretrained=False)
# input_num = model.classifier[3].in_features
# model.classifier[3]=nn.Linear(input_num,num_classes)



model = model.to(device)


# Define the data loaders for training, validation, and testing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, pin_memory=True)
train_test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True)



# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)



prev_loss = float('inf')
patience = 5  # number of epochs to wait for the loss to decrease
counter = 0




scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.001)
start_time = time.time()
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    # Check if the loss has decreased from the previous epoch
    if epoch_loss >= prev_loss:
        counter += 1
        if counter >= patience:
            print('Validation loss has not improved for %d epochs. Stopping training.' % patience)
            break
    else:
        counter = 0
        prev_loss = epoch_loss
    print('the loss havent improved since %d epochs.' %counter)
    print(f'Training loss: {epoch_loss}')

# Decrease the learning rate by a factor of gamma every step_size epochs
    scheduler.step()
    
    # Evaluate the model on validation data
    model.eval()
    val_preds = []
    val_labels = []
    print('train accuracy')
    for inputs, labels in tqdm(train_test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_confusion = confusion_matrix(val_labels, val_preds)
    print(f'training accuracy: {val_acc*100}')
    print(f'Confusion matrix:\n{val_confusion}')
    precision, recall, fscore, support = score(val_labels, val_preds, average='micro')

    print('precision: {}'.format(precision*100))
    print('recall: {}'.format(recall*100))
    print('fscore: {}'.format(fscore*100))
    print('support: {}'.format(support))

    print('val accuracy')
    
    model.eval()
    val_preds = []
    val_labels = []
    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_confusion = confusion_matrix(val_labels, val_preds)
    print(f'Validation accuracy: {val_acc*100}')
    print(f'Confusion matrix:\n{val_confusion}')
    precision, recall, fscore, support = score(val_labels, val_preds, average='micro')

    print('precision: {}'.format(precision*100))
    print('recall: {}'.format(recall*100))
    print('fscore: {}'.format(fscore*100))
    print('support: {}'.format(support))
end_time = time.time()
acc_time = end_time - start_time
print(f"\nAcc execution time: {acc_time} seconds")


#### saving the model
torch.save(model.state_dict(), 'model name.pt')



# Evaluate the model on the testing set
model.eval()
y_true = []
y_pred = []
test_loss = 0.0
num_samples = 0

start_time = time.time()


for inputs, labels in tqdm(test_loader):
    model.eval()
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    test_loss += loss.item() * inputs.size(0)
    num_samples += inputs.size(0)
    _, predictions = torch.max(outputs, 1)
    y_true += labels.cpu().tolist()
    y_pred += predictions.cpu().tolist()

test_loss /= num_samples
test_acc = accuracy_score(y_true, y_pred)
conf_mat = confusion_matrix(y_true, y_pred)
print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
print(f'Confusion matrix:\n{conf_mat}')

precision, recall, fscore, support = score(y_true, y_pred, average='macro')

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

end_time = time.time()
acc_time = end_time - start_time
print(f"\nAcc execution time: {acc_time} seconds")

##### loading the model
model.load_state_dict(torch.load('model name.pt'))