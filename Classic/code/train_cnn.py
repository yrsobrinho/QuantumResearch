import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

image_transforms = {"train": transforms.Compose([transforms.RandomRotation(15),
                                transforms.RandomVerticalFlip(),
                                transforms.GaussianBlur(kernel_size=(3, 3)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.1241, 0.1241, 0.1241], std=[0.1855, 0.1855, 0.1855])
                                ]),
                    "test": transforms.Compose([
                                transforms.GaussianBlur(kernel_size=(3, 3)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.1245, 0.1245, 0.1245], std=[0.1850, 0.1850, 0.1850])
                                ])
                    }


# Specify the directory containing the PNG images
train_path = 'C:\\Users\\enzog\\PycharmProjects\\PYTHON\\cancerResearch\\png\\train'
test_path = 'C:\\Users\\enzog\\PycharmProjects\\PYTHON\\cancerResearch\\png\\test'

# Create dataset and dataloader
train_dataset = ImageFolder(root=train_path, transform=image_transforms["train"])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = ImageFolder(root=test_path, transform=image_transforms["test"])
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze model parameters
for param in resnet50.parameters():
    param.requires_grad = False

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    #nn.ReLU(),
    #nn.Dropout(0.4),
    #nn.Linear(64, 1),
    nn.Sigmoid()
)


resnet50 = resnet50.to(device)

# Define Optimizer and Loss Function
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(resnet50.parameters(), lr=0.0001)
epochs = 100

for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch+1, epochs))
        # Set to training mode
        resnet50.train()
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1)
            labels = labels.to(device).type(torch.float)

            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = resnet50(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backpropagate the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            # Compute the accuracy
            predictions = outputs.round()
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

test_loss = 0.0
correct = 0.0
preds = []
labels_ = []

with torch.no_grad():
    resnet50.eval()
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.unsqueeze(1)
        labels = labels.to(device).type(torch.float)

        outputs = resnet50(inputs)
        preds.extend(outputs.round().cpu().numpy())
        labels_.extend(labels.round().cpu().numpy())

        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        correct += (outputs.round() == labels).float().sum().item()

acc = correct / len(test_loader.dataset)
avg_loss = test_loss / len(test_loader.dataset)
precision, recall, fscore, _ = precision_recall_fscore_support(labels_, preds, average="weighted", zero_division=np.nan)
print(f"Accuracy: {acc} | Precision: {precision} | Recall: {recall} | F1: {fscore} | Avg Loss: {avg_loss}")

conf_matrix = confusion_matrix(labels_, preds)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
