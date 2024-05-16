import torch
import torch.nn as nn
from torchvision import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.nn import functional as F
import seaborn as sns

train_losses = []
val_losses = []

DATA_DIR = "."
download_dataset = False
train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset)

X_train = train_mnist.data.float()
y_train = train_mnist.targets
X_test = test_mnist.data.float()
y_test = test_mnist.targets

test_size = X_test.shape[0]
indices = np.random.choice(X_train.shape[0], test_size, replace=False)

X_valid = X_train[indices]
y_valid = y_train[indices]

X_train = np.delete(X_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

X_train = X_train.reshape(-1, 28 * 28)
X_valid = X_valid.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std
X_test = (X_test - mean) / std

batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, dropout_rate=0.2):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, feature):
        output = self.linear1(feature)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return output

def train_and_evaluate(num_epochs, learning_rate, batch_size, hidden_size):
    input_size = 28 * 28
    num_classes = 10
    dropout_rate = 0.2

    model = MultinomialLogisticRegression(input_size, num_classes, hidden_size, dropout_rate)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        train_losses.append(train_loss)
        val_losses.append(val_loss / total_val)

        val_loss = val_loss / total_val
        val_accuracy = 100 * correct_val / total_val

        print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
              .format(epoch + 1, num_epochs, train_loss, val_loss, val_accuracy))

    print("Training completed!")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    strides = np.array(range(10))
    confmat = np.zeros((10, 10))
    for target, pred in zip(all_labels, all_preds):
        confmat[target, pred] += 1

    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the model on the 10000 test images: {:.2f}%'.format(accuracy))
    print(f"Correct: {correct}, Total: {total}")

    plt.figure(figsize=(12, 9))
    sns.heatmap(confmat, annot=True, cmap='Blues', xticklabels=strides, yticklabels=strides)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    torch.save(model.state_dict(), 'trained_model.pth')

    return model

num_epochs = 30
learning_rate = 0.01
batch_size = 128
hidden_size = 512
model = train_and_evaluate(num_epochs, learning_rate, batch_size, hidden_size)
