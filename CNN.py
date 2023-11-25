import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models as models

from torchvision import datasets, transforms
from torch.autograd import Variable

# Define a data transformation to convert images to PyTorch tensors and normalize them
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Download CIFAR 10 Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download the CIFAR-10 training dataset
train_dataset_CIFAR = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Download the CIFAR-10 test dataset
test_dataset_CIFAR = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders for training and testing
train_loader_CIFAR = torch.utils.data.DataLoader(train_dataset_CIFAR, batch_size=64, shuffle=True)
test_loader_CIFAR = torch.utils.data.DataLoader(test_dataset_CIFAR, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        out = self.network(xb)
        # Flatten the output for fully connected layers
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out


@torch.no_grad()
def evaluate(model, data_loader):
  '''Evaluates the accuracy of the model.'''
  model.eval()
  correct_labels = 0
  total_no_predict = 0
  for images, labels in data_loader:
      outputs = model(images)
      img, predicted = torch.max(outputs, 1)
      total_no_predict += labels.size(0)
      correct_labels += (predicted == labels).sum().item()

  accuracy =  correct_labels / total_no_predict
  return accuracy

def evaluate_loss(model, dataset, loss_fn):
  '''Evaluates the average loss between the model outputs (predicted data) and
  real labels. For our case, the loss function is always cross entropy loss.
  '''
  model.eval()
  total_loss = 0
  samples = len(dataset)
  with torch.no_grad():
      for images, labels in dataset:
          outputs = model(images)
          loss = loss_fn(outputs, labels)
          total_loss += loss.item()

  average_loss = total_loss / samples
  return average_loss

def fit(epochs, lr, model, train_loader, test_loader, optimizer_name="SGD", optimizer=torch.optim.SGD, momentum_factor=None, betas=None):
    performance = []
    if momentum_factor:
      optimizer = optimizer(model.parameters(), lr, momentum=momentum_factor)
    else:
      if (optimizer_name == "Adam"):
        optimizer = optimizer(model.parameters(), lr, betas=betas)
      else:
        optimizer = optimizer(model.parameters(), lr)
    error = nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(epochs):
      # Train the model
      model.train()
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()

      # Evaluate performance
      train_acc = evaluate(model, train_loader)
      test_acc = evaluate(model, test_loader)
      train_accuracy.append(train_acc)
      test_accuracy.append(test_acc)
      loss_train = evaluate_loss(model, train_loader, error)
      loss_test = evaluate_loss(model, test_loader, error)
      train_loss.append(loss_train)
      test_loss.append(loss_test)

      # Store performance metrics
      performance.append({
          'epoch': epoch,
          'train_loss': train_loss,
          'train_accuracy': train_acc,
          'val_accuracy': test_acc
      })

      print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss_train:.4f}, '
            f'Train Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%}')

    return performance, train_accuracy, test_accuracy, train_loss, test_loss


def plot_loss(train_steps, test_steps):
  plt.plot(range(len(train_steps)), train_steps, 'b', label='Train Loss')
  plt.plot(range(len(test_steps)), test_steps, 'r', label='Test Loss')
  plt.title('Training and Test Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

def plot_accuracy(train_steps, test_steps):
  plt.plot(range(len(train_steps)), train_steps, 'b', label='Train Accuracy')
  plt.plot(range(len(test_steps)), test_steps, 'r', label='Test Accuracy')
  plt.title('Training and Test Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

# Training CNN on the Fashion MNIST dataset
model = CNN()
history = fit(20, 0.01, model, train_loader, test_loader)
plot_loss(history[3], history[4])
plot_accuracy(history[1], history[2])
