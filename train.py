import argparse
import json
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for neural network")
    parser.add_argument("data_dir", type=str, help="Directory of the data")
    parser.add_argument("--save_dir", type=str, help="Directory to save checkpoints", default="./")
    parser.add_argument("--arch",type=str, help='Model architecture: "vgg16" or "densenet121"',default="VGG16",)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.001)
    parser.add_argument("--hidden_units", type=int, help="Hidden units for classifier", default=500)
    parser.add_argument("--epochs", type=int, help="Number of Epochs", default=10)
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training if available")
    return parser.parse_args()

def load_data(data_dir):
  # Define transforms for training, validation, and testing data
  # --- Training Data ---
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),  # Randomly rotate images by 30 degrees
        transforms.RandomResizedCrop(224),  # Randomly resize images to 224x224
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize tensors according to ImageNet standards
    ])

  # --- Validation Data ---
    valid_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize images to 256x256
        transforms.CenterCrop(224),  # Crop the center 224x224 region
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize tensors according to ImageNet standards
    ])

  # --- Testing Data ---
    test_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize images to 256x256
        transforms.CenterCrop(224),  # Crop the center 224x224 region
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize tensors according to ImageNet standards
    ])

  # Define datasets using ImageFolder
  # 'train_dir', 'valid_dir', and 'test_dir' should point to the respective directories containing the image data
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

  # Load datasets into dataloaders
  # Dataloaders handle batching and shuffling of data
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return trainloader, validloader, testloader
  
def build_model(arch, hidden_units):
  # build and train your model
    
  model = models.vgg16(pretrained=True)

  for param in model.parameters():
    param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Replacing the pretrained classifier with the one above
    model.classifier = classifier
    return model.arch

def validation(model, validloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(validloader), correct / total
  
def main():
    # Define model and training parameters
    n_classes = 102
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()
    print_every = 40

    # Early stopping instance
    early_stopping = EarlyStopping()

    # Move model to GPU if available
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Train the model
    for epoch in range(args.epochs):
        since = time.time()
        running_loss = 0.0

        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if ii % print_every == 0:
                model.eval()
                valid_loss, accuracy = validation(model, validloader, criterion)
                print(f"Epoch: {epoch+1}/{args.epochs}\nTraining Loss: {running_loss/print_every:.3f} | Valid Loss: {valid_loss:.3f} | Valid Accuracy: {accuracy:.3f}")
                running_loss = 0.0
                model.train()

            # Manually decrease learning rate every 10 epochs
            if epoch % 10 == 0:
                new_lr = args.learning_rate * 0.1
                optimizer.param_groups[0]['lr'] = new_lr
                print(f"Updated learning rate to {new_lr}")

        if early_stopping(valid_loss):
            print("Early stopping triggered")
            break
            time_elapsed = time.time() - since
            print(f"Epoch time: {time_elapsed:.0f}s")

    # Define checkpoint dictionary

    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': args.epochs}

    save_path = args_in.save_dir + 'checkpoint.pth'
    torch.save(checkpoint, save_path)
    print("model checkpoint saved ")


if __name__ == '__main__':
  main()
