import torch 
import torch.nn as nn
import torch.nn.functional as F

# exampler train__loader and test_loder 
train_loader = torch.utils.data.DataLoader
test_loader = torch.utils.data.DataLoader

# define nn class

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # making fully connected layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)   #op wali

    def forward(self,x):
        # relu on 1st layer 
        x=F.relu(self.fc1(x))
        x=self.fc2(x)  #op layer
        return F.log_softmax(x,dim=1)   #log softmax loss on op layer
    
# make model
model = NeuralNet()
# train
import torch.optim as optim
# cross entropy loss for classification
criterion = nn.CrossEntropyLoss()
# stochastic gd optimser we will use
optimizer = optim.SGD(model.parameters(), lr=0.01)

# noww looop training
for epoch in range(4):
    for inputs ,labels in train_loader:
        optimizer.zero_grad()             #clear the gredients
        output = model(inputs)            #forward pass
        loss = criterion(output,labels)   #loss calc
        loss.backward()                   #backward pass
        optimizer.step()                  #update weights


# evaluation of model
model.eval()
with torch.no_grad():
    correct = 0
    total= 0
    for inputs,labels in test_loader:
        output= model(inputs)
        _,predicted = torch.max(output.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy =100*correct/total
    print("accuracy = ",accuracy)






# actual setup of train and test loader using minst dataset




# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# # Define transformation to normalize data
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert images to tensors
#     transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and standard deviation
# ])

# # Download and load the training and testing datasets
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# # Create DataLoader instances to iterate through data in batches
# batch_size = 64

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
