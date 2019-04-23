import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn
import time

''' 
CNN model based on lecture slides and LeNet-5 Model, albeit with just 1 fully connected layer

Each Convolution layer consists of:
x ->
1) Convolution
2) Non-Linearity
3) Spacial Pooling
4) Normalization
-> y

This implementation swaps non-linearity and spacial pooling order, 
since those 2's order doesn't really matter and doing so should speed up computation.

'''
class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        layer1_channel = 16
        layer2_channel = 32
        # conv layers
        self.conv1 = nn.Conv2d(1,layer1_channel,kernel_size=5)
        self.conv2 = nn.Conv2d(layer1_channel,layer2_channel,kernel_size=5)
        # Spacial Pooling
        self.maxpool = nn.MaxPool2d(2)
        # Non linearity
        self.relu = nn.ReLU()
        # Normalization
        self.normalize1 = nn.BatchNorm2d(layer1_channel)
        self.normalize2 = nn.BatchNorm2d(layer2_channel)
        # fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):

        # run x through the two conv layers
        x = self.normalize1(self.relu(self.maxpool(self.conv1(x))))
        x = self.normalize2(self.relu(self.maxpool(self.conv2(x))))
        # flatten output to feed to fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class Dataset(Dataset):

    def __init__(self, file_name_X, file_name_Y):

        self.X = np.load(file_name_X)
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        self.X = self.X.reshape(-1,1,28,28)
        self.Y = np.load(file_name_Y)
        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(self.Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

'''
Followed the tutorial here:
https://seba-1511.github.io/tutorials/beginner/blitz/neural_networks_tutorial.html
for the training portion


'''
if __name__ == "__main__":

    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running On:",device)

    # Define Constants
    num_classes = 10
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 200

    # Load data
    train_data = Dataset(file_name_X="data/x_train.npy", file_name_Y="data/y_train.npy")
    test_data = Dataset(file_name_X="data/x_test.npy", file_name_Y="data/y_test.npy")

    # only do batching and shuffling on train_data
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    #instance of the Conv Net
    cnn = CNN(num_classes)
    cnn.to(device)

    # Cross Entropy Loss and Gradient Descent
    crossEntropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.5)
    losses = []
    
    # Run training
    start = time.time()

    for epoch in range(num_epochs):

        s_epoch = time.time()
        loss_save = 0

        for i, (images, labels) in enumerate(train_data_loader):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            
            # Calculate 1 forward iteration
            outputs = cnn(images)
            # Calculate Cross entropy
            loss = crossEntropy(outputs, labels)
            # differentiate wrt to loss
            loss.backward()
            # update weights
            optimizer.step()
            # Clear computed gradients
            optimizer.zero_grad()
            # store loss
            loss_save += loss.item()
        
        losses.append(loss_save)
        e_epoch = time.time()
            
        print ('Epoch: ', epoch, ' Loss: ', losses[-1], " Took: ", e_epoch - s_epoch)

    end = time.time()
    print ("Elapsed:", end-start)                

    # Evaluation
    out = cnn(Variable(test_data.X.to(device)))
    y_pred = (out.data.argmax(dim=1)).cpu().numpy()
    y_test = test_data.Y.numpy()

    # get classification rates
    avg_class_rate = np.sum(np.equal(y_pred, y_test)) / len(y_test)

    class_rate_per_class = [0.0] * num_classes
    for c in range(num_classes):
        argC = np.argwhere(y_test==c)
        class_rate_per_class[c] = np.sum(np.equal(y_pred[argC], c)) / len(argC)

    print ("Accuracy:", avg_class_rate)
    print ("Rate Per Class:", class_rate_per_class)

    import plotting
    plotting.plot_confusion_matrix(y_test,y_pred)
    plotting.plot_loss_vs_epoch(losses)


