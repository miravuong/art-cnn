# CNN model implementation
# libraries we need
import torch # Pytorch 
import numpy as np # Numpy 
import matplotlib.pyplot as plt # Lets us display data
import torchvision # We use this for the dataset we'll use

import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm # Lets us see loop progress

# Set up environment

if torch.cuda.is_available(): # Checks if CUDA is availiable, loads the device for computation to the GPU
    device = torch.device('cuda:0')
    print('Running on GPU')
    print(torch.cuda.get_device_name(0))

else:
    device = torch.device('cpu')
    print('Running on CPU')


# Download CIFAR 10 training and testing data sets from torchvision.datasets
train_dataset = torchvision.datasets.CIFAR10(root='./cifar10', transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# Batching:
# Since each image is 32 x 32 x 3 (32 pixels width by 32 pixels length, and for each pixel we have R, G, B).
# We divide our dataset of images into multiple subsets, in which we run these through a neural network 
# and apply gradient descent on each one of the subsets.

#The dataloader class makes it easy for us to handle and randomize data
#The train and test loader both have a 128 sized batches of images, and are shuffled to increase randomization(improves performance)
train_loader = torch.utils.data.DataLoader(train_dataset, 128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, 128, shuffle=True)

# Visualizing a sample from train loader
print(train_dataset[5])

# Extracting an image and label from the dataset
train_iter = iter(train_loader)
batch_images, batch_labels = next(train_iter)
# The label is a number between 0 and 9 - why might that be?  Isn't the label a category, not a number?
# The image is represented as a 3 by 32 by 32 matrix (A 3-dimensional array)
image, label = batch_images[0], batch_labels[0]

print(image.shape)
plt.imshow(image.permute(1,2,0)) 

plt.show()

# Defining a CNN class.
# What's a CNN? It's a neural network that applies a convolution to the input. Convolution: slide a kernel (matrix)
# over the image and repeatedly take the dot product of that matrix
# Stride and padding and kernel size affect the convolution.

# Overfitting is a common problem with neural networks, we will be using a technique called dropout to counter this.
# Dropout is when neurons are removed. The problem with overfitting is when we have a complicated function
# tailered to the data. if we remove neurons this will cause us to not always have a full complicated function

# Normalize data to grant neural networks better ease at writing generalized functions
# Relu as our activation function-- gives the go on when to activate a neuron or not
# Pooling to scale down images. We will use max pooling 

# Here we will do matrix multiplication.

# IMPLEMENTATION

#Functions ->
# The def __init__(self) is a constructor, where you outline the different layers and aspects of your custom class
# def forward is the function for forward propogation you give it an input X and it outputs tensore

#Layers ->
#In pytorch a nn.Conv2d layer is a convolution 2d layer, the arguments are as follows
#nn.Conv2d(Number of Input features maps, Number of features maps, Kernel Size, Stride Size, Padding Size )
#nn.BatchNorm2d is a batch normalization layer that takes in a 2d tensor the argument is the number of input feature maps


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional layer
        # Here we're defining a standard layer with Convolution, BatchNorm, and dropout

        # b x 3 x 32 x 32 -> b x 32 x 16 x 16
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        # Using ReLU activation function
        self.relu1 = nn.ReLU()
        # Adding dropout to prevent overfitting (recommend a rate of 0.1)
        self.dropout1 = nn.Dropout(0.1)

        # Second convolutional layer
        # b x 32 x 16 x 16 -> b x 64 x 8 x 8
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        # Adding a pooling layer to reduce spatial dimensions, b x 64 x 8 x 8 -> b x 64 x 4 x 4
        self.pool2 = nn.MaxPool2d(2,2)
        # Recommend rate of 0.05
        self.dropout2 = nn.Dropout(0.05)

        # Third convolutional layer
        # b x 64 x 4 x 4 -> b x 64 x 4 x 4
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn. ReLU()
        # Recommend rate of 0.05
        self.dropout3 = nn.Dropout(0.05)
        self.flatten = nn.Flatten()  # b x 64 x 4 x 4 -> b x (64 * 4 * 4)

        # Fully connected layer - classifying the features into 10 classes
        # 64 from the last conv layer, 10 for the number of classes, b x (64 * 4 * 4) -> b x 128
        self.fc = nn.Linear(64 * 4 * 4, 128)
        self.relu4 =  nn.ReLU()
        self.fc1 = nn.Linear(128, 10)  # b x 128 -> b x 10 - left this one in as a hint ;)

    # This is already done - we're just calling the functions we define
    def forward(self, x):
        # Describing the forward pass through the network
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # After all those conv layers we can finally pass into a fully connected layer
        # Think about it like the neural network does a bunch of pre processing to make the image easier to understand before looking at it
        x = self.flatten(x)  # Flattening the output of the conv layers for the fully connected layer
        x = self.fc(x)
        x = self.relu4(x)
        x = self.fc1(x)
        return x  # The softmax (or another activation) can be implicitly applied by the loss function

# We are creating an instance of our CNN model, after which we load to model to
# the device either GPU or CPU
model = CNN()

model.to(device)

#This is where we define our loss, in this case the loss is cross entropy
#Remember - the loss is a number that tells our model how good it's doing.  The specifics of how this works are a theory track topic
criterion = nn.CrossEntropyLoss()


#We define the optimizer here, the model.paramters() ar all the paramters of our model, lr is the learning rate
#Again, the specifics of how this optimizer works are a theory track topic
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

#This is the training loop, it will take the model, train loader, the optimizer and device
#It loops through each training data and trains the model
#Note the data is loaded in batches not single instances, this is important
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for i, batch in tqdm(enumerate(train_loader)):  # looping through
        inputs, labels = batch # The batch contains the inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)
        # TODO
        # Get the model output
        outputs = model.forward(inputs)
        # Calculate the loss (using the criterion defined above)
        loss = criterion(outputs, labels)
        # Call loss.backward - this actually computes the gradients
        loss.backward()
        # Step forward with the optimizer and then zero out the gradients
        optimizer.step()
        optimizer.zero_grad()
    print('End of epoch loss:', round(loss.item(), 3))



#This is the same as above except that there is no optimization just testing for accruacy
@torch.no_grad() # Letting torch know we don't need the gradients as we are only testing
def test(model, test_loader, device):
    # we've manually specified the classes - these are from the cifar-10 dataset
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Put the model in evaluation mode
    model.eval()
    correct = 0
    for i, batch in tqdm(enumerate(test_loader)):
         inputs, labels = batch
         inputs = inputs.to(device)
         labels = labels.to(device)
         # Get the output
         outputs = model(inputs)
         # Determine if it made the right prediction
         predictions = outputs.argmax(dim=1)
         # If it made the right prediction, increase the number correct
         correct += (predictions == labels)

    print(f"End of epoch accuracy: {100 * correct / len(test_dataset)}%")

    # visualizing the current model's performance
    for i in range(min(len(inputs), 8)):
        print('Guess:', classes[predictions[i]], '| Label:', classes[labels[i]])
        plt.imshow(inputs[i].cpu().permute(1,2,0))
        plt.show()
    

#This is where the training and testing loop is called
NUM_EPOCHS = 2 # One epoch is one loop through the training data

for epoch in range(NUM_EPOCHS):
    print("Epoch: ", epoch + 1)
    train_one_epoch(model, train_loader, optimizer, criterion, device)
    test(model, test_loader, device)

size = 0
for param in model.parameters():
    size += np.prod(param.shape)
print(f"Number of parameters: {size}")

# Sve the weights of your model
torch.save(model.state_dict(), "model.pth")

#Reload the weights you just saved
model_new = CNN()
model_new.load_state_dict(torch.load("model.pth"))
model_new.to(device)
model_new.eval()

test(model_new, test_loader, device)



