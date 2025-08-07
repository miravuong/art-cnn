import numpy as np
import torch
from torch import nn, utils
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import wget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")  # Make sure this says GPU!

# pre-process images
transform = transforms.Compose([transforms.ToTensor()])  # convert image to pytorch tensor

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

# these are the CIFAR-10 classes, 0 for plane, 1 for car, 2 for bird, etc.
classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# From cnn-cifar10.py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional layer
        # Here we're defining a standard layer with Convolution, BatchNorm, and dropout
                             #input output
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)  # b x 3 x 32 x 32 -> b x 32 x 16 x 16
        self.batchnorm1 = nn.BatchNorm2d(32)                               # (channel x height x width), b is batch size
        self.relu1 = nn.ReLU()  # Using ReLU activation function
        self.dropout1 = nn.Dropout(0.1)  # Adding dropout to prevent overfitting

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  # b x 32 x 16 x 16 -> b x 64 x 8 x 8
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)  # Adding a pooling layer to reduce spatial dimensions, b x 64 x 8 x 8 -> b x 64 x 4 x 4
        self.dropout2 = nn.Dropout(0.05)


        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # b x 64 x 4 x 4 -> b x 64 x 4 x 4.    #why stride DNE?
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.05)
        self.flatten = nn.Flatten()  # b x 64 x 4 x 4 -> b x (64 * 4 * 4)

        # Fully connected layer - classifying the features into 10 classes
        self.fc = nn.Linear(64 * 4 * 4, 128)  # 64 from the last conv layer, 10 for the number of classes, b x (64 * 4 * 4) -> b x 128
        self.relu4 =  nn.ReLU()
        self.fc1 = nn.Linear(128, 10)  # b x 128 -> b x 10
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


# Pull model weights from F23
weights_file = wget.download("https://github.com/kuanhenglin/ai-security-workshop/blob/f08ced8a4afb7de1120bfdbf468888c7be10fdd8/cifar10_model.pth?raw=true")

network = CNN()

network.load_state_dict(torch.load("model.pth", map_location=device))
network.to(device)
network.eval()

# Sanity Check: make sure we have loaded the correct model weights

@torch.no_grad()
def evaluate(loader, network):  # not the best implementation :P
    network.eval()
    accuracies = []

    for inputs, labels in loader:
        inputs = inputs.to(device)  # put inputs and labels on gpu
        labels = labels.to(device)
        outputs = network(inputs)  # pass inputs through network to get outputs

        accuracy = (torch.max(outputs, dim=1)[1] == labels).to(torch.float16).mean()  # accuracy
        accuracies.append(accuracy.cpu().numpy())

    return np.mean(accuracies)

accuracy = evaluate(test_loader, network)
print(f"Test accuracy: {str(accuracy * 100):.6}%")

# Display attack
def display_torch_image(image, label=None):
    if label is not None:  # add a title with label number and class name (if given)
        plt.title(f"{classes[label]} ({label})")
    plt.axis("off")  # turn off matplotlib axis (we don't need it for displaying images)
    # torch.moveaxis because torch images are C H W, but matplotlib wants H W C
    plt.imshow(torch.moveaxis(image, 0, -1).cpu(), vmin=0, vmax=1)

def display_attacked(image, image_attacked, noise, label, label_attacked):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].axis("off")
    axes[0].set_title(f"{classes[label]} ({label})", fontsize=16)
    axes[0].text(33.25, 16.5, "$+$", fontsize=24)
    axes[0].imshow(torch.moveaxis(image, 0, -1).cpu(), vmin=0, vmax=1)

    axes[1].axis("off")
    axes[1].set_title(f"noise (amplified 5x)", fontsize=16)
    axes[1].text(32.75, 16, "$=$", fontsize=24)
    axes[1].imshow(torch.moveaxis(noise, 0, -1).cpu(), vmin=0, vmax=1)

    axes[2].axis("off")
    axes[2].set_title(f"{classes[label_attacked]} ({label_attacked})", fontsize=16)
    axes[2].imshow(torch.moveaxis(image_attacked, 0, -1).cpu(), vmin=0, vmax=1)

# Attack implementation
def fgsm(network, image, label, epsilon=0.1, sign=True) -> np.ndarray:

    # forward pass
    image.requires_grad = True  # we will be computing gradients w.r.t. the image!

    output = network(image.unsqueeze(dim=0))[0]
    prediction =  output.max(dim=0)[1].cpu().numpy() # get network prediction, that is, the classification result
    loss = F.nll_loss(output, label) # get loss of output with the labels

    image_gradients = torch.autograd.grad(loss, image)[0]
    # get gradients of loss w.r.t. the image
    if sign:  # use sign of gradients (more stable, controlled magnitude)
        image_gradients =  image_gradients.sign()# get sign of image_gradients

    # fast gradient sign attack
    image_attacked =  (image + epsilon * image_gradients).clamp(0,1)# add noise (image_gradients) to image scaled by epsilon and clamped to [0, 1]
    output_attacked = network(image_attacked.unsqueeze(dim=0))[0] # pass attacked image through network
    # print(f'Output attacked: {output_attacked}')
    print(f'BAD: {output_attacked.max(dim=0)[1].cpu()}')
    print(f'GOOD: {output.max(dim=0)[1].cpu()}')
    prediction_attacked = output_attacked.max(dim=0)[1].cpu().numpy() # get network prediction from attacked output

    return (image_attacked, prediction_attacked, image_gradients)

# Display adversarial attack on an example image
def display_adversarial_attack(network, image, label, epsilon=0.1, fgsm_sign=True) -> None:
    batch_data = next(iter(train_loader))

    image = batch_data[0][0].to(device)
    label = batch_data[1][0].to(device)

    with torch.no_grad():
        prediction = network(image.unsqueeze(dim=0)).max(dim=1)[1][0].cpu().numpy()

    image_attacked, prediction_attacked, noise = fgsm(network,
                                                        image,
                                                        label,
                                                        epsilon=epsilon,
                                                        sign=fgsm_sign)

    display_attacked(image.detach(), image_attacked.detach(),
                    noise.detach() * 0.5 * (5 * epsilon) + 0.5,
                    label=prediction, label_attacked=prediction_attacked)


batch_data = next(iter(train_loader))
idx = random.randint(0, len(batch_data[0])-1)
image = batch_data[0][idx].to(device)
label = batch_data[1][idx].to(device)
with torch.no_grad():  # make sure original prediction is correct
    prediction = network(image.unsqueeze(dim=0)).max(dim=1)[1][0].cpu().numpy()
    display_torch_image(image, label=prediction)
