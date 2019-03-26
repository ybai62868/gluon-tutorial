import mxnet as mx
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import numpy as np


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create ndarrays.
x = nd.array([1])
x.attach_grad()
w = nd.array([2])
w.attach_grad()
b = nd.array([3])
b.attach_grad()

# Build a computational graph.
with autograd.record():
    y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print (x.grad)      # x.grad = 2
print (w.grad)      # w.grad = 1
print (b.grad)      # b.grad = 1


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #
x = nd.random.randn(10, 3)
y = nd.random.randn(10, 2)

# Build a fully connected layer.
dense = nn.Dense(2)
dense.initialize()
dense.forward(x)
print ('w: ', dense.weight.data())
print ('b: ', dense.bias.data())

# Build loss function and optimizer.
lr = 0.01
criterion = gluon.loss.L2Loss()
optimizer = gluon.Trainer(dense.collect_params(), 'sgd', {'learning_rate':lr})

# Forward pass
pred = dense(x)

# Compute loss
with autograd.record():
    pred = dense(x)
    loss = criterion(pred, y)
print ('loss: ', loss.mean().asscalar())

# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', dense.weight.grad())
print ('dL/db: ', dense.bias.grad())

# 1-step gradient descent.
batch_size = 2
optimizer.step(batch_size)

# Print out the loss after 1-step gradient descent.
pred = dense(x)
loss = criterion(pred, y)
print ('loss after 1 step optimization: ', loss.mean().asscalar())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a mxnet ndarry.
y = nd.array(x)

# Convert the mxnet ndarry to a numpy array.
z = y.asnumpy()


# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #

# Download and construct CIFAR-10 dataset.
train_dataset = datasets.CIFAR10(root = '../../data',
                                 train = True)


# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (image.shape)
print (label)

# Data loader (this provides queues and threads in a very simple way).
train_loader = gluon.data.DataLoader(dataset = train_dataset,
                                     batch_size = 64,
                                     shuffle = True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = next(data_iter)

# Actual usage of the data loader is as below.
for images, labels, in train_loader:
    # Training code should be written here.
    pass


# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #

# You should build your custom dataset as below.
class CustomDataset(gluon.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()
train_loader = gluon.data.DataLoader(dataset = custom_dataset,
                                     batch_size = 64,
                                     shuffle = True)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = gluon.model_zoo.vision.resnet18_v1(pretrained = True)

# If you want to finetune only the top layer of the model, set as below.
finetnue_net = gluon.model_zoo.vision.resnet18_v1(classes = 100)
finetnue_net.features = resnet.features
finetnue_net.output.initialize(init = init.Xavier())
finetnue_net.output.collect_params().setattr('lr_mult', 10)

# Forward pass.
images = nd.random.randn(64, 3, 224, 224)
outputs = finetnue_net(images)
print (outputs.shape)              # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
resnet.save_parameters('model.params')
resnet2 = gluon.model_zoo.vision.resnet18_v1()
resnet_new = resnet2.load_parameters('model.params')
