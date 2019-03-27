import mxnet as mx
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import time

# Device configuration
device = mx.cpu() if len(mx.test_utils.list_gpus()) == 0 else mx.gpu()
print ('current ctx =>', device)

# Hyper-parameters
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.01


# Image preprocessing modules
transformer = transforms.Compose([transforms.RandomFlipLeftRight(),
                                  transforms.RandomResizedCrop(32),
                                  transforms.ToTensor()])


# CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root = '../../data', train=True)

train_dataset = train_dataset.transform_first(transformer)

test_dataset = datasets.CIFAR10(root = '../../data', train=False)

test_dataset = test_dataset.transform_first(transformer)


# DataLoader
train_loader = gluon.data.DataLoader(train_dataset,
                                      batch_size = batch_size,
                                      shuffle = True)
            
test_loader = gluon.data.DataLoader(test_dataset,
                                    batch_size = batch_size,
                                    shuffle = True)

class VGG(nn.HybridBlock):
    def __init__(self, layers, filters, classes=10, batch_norm=False, **kwargs):
        super(VGG, self).__init__(**kwargs)
        assert len(layers) == len(filters)
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm)
            self.features.add(nn.Dense(4096, activation='relu', 
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(nn.Dropout(rate=0.5))
            self.features.add(nn.Dense(4096, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(nn.Dropout(rate=0.5))
            self.output = nn.Dense(classes,
                                   weight_initializer='normal',
                                   bias_initializer='zeros')
        
    def _make_features(self, layers, filters, batch_norm):
        featurizer = nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                         weight_initializer='normal',
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(nn.BatchNorm())
                featurizer.add(nn.Activation('relu'))
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer
    
    def hybrid_forward(self, F, x):
        out = self.features(x)
        out = self.output(out)
        return out


# Specification
vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}

def get_vgg(num_layers, **kwargs):
    layers, filters = vgg_spec[num_layers]
    net = VGG(layers, filters, **kwargs)
    return net


model = get_vgg(16)
model.initialize(ctx=device)
model.hybridize()

# Loss and optimizer
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': learning_rate})

# Updating learning rate
def update_lr(optimizer, lr, epoch):
    if (epoch+1) % 5 == 0:
        new_lr = lr / 3
        optimizer.set_learning_rate(new_lr)
    return optimizer


# Train the model
total_step = len(train_loader)
cur_lr = learning_rate
for epoch in range(num_epochs):
    optimizer = update_lr(optimizer, cur_lr, epoch)
    print ('learning_rate', optimizer.learning_rate)
    for i, (images, labels) in enumerate(train_loader):
        images = images.as_in_context(device)
        labels = labels.as_in_context(device)

        # Forward pass
        with autograd.record():
            outputs = model(images).as_in_context(device)
            loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step(batch_size)

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.mean().asscalar()))


# Test the model
correct = 0
total = 0
for images, labels in test_loader:
    images = images.as_in_context(device)
    labels = labels.as_in_context(device)
    outputs = model(images).as_in_context(device)
    predicted = outputs.argmax(axis=1)
    correct += (predicted == labels.astype('float32')).sum().asscalar()
    total += labels.size

print ('Accuracy of the model on the {} test images: {}%'.format(total, 100 * correct / total))

# Save the params
model.save_parameters('vgg.params')