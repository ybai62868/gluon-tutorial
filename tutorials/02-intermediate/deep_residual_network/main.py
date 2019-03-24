# ----------------------------------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                                             #
# See section 4.2 for the model architecture on CIFAR-10                                                #
# Some part of the code was referenced from below                                                       #
# https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/resnet.py   #
# ----------------------------------------------------------------------------------------------------- #

import mxnet as mx
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

# Device configuration
device = mx.cpu() if len(mx.test_utils.list_gpus()) == 0 else mx.gpu()
print ('current ctx =>', device)

# Hyper-parameters
num_epochs = 80
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


# 3x3 convolution
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


class BasicBlockV1(nn.HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None
    
    def hybrid_forward(self, F, x):
        residual = x
        out = self.body(x)
        if self.downsample:
            residual = self.downsample(residual)
        out = F.Activation(residual+out, act_type='relu')
        return out


class BottleneckV1(nn.HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, 1, channels//4))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

        
    def hybrid_forward(self, F, x):
        residual = x
        out = self.body(x)
        if self.downsample:
            residual = self.downsample(residual)
        out = F.Activation(residual+out, act_type='relu')
        return out


class BasicBlockV2(nn.HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None
        
    def hybrid_forward(self, F, x):
        residual = x
        out = self.bn1(x)
        out = F.Activation(out, act_type='relu')
        if self.downsample:
            residual = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.Activation(out, act_type='relu')
        out = self.conv2(out)

        return out + residual

class BottleneckV2(nn.HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels//4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels//4, stride, channels//4)
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None


    def hybrid_forward(self, F, x):
        residual = x
        out = self.bn1(x)
        out = F.Activation(out, act_type='relu')
        if self.downsample:
            residual = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.Activation(out, act_type='relu')
        out = self.conv2(out)

        out = self.bn3(out)
        out = F.Activation(out, act_type='relu')
        out = self.conv3(out)

        return out + residual




class ResNetV1(nn.HybridBlock):
    def __init__(self, block, layers, channels, classes=num_classes, thumbnail=False,  **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))
            
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i]))
                self.features.add(nn.GlobalAvgPool2D())

                self.output = nn.Dense(classes, in_units=channels[-1])
    
    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels = in_channels,
                            prefix = ''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        
        return layer
    
    def hybrid_forward(self, F, x):
        out = self.features(x)
        out = self.output(x)

        return out


class ResNetV2(nn.HybridBlock):
    def __init__(self, block, layers, channels, classes=num_classes, thumbnail=False, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))
            
            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                  stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())

            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels = in_channels,
                            prefix = ''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        out = self.features(x)
        out = self.output(out)

        return out


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}
resnet_net_versions = [ResNetV1, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]



def get_resnet(version, num_layers):
    block_type, layers, channels = resnet_spec[num_layers]
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels)
    return net


model = get_resnet(2, 18)
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
model.save_parameters('ResNet.params')