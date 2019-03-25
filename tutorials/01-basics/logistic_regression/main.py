import mxnet as mx
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

# Device configuration
device = mx.cpu() if len(mx.test_utils.list_gpus()) == 0 else mx.gpu()
print ('current ctx =>', device)

# Hyper-parameter
output_size = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.01


# FashionMNIST (images and labels)
transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(0.13, 0.31)])

train_dataset = datasets.FashionMNIST(root = '../../data',
                                      train = True)

train_dataset = train_dataset.transform_first(transformer)

test_dataset = datasets.FashionMNIST(root = '../../data',
                                     train = False)

test_dataset = test_dataset.transform_first(transformer)


# DataLoader
train_loader = gluon.data.DataLoader(dataset = train_dataset,
                                     batch_size = batch_size,
                                     shuffle = True)

test_loader = gluon.data.DataLoader(dataset = test_dataset,
                                    batch_size = batch_size,
                                    shuffle = False)


model = nn.Dense(output_size)
model.initialize(ctx = device)
model.hybridize()


# Loss and optimizer
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': learning_rate})


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).as_in_context(device)
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
    images = images.reshape(-1, 28*28).as_in_context(device)
    outputs = model(images).as_in_context(device)
    predicted = outputs.argmax(axis=1)
    total += labels.size
    correct += (predicted == labels.astype('float32')).sum().asscalar()

print ('Accuracy of the model on the {} test iamges: {}'.format(total, 100 * correct / total))

# Save the params
model.save_parameters('logistic_regression.params')

