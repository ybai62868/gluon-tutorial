import mxnet as mx
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn, rnn
from mxnet.gluon.data.vision import datasets, transforms

# Device configuration
device = mx.cpu() if len(mx.test_utils.list_gpus()) == 0 else mx.gpu()
print ('current ctx =>', device)

# Hyper-parameters
sequence_length = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# FashionMNIST (images and labels)
transformer = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root = '../../daata',
                                      train = True)

train_dataset = train_dataset.transform_first(transformer)

test_dataset = datasets.FashionMNIST(root = '../../data',
                                     train = False)

test_dataset = test_dataset.transform_first(transformer)


# DataLoader (input pipeline)
train_loader = gluon.data.DataLoader(dataset = train_dataset,
                                     batch_size = batch_size,
                                     shuffle = True)

test_loader = gluon.data.DataLoader(dataset = test_dataset,
                                    batch_size= batch_size,
                                    shuffle = False)


# Recurrent neural network (many-to-one)
class RNN(nn.Block):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = rnn.LSTM(hidden_size, num_layers)
        self.dense = nn.Dense(num_classes)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = nd.zeros((self.num_layers, x.shape[1], self.hidden_size), ctx=device)
        c0 = nd.zeros((self.num_layers, x.shape[1], self.hidden_size), ctx=device)


        # Forward propagate LSTM
        out, _ = self.lstm(x, [h0, c0])
        out = self.dense(out[:, -1, :])
        return out
    
model = RNN(hidden_size, num_layers, num_classes)
model.initialize(ctx=device)
model.hybridize()

# Loss and optimizer
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': learning_rate})


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, images.shape[2]).as_in_context(device)
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
    images = images.reshape(-1, sequence_length, images.shape[2]).as_in_context(device)
    lables = labels.as_in_context(device)
    outputs = model(images).as_in_context(device)
    predicted = outputs.argmax(axis=1)
    correct += (outputs == labels.astype('float32')).sum().asscalar()
    total += labels.size

print ('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total)) 


# Save the params
model.save_parameters('RNN.params')