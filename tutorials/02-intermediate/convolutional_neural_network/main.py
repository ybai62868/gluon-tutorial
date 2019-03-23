import mxnet as mx
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import time

# Device configuration
device = mx.cpu() if len(mx.test_utils.list_gpus()) == 0 else mx.gpu()
print (device)

# Hyper-parameters
num_epochs = 3
num_classes = 10
batch_size = 100
learning_rate = 0.001

# FashionMNIST 
transformer = transforms.Compose([transforms.ToTensor(),
								  transforms.Normalize(0.13, 0.31)])

train_dataset = datasets.FashionMNIST(root = '../../data',
							  		  train = True)

train_dataset = train_dataset.transform_first(transformer)


test_dataset = datasets.FashionMNIST(root = '../../data',
	                				 train = False)

test_dataset = test_dataset.transform_first(transformer)

# Data Loader
train_loader = gluon.data.DataLoader(dataset = train_dataset,
									 batch_size = batch_size,
									 shuffle = True)

test_loader = gluon.data.DataLoader(dataset = test_dataset,
									batch_size = batch_size,
									shuffle = False)

# Convolutional nerual network (two convolutional layers)
class ConvNet(nn.HybridBlock):
	def __init__(self, num_classes = 10):
		super(ConvNet, self).__init__()
		with self.name_scope():
			self.layer1 = nn.HybridSequential()
			self.layer1.add(
					nn.Conv2D(16, kernel_size = (5, 5), strides = (1, 1), padding = (2, 2)),
					nn.BatchNorm(),
					nn.Activation('relu'),
					nn.MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
				)
			self.layer2 = nn.HybridSequential()
			self.layer2.add(
					nn.Conv2D(32, kernel_size = (5, 5), strides = (1, 1), padding = (2, 2)),
					nn.BatchNorm(),
					nn.Activation('relu'),
					nn.MaxPool2D(pool_size = (2, 2), strides = (2, 2))
				)
			self.dense = nn.Dense(num_classes)

	def hybrid_forward(self, F, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = F.reshape(out, shape=[0,-1])
		out = self.dense(out)
		return out

model = ConvNet(num_classes)
model.initialize(ctx = device)
model.hybridize()

# Loss and optimizer
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': learning_rate})

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# Move data and labels to device
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

print ('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

# Save the params
model.save_parameters('ConvNet.params')
