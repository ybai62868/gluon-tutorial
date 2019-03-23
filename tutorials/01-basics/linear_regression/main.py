import mxnet as mx
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = mx.cpu() if len(mx.test_utils.list_gpus()) == 0 else mx.gpu()
print ('current ctx =>', device)


# Hyper-parameters
output_size = 1
num_epochs = 100
learning_rate = 0.01
batch_size = 3

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


# Linear regression model
model = nn.Dense(output_size)
model.initialize(ctx=device)
model.hybridize()


# Loss and optimizer 
criterion = gluon.loss.L2Loss()
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': learning_rate})

# Train the model
for epoch in range(num_epochs):
    # Convert numpy to mxnet.ndarry
    inputs = nd.array(x_train).as_in_context(device)
    targets = nd.array(y_train).as_in_context(device)

    # Forward pass
    with autograd.record():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # Backward and optimize
    loss.backward()
    optimizer.step(inputs.size)

    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, loss.mean().asscalar()))


# Plot
predicted = model(nd.array(x_train)).asnumpy()
plt.plot(x_train, y_train, 'ro', label = 'Original data')
plt.plot(x_train, predicted, label = 'Fitted line')
plt.legend()
plt.show()

# Save the params
model.save_parameters('linear_regression.params')
