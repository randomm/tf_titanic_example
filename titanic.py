import numpy as np
import tflearn
from lib.helpers import preprocess

# download titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# load csv dataset
from tflearn.data_utils import load_csv

# first row indicates data labels
data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)

# ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# preprocess data
data = preprocess(data, to_ignore)

# build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# define model
model = tflearn.DNN(net)

# training (apply adam algorithm by default)
model.fit(data, labels, n_epoch=20, batch_size=16, show_metric=True)

# training using a validation set
#model.fit(data, labels, n_epoch=20, batch_size=16, validation_set=0.15, show_metric=True)

# let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]

# preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)

# predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])
