import os
import gzip
import numpy as np
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def load_mnist(path, kind = 'train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
        
    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype = np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype = np.uint8).reshape(len(labels), 28, 28).astype(np.float64)
 
    return images, labels

def make_reshape(data):
    """Flatten data"""
    return data.reshape(data.shape[0], data.shape[1] * data.shape[2])

""" ----------------------------- Histogram ----------------------------- """

def make_histogram(x):
    """Histogram of x"""
    count = [0] * 256
    for i in range(len(x)):
        count[int(x[i])] += 1
    return count

def data_make_histogram(data):
    result = np.zeros( (data.shape[0], 256))
    for i in range(data.shape[0]):
        result[i] = make_histogram(data[i])
    return result

""" ----------------------------- Down Sample ----------------------------- """

def max_value(arr):
    return round(arr.max())
def min_value(arr):
    return round(arr.min())
def avg_value(arr):
    return round(arr.mean())

def down_sampling(arr, n, mode):
    result = np.zeros( (n, n) )
    len = arr.shape[0] // n
    for i in range(n):
        for j in range(n):
            sub_arr = arr[i * len : (i + 1) * len, j * len : (j + 1) * len]
            value = 0
            if (mode == -1):
                value = min_value(sub_arr)
            elif (mode == 1):
                value = max_value(sub_arr)
            else:
                value = avg_value(sub_arr)
            result[i][j] = value
    return result

def data_down_sampling(data, n, mode):
    """ Downsampling data to nxn"""
    """ mode = -1: min, mode = 1: max, mode = 0: average"""
    result = np.zeros( (data.shape[0], n, n) )
    for i in range(data.shape[0]):
        result[i] = down_sampling(data[i], n, mode)
        if ((i + 1) % 5000 == 0):
            print("Downsampling " + str(n) + 'x' + str(n) + ': ', i + 1, '/', data.shape[0])
    return result

""" ------------------------ Load and Show Data ------------------------ """

X_train, y_train = load_mnist('data/', kind = 'train')
X_test, y_test = load_mnist('data/', kind = 't10k')

x_train = make_reshape(X_train) #flatten training data
x_test = make_reshape(X_test)   #flatten testing data

print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

fig, ax = plt.subplots(nrows = 4, ncols = 5, sharex = True, sharey = True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0]
    ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
for i in range(10, 20):
    img = X_test[i]
    ax[i].imshow(img, cmap = 'Blues')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.show()
def Show(img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap = 'gray')
    plt.show()

""" --- TEST feature extraction ---"""

""" histogram """
train_hist = data_make_histogram(x_train)
test_hist = data_make_histogram(x_test)
"""downsampling training data"""
#train_min_4 = data_down_sampling(X_train, 4, -1)
#train_max_4 = data_down_sampling(X_train, 4, 1)
train_avg_4 = data_down_sampling(X_train, 4, 0)

#train_min_7 = data_down_sampling(X_train, 7, -1)
#train_max_7 = data_down_sampling(X_train, 7, 1)
train_avg_7 = data_down_sampling(X_train, 7, 0)
#
"""flatten downsapling training data"""
#flat_train_min_4 = make_reshape(temp_min_4)
#flat_train_max_4 = make_reshape(temp_max_4)
flat_train_avg_4 = make_reshape(train_avg_4)

#flat_train_min_7 = make_reshape(temp_min_7)
#flat_train_max_7 = make_reshape(temp_max_7)
flat_train_avg_7 = make_reshape(train_avg_7)

"""downsampling and flatten test data"""
test_avg_4 = data_down_sampling(X_test, 4, 0)
test_avg_7 = data_down_sampling(X_test, 7, 0)
flat_test_avg_4 = make_reshape(test_avg_4)
flat_test_avg_7 = make_reshape(test_avg_7)

""" ------------------------ KNN Classifier ------------------------ """

def find_distance(pattern, img_train):
    result = np.zeros(60000)
    for i in range(60000):
        result[i] = np.linalg.norm(img_train[i] - pattern)
    return result

def find_k_nearest(arr, K):
    return arr.argsort()[:K]

def KNN(pattern, img_train, K = 179):
    lst = find_k_nearest(find_distance(pattern, img_train), K)
    min_label = []
    for i in range(K):
        min_label.append(y_train[lst[i]])
    result = Counter(min_label).most_common(1)
    prediction, voting = result[0]
    print('Predict: ', prediction)
    print('Accuracy of %dNN with major voting: %.2f' % (K, 100 * voting / K), '%')
    return prediction

def KNN_testing(img_train, img_test, K=179):
    print("KNN Classifier")
    correct = 0
    for i in range(img_test.shape[0]):
        print('Test', i, '/ 10000 :')
        pred = KNN(img_test[i], img_train, K)
        print("Answer: ", y_test[i])
        if (pred == y_test[i]):
            print("Correct!")
            correct += 1
        else:
            print("Incorrect!")

    print("Accuracy: ", correct / img_test.shape[0] * 100, '%')

#KNN_testing(flat_train_avg_4, flat_test_avg_4, 300)

""" ------------------------ Average-vector Classifier ------------------------ """

def avg_vector(data_img, data_lab, num_labels = 10):
    result = np.zeros((num_labels, data_img.shape[1]))
    for i in range(num_labels):
        result[i] = (np.mean(data_img[data_lab == i], axis = 0))
    return result

def MTB_classifier(pattern, MTB):
    dist = np.zeros(10)
    for i in range(10):
        dist[i] = np.linalg.norm(MTB[i] - pattern)
    return dist.argmin()

def MTB_testing():
    print('Average model classifier')
    correct = 0
    MTB = avg_vector(x_train, y_train)
    for i in range(x_test.shape[0]):
        pred = MTB_classifier(x_test[i], MTB)
        print('Prediction:', pred)
        print('Answer:', y_test[i])
        if (pred == y_test[i]):
            print('Correct!')
            correct += 1
        else:
            print('Incorrect!')
    print("Accuracy: ", correct / 10000 * 100)

#MTB_testing()

""" ------------------------ Artificial Neural Network ------------------------ """
print("Artificial Neural Network")

def train_neural_network(img_train, type):
    #img_test = img_test / 255
    img_train = img_train / 255
    #img_test = img_test.reshape(img_test.shape + (1, ))
    img_train = img_train.reshape(img_train.shape + (1, ))
    #build neural network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (img_train.shape[1:])))  # input-layer
    model.add(tf.keras.layers.Dense(256, activation = 'relu'))  # hidden-layer with 256 neural
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))  # output-layer
    #training neural network
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(img_train, y_train, epochs = 10, verbose = 2)
    #val_loss, val_acc = model.evaluate(img_test, y_test)
    model.save(type + '_model.h5')
    del model
    return val_loss, val_acc

def ANN_testing(img_test, type):
    model = load_model('ANN_model/' + type + '_model.h5')
    if (type == 'histogram'):
        tf.keras.utils.normalize(img_test)
    else:
        img_test = img_test.reshape(img_test.shape + (1, )) / 255
    val_loss, val_acc = model.evaluate(img_test, y_test)
    print('Accuracy percentage of ANN on', end = ' ')
    print(type + ': %.2f %%' % (100 * val_acc))
    predictions = model.predict(img_test)
    print('Predict:', np.argmax(predictions[:20], axis = 1))
    print("Answer: ", y_test[:20])
    print('Rating:', np.max(predictions[:20], axis=1), end='\n')
    del model

ANN_testing(X_test, 'flatten')
# ANN_testing(test_hist, 'histogram')
# ANN_testing(test_avg_4, 'downsample4')
# ANN_testing(test_avg_7, 'downsample7')


