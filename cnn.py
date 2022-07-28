# Common imports
import numpy as np
import os


# to make this notebook's output stable across runs
np.random.seed(42)
import pandas as pd
# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

imagesDF = pd.read_csv ("imagesDF.csv")

uniqueLabels = imagesDF['labels'].unique()

length = len(uniqueLabels)
for x in range(length):
    label = uniqueLabels[x]
    imagesDF.replace(to_replace={'labels': label} , value=ord(label), inplace = True)

#Separate labels from the Df 
labelsDF = imagesDF['labels']

#Drop irrelevant column and label column
imagesDF.drop(columns =["labels"], inplace = True)
imagesDF.drop(columns =["Unnamed: 0"], inplace = True)

# Dfs to Arrays
imagesArray = imagesDF.to_numpy()
labelsArray = labelsDF.to_numpy()

# Import TensorFlow
import tensorflow as tf
tf.__version__

# Set Random Seeds
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Build Neural Network
height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 91

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int64, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
 # Train Test Splits
from sklearn.model_selection import train_test_split

(X_train, X_test, y_train, y_test) = train_test_split(
   imagesArray, labelsArray, test_size=0.2, random_state=11
)

# Validation Splits
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# View Training Set and Test Set
print (np.unique (y_train, return_counts=True))
print (np.unique (y_test, return_counts=True))
np.__version__

# Define a shuffling function
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
 
# Fit the CNN model
n_epochs = 10
batch_size = 500

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        
        # Dr. Phil: This inner loop, with shuffle_batch achieves one epoch, 
        # looping over randomly selected batches
        
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            
             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)

# Results Interpretation
# As the batch increases, the training set and test set accuracy are increasing as well. The test set accuracy score converges to 98.378%, 
# which means that the CNN model is predicting the individual character ASCII codes of the CAPTCHA images better than the random forest model, 
# at least a percentage or less better













