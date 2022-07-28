# CNN-Breaking-CAPTCHA-Security-Codes
> CNNs are a type of neural network that allow greater extraction of features from captured images. Unlike classical models, CNNs take image data, train the model, and then classify the features automatically for healthier classification. The image data was fit to the CNN model with 80% in the training set and 20% in the testing set. Since the technology that was used to fit the image data to the CNN did not allow for the label to be a string, the 32 individual characters strings of the CAPTCHA images were converted to their corresponding ASCII codes, as illustrated above

## Data
The images data used for this model went through some image processing and was flattened to a csv
> See imageProcessing.py in this repo to see how that was done

## Read in Images DataFrame

``` {.python}
imagesDF = pd.read_csv ("imagesDF.csv")
```

## Converting image labels to Unicode characters
``` {.python}
length = len(uniqueLabels)
for x in range(length):
    label = uniqueLabels[x]
    imagesDF.replace(to_replace={'labels': label} , value=ord(label), inplace = True)
```

## Pandas Dfs to NumPy Arrays
``` {.python}
imagesArray = imagesDF.to_numpy()
labelsArray = labelsDF.to_numpy()
```
## Set Random Seed
``` {.python}
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

```


# Model Building
## Define Neural NetworK

``` {.python}
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
```

## Train Test Split 

``` {.python}
(X_train, X_test, y_train, y_test) = train_test_split(
   imagesArray, labelsArray, test_size=0.2, random_state=11
)
```

### Validation Splits 
``` {.python}
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
```

# Fit Model

``` {.python}

```

``` {.python}
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

```

## Results
|Batch| Batch Accuracy| Test Accuracy|
|---|---|---|
|0 | 0.790099 | 0.79373664|
|1| 0.9643564| 0.96038234|
|2 | 0.980198 | 0.9771098|
|3 | 0.97821784 | 0.9800025|
|4 | 0.9841584| 0.98063135|
|5|0.990099|0.9825179|
|6 | 0.9841584| 0.9835241|
|7 | 0.9940594 | 0.9837756|
|8 | 0.9920792 | 0.9839014|
|9 | 0.98613864 |0.9837756|

> * As the batch increases, the training set and test set accuracy are increasing as well. The test set accuracy score converges to 98.378%, which means that the CNN model is predicting the individual character ASCII codes of the CAPTCHA images better than the random forest model, at least a percentage or less better
> * The creation of a dataset of characters separated from CAPTCHA images allowed for the fitting of that image data to a convolutional neural network. The purpose of predicting the characters in a CAPTCHA was to give bots the ability to bypass security systems which use CAPTCHA to deny bots entry to their websites. The model used in this research predicted the characters in CAPTCHA images with 98.378% accuracy. A bot equipped with this model will be able to bypass anti-bot security systems equipped with captcha 98% of the time, by simply entering the characters it predicts to be in the image and ticking the “I am not a robot” box.







