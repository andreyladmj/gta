import tensorflow as tf

net_params = {}

def neural_net_image_input(image_shape, name='x'):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    n_input_1 = image_shape[0]
    n_input_2 = image_shape[1]
    n_input_3 = image_shape[2]
    return tf.placeholder(tf.float32,[None, n_input_1, n_input_2, n_input_3], name=name)


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, [None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, None, name='keep_prob')


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """

    #layer = conv2d_maxpool(x, 16, (4,4), (1,1), (2,2), (2,2))
    layer = create_convolution_layers(x)
    tf.nn.dropout(layer, keep_prob=keep_prob)

    #layer = flatten(layer)
    layer = tf.contrib.layers.flatten(layer)
    #layer = fully_conn(layer, 400)
    layer = tf.contrib.layers.fully_connected(layer, 2000)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 1000)
    layer = tf.nn.dropout(layer, keep_prob)

    res = tf.contrib.layers.fully_connected(layer, 4, activation_fn=None)

    return res


def create_convolution_layers(X):
    #Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')

    #alex net
    # [227x227x3] INPUT
    # [55x55x96] CONV1: 96 11x11 filters at stride 4, pad 0
    # [27x27x96] MAX POOL1: 3x3 filters at stride 2
    # [27x27x96] NORM1: Normalization layer
    # [27x27x256] CONV2: 256 5x5 filters at stride 1, pad 2
    # [13x13x256] MAX POOL2: 3x3 filters at stride 2
    # [13x13x256] NORM2: Normalization layer
    # [13x13x384] CONV3: 384 3x3 filters at stride 1, pad 1
    # [13x13x384] CONV4: 384 3x3 filters at stride 1, pad 1
    # [13x13x256] CONV5: 256 3x3 filters at stride 1, pad 1
    # [6x6x256] MAX POOL3: 3x3 filters at stride 2
    # [4096] FC6: 4096 neurons
    # [4096] FC7: 4096 neurons
    # [1000] FC8: 1000 neurons (class scores)

    #252 x 189
    #nn = add_conv_relu_maxPool()
    nn = create_conv2d(X, 128, strides=[8,8], w_name='W1')
    nn = tf.nn.relu(nn, name='W1_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    nn = create_conv2d(nn, 256, strides=[4,4], w_name='W2')
    nn = tf.nn.relu(nn, name='W2_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = create_conv2d(nn, 256, strides=[3,3], w_name='W3')
    nn = tf.nn.relu(nn, name='W3_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = create_conv2d(nn, 512, strides=[3,3], w_name='W4')
    nn = tf.nn.relu(nn, name='W4_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = create_conv2d(nn, 512, strides=[3,3], w_name='W5')
    nn = tf.nn.relu(nn, name='W5_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    return nn

def add_conv_relu_maxPool(cnn, filters, strides, name):
    # nn = create_conv2d(nn, 512, strides=[3,3], w_name='W5')
    # nn = tf.nn.relu(nn)
    # nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    cnn = create_conv2d(cnn, filters, strides=strides, w_name=name)
    cnn = tf.nn.relu(cnn)
    return tf.nn.max_pool(cnn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

def create_convolution_layers_ORIGINAL(X):
    #Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')

    Z1 = create_conv2d(X, 32, strides=[8,8], w_name='W1')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    Z2 = create_conv2d(P1, 64, strides=[4,4], w_name='W2')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    Z3 = create_conv2d(P2, 128, strides=[2,2], w_name='W3')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    return P3

def get_weights_shape(X, conv_num_outputs, strides):
    depth = X.get_shape().as_list()[-1]
    w_size = [strides[0], strides[1], depth, conv_num_outputs]
    c_strides = [1, strides[0], strides[1], 1]
    return w_size, c_strides

def create_conv2d(X, conv_num_outputs, strides, w_name):
    depth = X.get_shape().as_list()[-1]
    w_size = [strides[0], strides[1], depth, conv_num_outputs]
    c_strides = [1, strides[0], strides[1], 1]
    W = tf.get_variable(w_name, w_size, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Z = tf.nn.conv2d(X, W, strides=c_strides, padding='SAME', name=w_name+'_conv2d')
    return Z

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """

    depth = x_tensor.get_shape().as_list()[-1]
    bias = tf.Variable(tf.zeros(conv_num_outputs))

    c_strides = [1, conv_strides[0], conv_strides[1], 1]
    p_ksize = [1, pool_ksize[0], pool_ksize[1], 1]
    p_strides = [1, pool_strides[0], pool_strides[1], 1]

    # 2x2x5x10
    weight= tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], depth, conv_num_outputs]))


    conv = tf.nn.conv2d(x_tensor, weight, c_strides, 'SAME') + bias
    conv = tf.nn.relu(conv)

    pool = tf.nn.max_pool(conv, p_ksize, p_strides, 'SAME')

    return pool

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    b, w, h, d = x_tensor.get_shape().as_list()
    img_size = w * h * d
    return tf.reshape(x_tensor, [-1, img_size])

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs]))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), bias))

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul(x_tensor, weight), bias)

def get_bias(n):
    return tf.Variable(tf.zeros(n))