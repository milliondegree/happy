import tensorflow as tf
import numpy as np

weight_path = '/nfs/project/pre_param/vgg16_weights.npz'
weight_file = np.load(weight_path)

def bottle_layer(parent, channel_in, channel_1, channel_2, is_training, name, stride = 1, if_reuse = False):
	with tf.variable_scope(name):
		conv_1 = conv2d_layer(parent, [1, 1, channel_in, channel_1], [1, stride, stride, 1], 'conv_1', if_reuse = if_reuse)
		conv_1_bn = tf.nn.relu(bn_layer(conv_1, is_training, 'bn_1'))
		conv_2 = conv2d_layer(conv_1_bn, [3, 3, channel_1, channel_1], [1, 1, 1, 1], 'conv_2', if_reuse = if_reuse)
		conv_2_bn = tf.nn.relu(bn_layer(conv_2, is_training, 'bn_2'))
		conv_3 = conv2d_layer(conv_2_bn, [1, 1, channel_1, channel_2], [1, 1, 1, 1], 'conv_3', if_reuse = if_reuse)
		conv_3_bn = bn_layer(conv_3, is_training, 'bn_3')

		if channel_in != channel_2:
			parent = conv2d_layer(parent, [1, 1, channel_in, channel_2], [1, stride, stride, 1], 'short_cut', if_reuse = if_reuse)
			parent = bn_layer(parent, is_training, 'bn_parent')
		
		return tf.nn.relu(conv_3_bn + parent)

def bottle_atrous_layer(parent, channel_in, channel_1, channel_2, dilation, is_training, name, stride = 1, if_reuse = False):
	with tf.variable_scope(name):
		conv_1 = conv2d_layer(parent, [1, 1, channel_in, channel_1], [1, stride, stride, 1], 'conv_1', if_reuse = if_reuse)
		conv_1_bn = tf.nn.relu(bn_layer(conv_1, is_training, 'bn_1'))
		conv_2 = atrous_conv_layer(conv_1_bn, [3, 3, channel_1, channel_1], dilation, 'atrous_2', if_reuse = if_reuse)
		conv_2_bn = tf.nn.relu(bn_layer(conv_2, is_training, 'bn_2'))
		conv_3 = conv2d_layer(conv_2_bn, [1, 1, channel_1, channel_2], [1, 1, 1, 1], 'conv_3', if_reuse = if_reuse)
		conv_3_bn = bn_layer(conv_3, is_training, 'bn_3')

		if channel_in != channel_2:
			parent = conv2d_layer(parent, [1, 1, channel_in, channel_2], [1, stride, stride, 1], 'short_cut', if_reuse = if_reuse)
			parent = bn_layer(parent, is_training, 'bn_parent')
		
		return tf.nn.relu(conv_3_bn + parent)


def stack_layer(parent, channel_in, channel_1, channel_2, is_training, name):

    with tf.variable_scope(name):

        conv_1 = conv2d_layer(parent, [3, 3, channel_in, channel_1], [1, 1, 1, 1], 'conv_1', if_reuse = False)
        bn_1 = bn_layer(conv_1, is_training, 'bn_1')
        conv_1_relu = tf.nn.relu(bn_1, name = 'relu_1')
        conv_2 = conv2d_layer(conv_1_relu, [3, 3, channel_1, channel_2], [1, 1, 1, 1], 'conv_2', if_reuse = False)
        bn_2 = bn_layer(conv_2, is_training, 'bn_2')

        if channel_in != channel_2:
            shortcut = conv2d_layer(parent, [1, 1, channel_in, channel_2], [1, 1, 1, 1], 'shortcut', if_reuse = False)
        else:
            shortcut = parent

    return tf.nn.relu(bn_2 + shortcut)




def conv2d_layer(parent, kernal_size, stride, name, if_bias = True, if_relu = False, if_reuse = True):
    '''conv2d with xavier initializer'''
    with tf.variable_scope(name):
        if if_reuse:
            init_w = tf.constant_initializer(weight_file[name+'_W'])
            init_b = tf.constant_initializer(weight_file[name+'_b'])
        else:
            init_w = init_b = tf.contrib.layers.xavier_initializer_conv2d(dtype = tf.float32)
        
        weights = tf.get_variable(name = 'weights', shape = kernal_size, dtype = 'float32', initializer = init_w)
        conv = tf.nn.conv2d(parent, weights, stride, padding = 'SAME')
    
        if if_bias:
            bias = tf.get_variable(name = 'bias', shape = [kernal_size[-1]], dtype = 'float32', initializer = init_b)
            conv_with_bias = tf.nn.bias_add(conv, bias)
        else:
            conv_with_bias = conv 
    
        if if_relu:
            return tf.nn.relu(conv_with_bias)
        else:
            return conv_with_bias
        

def atrous_conv_layer(parent, kernal_size, rate, name, if_bias = True, if_relu = False, if_reuse = True):
    '''
    Implementation of atrous convolutional layer
    kernal_size = [H, W, in_C, out_C]
    '''
    with tf.variable_scope(name):
        if if_reuse:
            init_w = tf.constant_initializer(weight_file[name+'_W'])
            init_b = tf.constant_initializer(weight_file[name+'_b'])
        else:
            init_w = init_b = tf.contrib.layers.xavier_initializer_conv2d(dtype = tf.float32)

        weights = tf.get_variable(name = 'weights', shape = kernal_size, dtype = 'float32', initializer = init_w)
        atrous_conv = tf.nn.atrous_conv2d(parent, weights, rate = rate, padding = 'SAME')

        if if_bias:
            bias = tf.get_variable(name = 'bias', shape = [kernal_size[-1]], dtype = 'float32', initializer = init_b)
            conv_with_bias = tf.nn.bias_add(atrous_conv, bias)
        else:
            conv_with_bias = atrous_conv 

        if if_relu:
            return tf.nn.relu(conv_with_bias)
        else:
            return conv_with_bias

def fc_layer(parent, output_channal, name, if_bias = True, if_relu = False):
    with tf.variable_scope(name):
        N, W = parent.shape.as_list()
        init = tf.contrib.layers.xavier_initializer_conv2d(dtype = tf.float32)
        weights = tf.get_variable(name = 'weights', shape = [W, output_channal], dtype = 'float32', initializer = init)
        fc = tf.matmul(parent, weights)

        if if_bias:
            bias = tf.get_variable(name = 'bias', shape = [output_channal], dtype = 'float32', initializer = init)
            fc_with_bias = tf.nn.bias_add(fc, bias)
        else:
            fc_with_bias = fc

        if if_relu:
            return tf.nn.relu(fc_with_bias)
        else:
            return fc_with_bias

def bn_layer(parent, is_training, name):

    with tf.variable_scope(name):
        shape = parent.shape
        param_shape = shape[-1:]

        pop_mean = tf.get_variable("mean", param_shape, initializer = tf.constant_initializer(0.0), trainable=False)
        pop_var = tf.get_variable("variance", param_shape, initializer = tf.constant_initializer(1.0), trainable=False)
        epsilon = 1e-4
        decay = 0.99

        scale = tf.get_variable('scale', param_shape, initializer = tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', param_shape, initializer = tf.constant_initializer(0.0))

        def True_fn():
            batch_mean, batch_var = tf.nn.moments(parent, list(range((len(shape) - 1))))

            train_mean = tf.assign(pop_mean,
                           pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                          pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                output = tf.nn.batch_normalization(parent,
                batch_mean, batch_var, beta, scale, epsilon, name = name)
                return output

        def False_fn():
            output = tf.nn.batch_normalization(parent,
            pop_mean, pop_var, beta , scale, epsilon, name = name)
            return output

    return tf.cond(is_training, True_fn, False_fn)

def upsample_layer(bottom, shape, n_channels, upscale_factor, num_classes, name):
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        output_shape = [shape[0], shape[1], shape[2], num_classes]
        filter_shape = [kernel_size, kernel_size, num_classes, n_channels]
        weights = _get_bilinear_filter(filter_shape, upscale_factor)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        bias_init = tf.constant(0.0, shape=[num_classes])
        bias = tf.get_variable('bias', initializer=bias_init)
        dconv_with_bias = tf.nn.bias_add(deconv, bias)
    return dconv_with_bias
    

def SCNN_DURL(parent, w, name):
    init = tf.contrib.layers.xavier_initializer(dtype = tf.float32)
    N, H, W, C = parent.shape.as_list()
    
    # downward and upward  operation
    d_weights = tf.get_variable('downKernel', [w, C, C], initializer = init)
    u_weights = tf.get_variable('upKernel', [w, C, C], initializer = init)
    tlist = tf.split(parent, H, axis = 1)
    for i in xrange(H):
        tlist[i] = tf.reshape(tlist[i], [N, W, C])
        if i != 0:
            tlist[i] += tf.nn.relu(tf.nn.conv1d(tlist[i-1], d_weights, 1, 'SAME'))
    for i in reversed(xrange(H)):
        if i != H-1:
            tlist[i] += tf.nn.relu(tf.nn.conv1d(tlist[i+1], u_weights, 1, 'SAME'))
    current = tf.stack(tlist, axis = 1)

    #rightward and leftward operation
    r_weights = tf.get_variable('rightKernel', [w, C, C], initializer = init)
    l_weights = tf.get_variable('leftKernel', [w, C, C], initializer = init)
    tlist = tf.split(current, W, axis = 2)
    for i in xrange(W):
        tlist[i] = tf.reshape(tlist[i], [N, H, C])
        if i != 0:
            tlist[i] += tf.nn.relu(tf.nn.conv1d(tlist[i-1], r_weights, 1, 'SAME'))
    for i in reversed(xrange(H)):
        if i != W-1:
            tlist[i] += tf.nn.relu(tf.nn.conv1d(tlist[i+1], l_weights, 1, 'SAME'))
    current = tf.stack(tlist, axis = 2)
    return current

def _get_bilinear_filter(filter_shape, upscale_factor):
    kernel_size = filter_shape[1]
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
                1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        for j in range(filter_shape[3]):
            weights[:, :, i, j] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights

