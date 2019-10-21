from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import rnn
import config_autovc as config


tf.logging.set_verbosity(tf.logging.INFO)




def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    # try:
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1], name = name)

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

  return deconv

def selu(x):
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def bi_static_stacked_RNN(x, scope='RNN', lstm_size = config.lstm_size):
    """
    Input and output in batch major format
    """
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        output = x
        num_layer = 2
        # for n in range(num_layer):
        lstm_fw = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)

        _initial_state_fw = lstm_fw.zero_state(config.batch_size, tf.float32)
        _initial_state_bw = lstm_bw.zero_state(config.batch_size, tf.float32)

        output, _state1, _state2 = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw, lstm_bw, output, 
                                                  initial_state_fw=_initial_state_fw,
                                                  initial_state_bw=_initial_state_bw, 
                                                  scope='BLSTM_'+scope)
        output = tf.stack(output)
        output_fw = output[0]
        output_bw = output[1]
        output = tf.transpose(output, [1,0,2])


        # output = tf.layers.dense(output, config.output_features, activation=tf.nn.relu) # Remove this to use cbhg

        return output




def bi_dynamic_RNN(x, input_lengths, scope='RNN'):
    """
    Stacked dynamic RNN, does not need unpacking, but needs input_lengths to be specified
    """

    with tf.variable_scope(scope):

        cell = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)

        outputs, states  = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            dtype=tf.float32,
            sequence_length=input_lengths,
            inputs=x)

        outputs = tf.concat(outputs, axis=2)

    return outputs


def RNN(x, scope='RNN'):
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=config.out_lstm)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        outputs=tf.stack(outputs)
        outputs = tf.transpose(outputs, [1,0,2])

    return outputs



def nr_wavenet_block(conditioning, is_train, dilation_rate = 2):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])

    con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
    con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
    con_sig_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = "wave_1_{}".format(dilation_rate)), training = is_train, name = "wave_1_BN_{}".format(dilation_rate))
    con_sig_backward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = "wave_2_{}".format(dilation_rate)), training = is_train, name = "wave_2_BN_{}".format(dilation_rate))
    # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

    sig = tf.sigmoid(con_sig_forward+con_sig_backward)


    con_tanh_forward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = "wave_3_{}".format(dilation_rate)), training = is_train, name = "wave_3_BN_{}".format(dilation_rate))
    con_tanh_backward = tf.layers.batch_normalization(tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = "wave_4_{}".format(dilation_rate)), training = is_train, name = "wave_4_BN_{}".format(dilation_rate))
    # con_tanh = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

    tanh = tf.tanh(con_tanh_forward+con_tanh_backward)


    outputs = tf.multiply(sig,tanh)

    skip = tf.layers.batch_normalization(tf.layers.conv1d(outputs,config.wavenet_filters,1, name = "wave_5_{}".format(dilation_rate)), training = is_train, name = "wave_5_BN_{}".format(dilation_rate))

    residual = skip + conditioning

    return skip, residual


def nr_wavenet(inputs, is_train, num_block = config.wavenet_layers):


    receptive_field = 2**num_block

    first_conv = tf.layers.batch_normalization(tf.layers.conv1d(inputs, config.wavenet_filters, 1, name = "wave_pre"), training = is_train, name = "wave_pre_BN")
    skips = []
    skip, residual = nr_wavenet_block(first_conv, is_train, dilation_rate=1)
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, is_train, dilation_rate=2**(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    harm = tf.layers.batch_normalization(tf.layers.conv1d(output,config.wavenet_filters,1, name = "wave_harm_1"), training = is_train, name = "wave_harm_1_BN")

    ap = tf.layers.batch_normalization(tf.layers.conv1d(output,config.wavenet_filters,1, name = "wave_ap_1"), training = is_train, name = "wave_ap_1_BN")

    harm = tf.layers.dense(harm, 60, activation=None, name = "wave_harm_2")

    ap = tf.layers.dense(ap, 4, activation=None, name = "wave_ap_2")

        
    return harm, ap



def nr_wavenet_f0(inputs, harm, ap, is_train, num_block = config.wavenet_layers):


    receptive_field = 2**num_block

    inputs = tf.concat([inputs, harm, ap], axis=-1)

    first_conv = tf.layers.batch_normalization(tf.layers.conv1d(inputs, config.wavenet_filters, 1, name = "wave_pre"), training = is_train, name = "wave_pre_BN")
    skips = []
    skip, residual = nr_wavenet_block(first_conv, is_train, dilation_rate=1)
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, is_train, dilation_rate=2**(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    f0 = tf.layers.batch_normalization(tf.layers.conv1d(output,config.wavenet_filters,1, name = "wave_f0_1"), training = is_train, name = "wave_f0_1_BN")

    if config.f0_mode == "cont":
        f0 = tf.layers.dense(f0, 1, activation=None, name = "wave_f0_2")
    elif config.f0_mode == 'discrete':
        f0 = tf.layers.dense(f0, config.cqt_bins, activation=None, name = "wave_f0_2")

    return f0
    
def nr_wavenet_vuv(inputs, harm, ap, f0, is_train, num_block = config.wavenet_layers):


    receptive_field = 2**num_block

    inputs = tf.concat([inputs, harm, ap, f0], axis=-1)

    first_conv = tf.layers.batch_normalization(tf.layers.conv1d(inputs, config.wavenet_filters, 1, name = "wave_pre"), training = is_train, name = "wave_pre_BN")
    skips = []
    skip, residual = nr_wavenet_block(first_conv, is_train, dilation_rate=1)
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, is_train, dilation_rate=2**(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    vuv = tf.layers.batch_normalization(tf.layers.conv1d(output,config.wavenet_filters,1, name = "wave_vuv_1"), training = is_train, name = "wave_vuv_1_BN")

    vuv = tf.layers.dense(vuv, 1, activation=tf.nn.sigmoid, name = "wave_vuv_2")

    return vuv

def encoder_conv_block(inputs, layer_num, is_train, num_filters = config.filters):

    output = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(inputs, num_filters * 2**int(layer_num/2), (config.filter_len,1)
        , strides=(2,1),  padding = 'same', name = "G_"+str(layer_num))), training = is_train, name = "GBN_"+str(layer_num))
    return output

def decoder_conv_block(inputs, layer, layer_num, is_train, num_filters = config.filters):

    deconv = tf.image.resize_images(inputs, size=(int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # embedding = tf.tile(embedding,[1,int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1,1])

    deconv = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(deconv, layer.shape[-1]
        , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "D_"+str(layer_num))), training = is_train, name =  "DBN_"+str(layer_num))

    # embedding =tf.nn.relu(tf.layers.conv2d(embedding, layer.shape[-1]
    #     , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "DEnc_"+str(layer_num)))

    deconv =  tf.concat([deconv, layer], axis = -1)

    return deconv

def encoder_decoder_archi(inputs, is_train):
    """
    Input is assumed to be a 4-D Tensor, with [batch_size, phrase_len, 1, features]
    """

    encoder_layers = []

    encoded = inputs

    encoder_layers.append(encoded)

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block(encoded, i, is_train)
        encoder_layers.append(encoded)
    
    encoder_layers.reverse()



    decoded = encoder_layers[0]

    for i in range(config.encoder_layers):
        decoded = decoder_conv_block(decoded, encoder_layers[i+1], i, is_train)

    return decoded

def enc_dec_f0(inputs, is_train):

    # inputs = tf.concat([inputs, harm, ap], axis=-1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
        , name = "S_in"), training = is_train,name = 'S_in_BN')


    output = encoder_decoder_archi(inputs, is_train)

    if config.f0_mode == "cont":
        f0 = tf.layers.dense(output, 1, activation=None, name = "wave_f0_2")
        return tf.reshape(f0, [config.batch_size, config.max_phr_len , -1])
    elif config.f0_mode == 'discrete':
        f0 = tf.layers.dense(output, config.cqt_bins, activation=None, name = "wave_f0_2")
        return tf.squeeze(f0)

    

def enc_dec_vuv(inputs,f0,ap, is_train):

    inputs = tf.concat([inputs, f0, ap], axis=-1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
        , name = "S_in"), training = is_train,name = 'S_in_BN')


    output = encoder_decoder_archi(inputs, is_train)

    vuv = tf.layers.dense(output, 1, activation=tf.nn.sigmoid, name = "wave_vuv_2")

    return tf.reshape(vuv, [config.batch_size, config.max_phr_len , -1])


def content_encoder(inputs, singer_label, is_train):

    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    inputs = tf.concat([inputs, singer_label], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])


    conv_1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs, 512, (5,1), name = "conv_1",padding='same'), training = is_train, name = "conv_1_BN"))

    conv_2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_1, 512, (5,1), name = "conv_2",padding='same'), training = is_train, name = "conv_2_BN"))

    conv_3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_2, 512, (5,1), name = "conv_3",padding='same'), training = is_train, name = "conv_3_BN"))

    conv_3 = tf.reshape(conv_3,[config.batch_size, config.max_phr_len , -1] )

    lstm_op = bi_static_stacked_RNN(conv_3, scope = "Encode")

    lstm_fow = lstm_op[:,:, :config.lstm_size]

    lstm_back = lstm_op[:,:, config.lstm_size:]

    emb = []

    for i in range(int(config.max_phr_len/config.code_sam)):
        emb.append(tf.concat([lstm_fow[:, i*config.code_sam,:], lstm_back[:, (i+1)*config.code_sam-1, :]], axis = -1))

    emb = tf.stack(emb)

    return emb


def decoder(emb, singer_label, is_train):


    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    embo = tf.tile(tf.reshape(emb[0],[config.batch_size,1,-1]),[1,config.code_sam,1])

    for i in range(1, int(config.max_phr_len/config.code_sam)):
        embs = tf.tile(tf.reshape(emb[i],[config.batch_size,1,-1]),[1,config.code_sam,1])

        embo = tf.concat([embo, embs], axis = 1)

    # import pdb;pdb.set_trace()

    inputs = tf.concat([embo, singer_label], axis = -1)

    

    lstm_op_1 = RNN(inputs, scope = "Decode_1")

    lstm_op_1 = tf.reshape(lstm_op_1, [config.batch_size, config.max_phr_len , 1, -1])

    conv_1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(lstm_op_1, 512, (5,1), name = "op_conv_1",padding='same'), training = is_train, name = "op_conv_1_BN"))

    conv_2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_1, 512, (5,1), name = "op_conv_2",padding='same'), training = is_train, name = "op_conv_2_BN"))

    conv_3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_2, 512, (5,1), name = "op_conv_3",padding='same'), training = is_train, name = "op_conv_3_BN"))

    conv_3 = tf.reshape(conv_3,[config.batch_size, config.max_phr_len , -1] )

    lstm_op_2 = RNN(conv_3, scope = "Decode_2")

    lstm_op_3 = RNN(lstm_op_2, scope = "Decode_3")

    # lstm_op_3 = bi_static_stacked_RNN(lstm_op_2, lstm_size = config.out_lstm, scope = "Decode_3")

    output = tf.layers.conv1d(lstm_op_3, config.output_features, 1, name = "decode_output")


    return output


def post_net(inputs, is_train):

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    conv_1 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(inputs, 512, (5,1), name = "post_conv_1",padding='same'), training = is_train, name = "post_conv_1_BN"))

    conv_2 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(conv_1, 512, (5,1), name = "post_conv_2",padding='same'), training = is_train, name = "post_conv_2_BN"))

    conv_3 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(conv_2, 512, (5,1), name = "post_conv_3",padding='same'), training = is_train, name = "post_conv_3_BN"))   

    conv_4 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(conv_3, 512, (5,1), name = "post_conv_4",padding='same'), training = is_train, name = "post_conv_4_BN"))

    output = tf.layers.conv2d(conv_4, config.output_features, (5,1), name = "post_conv_5",padding='same')

    return tf.squeeze(output)

def post_net_stft(inputs, mixture, is_train):

    inputs = tf.concat([inputs, mixture], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    conv_1 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(inputs, 512, (5,1), name = "post_conv_1",padding='same'), training = is_train, name = "post_conv_1_BN"))

    conv_2 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(conv_1, 512, (5,1), name = "post_conv_2",padding='same'), training = is_train, name = "post_conv_2_BN"))

    conv_3 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(conv_2, 512, (5,1), name = "post_conv_3",padding='same'), training = is_train, name = "post_conv_3_BN"))   

    conv_4 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(conv_3, 512, (5,1), name = "post_conv_4",padding='same'), training = is_train, name = "post_conv_4_BN"))

    output = tf.layers.conv2d(conv_4, config.output_features, (5,1), name = "post_conv_5",padding='same')

    return tf.squeeze(output)

def content_encoder_stft(inputs, is_train):


    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])


    conv_1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs, 512, (5,1), name = "conv_1",padding='same'), training = is_train, name = "conv_1_BN"))

    conv_2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_1, 512, (5,1), name = "conv_2",padding='same'), training = is_train, name = "conv_2_BN"))

    conv_3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_2, 512, (5,1), name = "conv_3",padding='same'), training = is_train, name = "conv_3_BN"))

    conv_3 = tf.reshape(conv_3,[config.batch_size, config.max_phr_len , -1] )

    lstm_op = bi_static_stacked_RNN(conv_3, scope = "Encode")

    lstm_fow = lstm_op[:,:, :config.lstm_size]

    lstm_back = lstm_op[:,:, config.lstm_size:]

    emb = []

    for i in range(int(config.max_phr_len/config.code_sam)):
        emb.append(tf.concat([lstm_fow[:, i*config.code_sam,:], lstm_back[:, (i+1)*config.code_sam-1, :]], axis = -1))

    emb = tf.stack(emb)

    return emb

def decoder_stft(emb, mixture, is_train):

    if config.use_speaker:

        mixture = tf.tile(tf.reshape(mixture,[config.batch_size,1,-1]),[1,config.max_phr_len,1])


    embo = tf.tile(tf.reshape(emb[0],[config.batch_size,1,-1]),[1,config.code_sam,1])

    for i in range(1, int(config.max_phr_len/config.code_sam)):
        embs = tf.tile(tf.reshape(emb[i],[config.batch_size,1,-1]),[1,config.code_sam,1])

        embo = tf.concat([embo, embs], axis = 1)

    # mixture = tf.layers.conv1d(mixture, config.mixture_encoding, 1, name = "mixture_encoding")

    inputs = tf.concat([embo, mixture], axis = -1)

    

    lstm_op_1 = RNN(inputs, scope = "Decode_1")

    lstm_op_1 = tf.reshape(lstm_op_1, [config.batch_size, config.max_phr_len , 1, -1])

    conv_1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(lstm_op_1, 512, (5,1), name = "op_conv_1",padding='same'), training = is_train, name = "op_conv_1_BN"))

    conv_2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_1, 512, (5,1), name = "op_conv_2",padding='same'), training = is_train, name = "op_conv_2_BN"))

    conv_3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_2, 512, (5,1), name = "op_conv_3",padding='same'), training = is_train, name = "op_conv_3_BN"))

    conv_3 = tf.reshape(conv_3,[config.batch_size, config.max_phr_len , -1] )

    lstm_op_2 = RNN(conv_3, scope = "Decode_2")

    lstm_op_3 = RNN(lstm_op_2, scope = "Decode_3")

    # lstm_op_3 = bi_static_stacked_RNN(lstm_op_2, lstm_size = config.out_lstm, scope = "Decode_3")

    output = tf.layers.conv1d(lstm_op_3, config.output_features, 1, name = "decode_output")


    return output
