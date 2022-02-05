import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Zeros, glorot_normal
from tensorflow.keras.regularizers import l2
try:
    unicode
except NameError:
    unicode = str

def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, (str, unicode)):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer
    
class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)
    

class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)
    
def DF():
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
    flatten_fm1 = Flatten()(emb_fm1)
    fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([emb_list[idx], emb_list[idy]]))

    concat = Concatenate(axis = 1)(mul_layer)
    flatten_fm2 = Flatten()(concat)
    fm2_out = Dense(1, activation = 'relu')(flatten_fm2)

    fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)
    hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
    hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
    deep_out = hid2_deep

    out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm_out, deep_out]))

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model


from tensorflow.keras import backend as K
def DF2():
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
    flatten_fm1 = Flatten()(emb_fm1)
    fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([emb_list[idx], emb_list[idy]]))

    concat = Concatenate(axis = 1)(mul_layer)
    flatten_fm2 = Flatten()(concat)
    fm2_out = Dense(16, activation = 'relu')(flatten_fm2)

#     fm_out = Concatenate(axis = 1)([fm1_out, fm2_out])
    fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)
    hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
    hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
#     hid3_deep = Dropout(0.5)(Dense(64, activation = 'relu')(hid2_deep))
#     hid4_deep = Dropout(0.5)(Dense(32, activation = 'relu')(hid3_deep))
#     hid5_deep = Dropout(0.5)(Dense(16, activation = 'relu')(hid4_deep))
#     deep_out = Dropout(0.5)(Dense(1, activation = 'relu')(hid2_deep))
    deep_out = hid2_deep

    out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm_out, deep_out]))

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF3():
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
    flatten_fm1 = Flatten()(emb_fm1)
    fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([emb_list[idx], emb_list[idy]]))

    concat = Concatenate(axis = 1)(mul_layer)
    flatten_fm2 = Flatten()(concat)
    fm2_out = Dense(1, activation = 'relu')(flatten_fm2)

    fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)
    hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
    hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
    deep_out = hid2_deep

    out = Dense(1, activation = 'sigmoid')(deep_out)
#     out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm_out, deep_out]))

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model


def DF4():
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Reshape([16])(Embedding(112872, 16)(feedid))
    authorid_emb = Reshape([16])(Embedding(18789, 16)(authorid))
    bgm_song_id_emb = Reshape([16])(Embedding(25160, 16)(bgm_song_id))
    bgm_singer_id_emb = Reshape([16])(Embedding(17501, 16)(bgm_singer_id))
    userid_emb = Reshape([16])(Embedding(250249, 16)(userid))

    emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
    flatten_fm1 = Flatten()(emb_fm1)
    fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([emb_list[idx], emb_list[idy]]))

    concat = Concatenate(axis = 1)(mul_layer)
    flatten_fm2 = Flatten()(concat)
    fm2_out = Dense(1, activation = 'relu')(flatten_fm2)

    fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)
    hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
    hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
    deep_out = hid2_deep

    out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm_out, deep_out]))

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF5():
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    feedid_emb_c = Embedding(112872, 8)(feedid)
    authorid_emb_c = Embedding(18789, 8)(authorid)
    bgm_song_id_emb_c = Embedding(25160, 8)(bgm_song_id)
    bgm_singer_id_emb_c = Embedding(17501, 8)(bgm_singer_id)
    userid_emb_c = Embedding(250249, 8)(userid)
    
    u_f = Concatenate(axis = -1)([userid_emb_c, feedid_emb_c])
    u_a = Concatenate(axis = -1)([userid_emb_c, authorid_emb_c])
    u_so = Concatenate(axis = -1)([userid_emb_c, bgm_song_id_emb_c])
    u_si = Concatenate(axis = -1)([userid_emb_c, bgm_singer_id_emb_c])
    
    
    
    emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
    flatten_fm1 = Flatten()(emb_fm1)
    fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb, u_f, u_a, u_so, u_si]
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([emb_list[idx], emb_list[idy]]))

    concat = Concatenate(axis = 1)(mul_layer)
    flatten_fm2 = Flatten()(concat)
    fm2_out = Dense(1, activation = 'relu')(flatten_fm2)

    fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb,
                                     u_f, u_a, u_so, u_si])
    flatten_deep = Flatten()(deep_input)
    hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
    hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
    deep_out = hid2_deep

    out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm_out, deep_out]))

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF6():
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    feedid_emb_c = Embedding(112872, 8)(feedid)
    authorid_emb_c = Embedding(18789, 8)(authorid)
    bgm_song_id_emb_c = Embedding(25160, 8)(bgm_song_id)
    bgm_singer_id_emb_c = Embedding(17501, 8)(bgm_singer_id)
    userid_emb_c = Embedding(250249, 8)(userid)
    
    u_f = Concatenate(axis = -1)([userid_emb_c, feedid_emb_c])
    u_a = Concatenate(axis = -1)([userid_emb_c, authorid_emb_c])
    u_so = Concatenate(axis = -1)([userid_emb_c, bgm_song_id_emb_c])
    u_si = Concatenate(axis = -1)([userid_emb_c, bgm_singer_id_emb_c])
    
    u_f_list = []
    lists = [feedid_emb_c, authorid_emb_c, bgm_song_id_emb_c, bgm_singer_id_emb_c, userid_emb_c]
    for idx in range(len(lists)):
        for idy in range(idx + 1, len(lists)):
            u_f_list.append(Concatenate(axis = -1)([lists[idx], lists[idy]]))
    
    emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
    flatten_fm1 = Flatten()(emb_fm1)
    fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb] + u_f_list
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([emb_list[idx], emb_list[idy]]))

    concat = Concatenate(axis = 1)(mul_layer)
    flatten_fm2 = Flatten()(concat)
    fm2_out = Dense(1, activation = 'relu')(flatten_fm2)

    fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb] + u_f_list)
    flatten_deep = Flatten()(deep_input)
    hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
    hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
    deep_out = hid2_deep

    out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm_out, deep_out]))

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF7():
    print('use DF7...')
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    feedid_emb_c = Embedding(112872, 8)(feedid)
    authorid_emb_c = Embedding(18789, 8)(authorid)
    bgm_song_id_emb_c = Embedding(25160, 8)(bgm_song_id)
    bgm_singer_id_emb_c = Embedding(17501, 8)(bgm_singer_id)
    userid_emb_c = Embedding(250249, 8)(userid)
    
    u_f = Concatenate(axis = -1)([userid_emb_c, feedid_emb_c])
    u_a = Concatenate(axis = -1)([userid_emb_c, authorid_emb_c])
    u_so = Concatenate(axis = -1)([userid_emb_c, bgm_song_id_emb_c])
    u_si = Concatenate(axis = -1)([userid_emb_c, bgm_singer_id_emb_c])
    
    
    
    emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
    flatten_fm1 = Flatten()(emb_fm1)
    fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb, u_f, u_a, u_so, u_si]
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([emb_list[idx], emb_list[idy]]))

    concat = Concatenate(axis = 1)(mul_layer)
    flatten_fm2 = Flatten()(concat)
    fm2_out = Dense(1, activation = 'relu')(flatten_fm2)

#     fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb,
                                     u_f, u_a, u_so, u_si])
    flatten_deep = Flatten()(deep_input)
    hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
    hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
    deep_out = hid2_deep

    out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out, deep_out]))

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF8():
    print('use DF8...')
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    feedid_emb_1 = Reshape([1])(Embedding(112872, 1)(feedid))
    authorid_emb_1 = Reshape([1])(Embedding(18789, 1)(authorid))
    bgm_song_id_emb_1 = Reshape([1])(Embedding(25160, 1)(bgm_song_id))
    bgm_singer_id_emb_1 = Reshape([1])(Embedding(17501, 1)(bgm_singer_id))
    userid_emb_1 = Reshape([1])(Embedding(250249, 1)(userid))

    feedid_emb_c = Embedding(112872, 8)(feedid)
    authorid_emb_c = Embedding(18789, 8)(authorid)
    bgm_song_id_emb_c = Embedding(25160, 8)(bgm_song_id)
    bgm_singer_id_emb_c = Embedding(17501, 8)(bgm_singer_id)
    userid_emb_c = Embedding(250249, 8)(userid)

    u_f = Concatenate(axis = -1)([userid_emb_c, feedid_emb_c])
    u_a = Concatenate(axis = -1)([userid_emb_c, authorid_emb_c])
    u_so = Concatenate(axis = -1)([userid_emb_c, bgm_song_id_emb_c])
    u_si = Concatenate(axis = -1)([userid_emb_c, bgm_singer_id_emb_c])


    fm1_out = Add()([feedid_emb_1, authorid_emb_1, bgm_song_id_emb_1, bgm_singer_id_emb_1, userid_emb_1])
    #     emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
    #     flatten_fm1 = Flatten()(emb_fm1)
    #     fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb, u_f, u_a, u_so, u_si]
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([emb_list[idx], emb_list[idy]]))

    concat = Concatenate(axis = 1)(mul_layer)
    flatten_fm2 = Flatten()(concat)
    fm2_out = Dense(1, activation = 'relu')(flatten_fm2)

    #     fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb,
                                     u_f, u_a, u_so, u_si])
    flatten_deep = Flatten()(deep_input)
    hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
    hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
    deep_out = hid2_deep

    out = Dense(1, activation = 'sigmoid')(Concatenate()([fm1_out, fm2_out, deep_out]))

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF8():
    print('use DF8...')
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    feedid_emb_1 = Reshape([1])(Embedding(112872, 1)(feedid))
    authorid_emb_1 = Reshape([1])(Embedding(18789, 1)(authorid))
    bgm_song_id_emb_1 = Reshape([1])(Embedding(25160, 1)(bgm_song_id))
    bgm_singer_id_emb_1 = Reshape([1])(Embedding(17501, 1)(bgm_singer_id))
    userid_emb_1 = Reshape([1])(Embedding(250249, 1)(userid))

    feedid_emb_c = Embedding(112872, 8)(feedid)
    authorid_emb_c = Embedding(18789, 8)(authorid)
    bgm_song_id_emb_c = Embedding(25160, 8)(bgm_song_id)
    bgm_singer_id_emb_c = Embedding(17501, 8)(bgm_singer_id)
    userid_emb_c = Embedding(250249, 8)(userid)

    u_f = Concatenate(axis = -1)([userid_emb_c, feedid_emb_c])
    u_a = Concatenate(axis = -1)([userid_emb_c, authorid_emb_c])
    u_so = Concatenate(axis = -1)([userid_emb_c, bgm_song_id_emb_c])
    u_si = Concatenate(axis = -1)([userid_emb_c, bgm_singer_id_emb_c])


    fm1_out = Add()([feedid_emb_1, authorid_emb_1, bgm_song_id_emb_1, bgm_singer_id_emb_1, userid_emb_1])
    
    
    
    #     emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
    #     flatten_fm1 = Flatten()(emb_fm1)
    #     fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([Reshape([16])(emb_list[idx]), Reshape([16])(emb_list[idy])]))


    class SumLayer(Layer):
        def __init__(self, **kwargs):
            super(SumLayer, self).__init__(**kwargs)
        def call(self, inputs):
    #         print(inputs.shape)
            inputs = K.expand_dims(inputs)
            return K.sum(inputs, axis=1)
        def compute_output_shape(self, input_shape):
            return tuple([input_shape[0], 1])

    sum_layer = Add()(mul_layer)

    fm2_out = SumLayer()(sum_layer)

    #     concat = Concatenate(axis = 1)(mul_layer)
    #     flatten_fm2 = Flatten()(concat)
    #     fm2_out = Dense(1, activation = 'relu')(flatten_fm2)

    #     fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)
#     hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
#     hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
#     deep_out = Dropout(0.5)(Dense(1, activation = 'relu')(hid2_deep))

    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=502)(flatten_deep)

    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=502))(hid_deep)

    out = Add()([fm1_out, fm2_out, deep_out])
#     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF9():
    print('use DF9...')
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    feedid_emb_1 = Reshape([1])(Embedding(112872, 1)(feedid))
    authorid_emb_1 = Reshape([1])(Embedding(18789, 1)(authorid))
    bgm_song_id_emb_1 = Reshape([1])(Embedding(25160, 1)(bgm_song_id))
    bgm_singer_id_emb_1 = Reshape([1])(Embedding(17501, 1)(bgm_singer_id))
    userid_emb_1 = Reshape([1])(Embedding(250249, 1)(userid))

    feedid_emb_c = Embedding(112872, 8)(feedid)
    authorid_emb_c = Embedding(18789, 8)(authorid)
    bgm_song_id_emb_c = Embedding(25160, 8)(bgm_song_id)
    bgm_singer_id_emb_c = Embedding(17501, 8)(bgm_singer_id)
    userid_emb_c = Embedding(250249, 8)(userid)

    u_f = Concatenate(axis = -1)([userid_emb_c, feedid_emb_c])
    u_a = Concatenate(axis = -1)([userid_emb_c, authorid_emb_c])
    u_so = Concatenate(axis = -1)([userid_emb_c, bgm_song_id_emb_c])
    u_si = Concatenate(axis = -1)([userid_emb_c, bgm_singer_id_emb_c])

    fm1_out = reduce_sum(input_tensor = Concatenate()([feedid_emb_1, authorid_emb_1, bgm_song_id_emb_1, bgm_singer_id_emb_1, userid_emb_1]), axis = -1, keep_dims = True)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    fm2_out = FM()(Concatenate(axis = 1)(emb_list))
    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)

#     hid1_deep = Dropout(0)(Dense(128, activation = 'relu')(flatten_deep))
#     hid2_deep = Dropout(0)(Dense(128, activation = 'relu')(hid1_deep))
#     deep_out = tf.keras.layers.Dense(
#         1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid2_deep)
    
    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)

    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)

    out = Add()([fm1_out, fm2_out, deep_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF10():
    print('use DF10...')
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    feedid_emb_1 = Reshape([1])(Embedding(112872, 1)(feedid))
    authorid_emb_1 = Reshape([1])(Embedding(18789, 1)(authorid))
    bgm_song_id_emb_1 = Reshape([1])(Embedding(25160, 1)(bgm_song_id))
    bgm_singer_id_emb_1 = Reshape([1])(Embedding(17501, 1)(bgm_singer_id))
    userid_emb_1 = Reshape([1])(Embedding(250249, 1)(userid))
    
    fm1_out = reduce_sum(input_tensor = Concatenate()([feedid_emb_1, authorid_emb_1, bgm_song_id_emb_1, bgm_singer_id_emb_1, userid_emb_1]), axis = -1, keep_dims = True)
    
#     emb_fm1 = Concatenate(axis = 1)([feedid, authorid, bgm_song_id, bgm_singer_id, userid])
#     flatten_fm1 = Flatten()(emb_fm1)
#     fm1_out = Dense(1, activation = 'linear')(flatten_fm1)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    mul_layer = []
    for idx in range(len(emb_list)):
        for idy in range(idx+1, len(emb_list)):
            mul_layer.append(multiply([emb_list[idx], emb_list[idy]]))

    concat = Concatenate(axis = 1)(mul_layer)
    flatten_fm2 = Flatten()(concat)
    fm2_out = Dense(1, activation = 'relu')(flatten_fm2)

    fm_out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm1_out, fm2_out]))

    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)
    hid1_deep = Dropout(0.5)(Dense(128, activation = 'relu')(flatten_deep))
    hid2_deep = Dropout(0.5)(Dense(128, activation = 'relu')(hid1_deep))
    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid2_deep)

    out = Dense(1, activation = 'sigmoid')(Concatenate(axis = 1)([fm_out, deep_out]))

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF11():
    print('use DF11...')
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    feedid_emb_1 = Reshape([1])(Embedding(112872, 1)(feedid))
    authorid_emb_1 = Reshape([1])(Embedding(18789, 1)(authorid))
    bgm_song_id_emb_1 = Reshape([1])(Embedding(25160, 1)(bgm_song_id))
    bgm_singer_id_emb_1 = Reshape([1])(Embedding(17501, 1)(bgm_singer_id))
    userid_emb_1 = Reshape([1])(Embedding(250249, 1)(userid))

    feedid_emb_c = Embedding(112872, 8)(feedid)
    authorid_emb_c = Embedding(18789, 8)(authorid)
    bgm_song_id_emb_c = Embedding(25160, 8)(bgm_song_id)
    bgm_singer_id_emb_c = Embedding(17501, 8)(bgm_singer_id)
    userid_emb_c = Embedding(250249, 8)(userid)

    u_f = Concatenate(axis = -1)([userid_emb_c, feedid_emb_c])
    u_a = Concatenate(axis = -1)([userid_emb_c, authorid_emb_c])
    u_so = Concatenate(axis = -1)([userid_emb_c, bgm_song_id_emb_c])
    u_si = Concatenate(axis = -1)([userid_emb_c, bgm_singer_id_emb_c])

    fm1_out = reduce_sum(input_tensor = Concatenate()([feedid_emb_1, authorid_emb_1, bgm_song_id_emb_1, bgm_singer_id_emb_1, userid_emb_1]), axis = -1, keep_dims = True)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    fm2_out = FM()(Concatenate(axis = 1)(emb_list))
    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)

    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)

    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)

    conv_input = Conv2D(32, 3)(Reshape([deep_input.shape[1],deep_input.shape[2],1])(deep_input))
    conv_flatten = Flatten()(conv_input)
    conv_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_flatten)
    conv_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_deep)


    out = Add()([fm1_out, fm2_out, deep_out, conv_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF12():
    print('use DF12...')
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)

    feedid_emb_1 = Reshape([1])(Embedding(112872, 1)(feedid))
    authorid_emb_1 = Reshape([1])(Embedding(18789, 1)(authorid))
    bgm_song_id_emb_1 = Reshape([1])(Embedding(25160, 1)(bgm_song_id))
    bgm_singer_id_emb_1 = Reshape([1])(Embedding(17501, 1)(bgm_singer_id))
    userid_emb_1 = Reshape([1])(Embedding(250249, 1)(userid))

    feedid_emb_c = Embedding(112872, 8)(feedid)
    authorid_emb_c = Embedding(18789, 8)(authorid)
    bgm_song_id_emb_c = Embedding(25160, 8)(bgm_song_id)
    bgm_singer_id_emb_c = Embedding(17501, 8)(bgm_singer_id)
    userid_emb_c = Embedding(250249, 8)(userid)

    u_f = Concatenate(axis = -1)([userid_emb_c, feedid_emb_c])
    u_a = Concatenate(axis = -1)([userid_emb_c, authorid_emb_c])
    u_so = Concatenate(axis = -1)([userid_emb_c, bgm_song_id_emb_c])
    u_si = Concatenate(axis = -1)([userid_emb_c, bgm_singer_id_emb_c])

    fm1_out = reduce_sum(input_tensor = Concatenate()([feedid_emb_1, authorid_emb_1, bgm_song_id_emb_1, bgm_singer_id_emb_1, userid_emb_1]), axis = -1, keep_dims = True)

    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    fm2_out = FM()(Concatenate(axis = 1)(emb_list))
    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)

    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)

    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)

#     conv_input = Conv2D(32, 3)(Reshape([deep_input.shape[1],deep_input.shape[2],1])(deep_input))
#     conv_flatten = Flatten()(conv_input)
    conv_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)
    conv_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_deep)


    out = Add()([fm1_out, fm2_out, deep_out, conv_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DF13():
    print('use DF13...')
    feedid = Input(shape=(1,), name = 'feedid')
    authorid = Input(shape=(1,), name = 'authorid')
    bgm_song_id = Input(shape=(1,), name = 'bgm_song_id')
    bgm_singer_id = Input(shape=(1,), name = 'bgm_singer_id')
    userid = Input(shape=(1,), name = 'userid')

    # 二次项和DNN层sparse特征Embedding
    feedid_emb = Embedding(112872, 16)(feedid)
    authorid_emb = Embedding(18789, 16)(authorid)
    bgm_song_id_emb = Embedding(25160, 16)(bgm_song_id)
    bgm_singer_id_emb = Embedding(17501, 16)(bgm_singer_id)
    userid_emb = Embedding(250249, 16)(userid)
    # 卷积层sparse特征Embedding
    feedid_emb_co = Reshape([8,8,1])(Embedding(112872, 64)(feedid))
    authorid_emb_co = Reshape([8,8,1])(Embedding(18789, 64)(authorid))
    bgm_song_id_emb_co = Reshape([8,8,1])(Embedding(25160, 64)(bgm_song_id))
    bgm_singer_id_emb_co = Reshape([8,8,1])(Embedding(17501, 64)(bgm_singer_id))
    userid_emb_co = Reshape([8,8,1])(Embedding(250249, 64)(userid))
    # 一次项特征Embedding
    feedid_emb_1 = Reshape([1])(Embedding(112872, 1)(feedid))
    authorid_emb_1 = Reshape([1])(Embedding(18789, 1)(authorid))
    bgm_song_id_emb_1 = Reshape([1])(Embedding(25160, 1)(bgm_song_id))
    bgm_singer_id_emb_1 = Reshape([1])(Embedding(17501, 1)(bgm_singer_id))
    userid_emb_1 = Reshape([1])(Embedding(250249, 1)(userid))
    # 特征拼接Embedding
    feedid_emb_c = Embedding(112872, 8)(feedid)
    authorid_emb_c = Embedding(18789, 8)(authorid)
    bgm_song_id_emb_c = Embedding(25160, 8)(bgm_song_id)
    bgm_singer_id_emb_c = Embedding(17501, 8)(bgm_singer_id)
    userid_emb_c = Embedding(250249, 8)(userid)
    # 特征拼接
    u_f = Concatenate(axis = -1)([userid_emb_c, feedid_emb_c])
    u_a = Concatenate(axis = -1)([userid_emb_c, authorid_emb_c])
    u_so = Concatenate(axis = -1)([userid_emb_c, bgm_song_id_emb_c])
    u_si = Concatenate(axis = -1)([userid_emb_c, bgm_singer_id_emb_c])
    # 一次项结果
    fm1_out = reduce_sum(input_tensor = Concatenate()([feedid_emb_1, authorid_emb_1, bgm_song_id_emb_1, bgm_singer_id_emb_1, userid_emb_1]), axis = -1, keep_dims = True)
    # 二次项计算
    emb_list = [feedid_emb, authorid_emb, bgm_song_id_emb, bgm_singer_id_emb, userid_emb]
    fm2_out = FM()(Concatenate(axis = 1)(emb_list))
    # DNN
    deep_input = Concatenate(axis=1)([feedid_emb, authorid_emb,
                                bgm_song_id_emb, bgm_singer_id_emb, userid_emb])
    flatten_deep = Flatten()(deep_input)

    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)

    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)

    # DNN Embedding+卷积+DNN
    conv_input = Conv2D(32, 3)(Reshape([deep_input.shape[1],deep_input.shape[2],1])(deep_input))
    conv_flatten = Flatten()(conv_input)
    conv_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_flatten)
    conv_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_deep)

    # 卷积Embedding+卷积+DNN
    conv_emb = Concatenate(axis = 3)([feedid_emb_co, authorid_emb_co, bgm_song_id_emb_co, bgm_singer_id_emb_co, userid_emb_co])
    conv1 = Conv2D(32,3,padding = 'same')(conv_emb)
#     conv2 = Conv2D(32,3, padding = 'same')(conv1)
#     pool = AveragePooling2D()(conv2)
#     conv = Add()([pool, Conv2D(32,3,2, padding = 'same')(conv_emb)])
    conv_t_flatten = Flatten()(conv1)
    conv_t_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_t_flatten)
    conv_t_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_t_deep)

    # 结果
    out = Add()([fm1_out, fm2_out, deep_out, conv_out, conv_t_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)

    model = Model(inputs = [feedid, authorid, bgm_song_id, bgm_singer_id, userid],
                 outputs = [out])
    return model

def DFX(sparse_features, single_dense_features = [], multi_dense_features = []):
    sparse_input = [Input(shape=(1,)) for x in range(len(sparse_features))]
    sparse_emb_1 = [Reshape([1])(Embedding(sparse_features[idx], 1)(sparse_input[idx])) for idx in range(len(sparse_input))]
    sparse_emb_16 = [(Embedding(sparse_features[idx], 16)(sparse_input[idx])) for idx in range(len(sparse_input))]
    sparse_emb_co = [Reshape([8,8,1])(Embedding(sparse_features[idx], 64)(sparse_input[idx])) for idx in range(len(sparse_input))]
    sparse_emb_8 = [(Embedding(sparse_features[idx], 8)(sparse_input[idx])) for idx in range(len(sparse_input))]
    # 一次项
    fm1_out = reduce_sum(input_tensor = Concatenate()(sparse_emb_1), axis = -1, keep_dims = True)
    # 二次项
    fm2_out = FM()(Concatenate(axis = 1)(sparse_emb_16))
    # DNN
    deep_input = Concatenate(axis=1)(sparse_emb_16)
    flatten_deep = Flatten()(deep_input)
    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)
    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)
    # DNN Embedding+卷积+DNN
    conv_input = Conv2D(32, 3)(Reshape([deep_input.shape[1],deep_input.shape[2],1])(deep_input))
    conv_flatten = Flatten()(conv_input)
    conv_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_flatten)
    conv_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_deep)
    # 卷积Embedding+卷积+DNN
    conv_emb = Concatenate(axis = 3)(sparse_emb_co)
    conv1 = Conv2D(32,3,padding = 'same')(conv_emb)
    conv_t_flatten = Flatten()(conv1)
    conv_t_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_t_flatten)
    conv_t_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_t_deep)
    # 结果
    out = Add()([fm1_out, fm2_out, deep_out, conv_out, conv_t_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)
    model = Model(inputs = sparse_input,
                 outputs = [out])
    return model

def DFX1(sparse_features, dense_features = None, con_dense_features = None):
    sparse_features_names = [k for k in sparse_features]
    sparse_input = [Input(shape=(1,), name = sparse_features_names[x]) for x in range(len(sparse_features_names))]
    sparse_emb_1 = [Reshape([1])(Embedding(sparse_features[sparse_features_names[idx]], 1)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_16 = [(Embedding(sparse_features[sparse_features_names[idx]], 16)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_co = [Reshape([8,8,1])(Embedding(sparse_features[sparse_features_names[idx]], 64)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_8 = [(Embedding(sparse_features[sparse_features_names[idx]], 8)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    
    if dense_features != None:
        dense_features_names = [k for k in dense_features]
        dense_input = [Input(shape=(dense_features[x],), name = dense_features_names[x]) for x in range(len(dense_features_names))]
        dense_emb_1 = [Dense(1)(x) for x in dense_input]
        dense_emb_16 = [Dense(16)(x) for x in dense_input]
        dense_emb_8 = [Dense(8)(x) for x in dense_input]
        dense_emb_co = [Reshape([8,8,1])((Dense(64))(x)) for x in dense_input]
    else:
        dense_emb_1 = []
        dense_emb_16 = []
        dense_emb_8 = []
        dense_emb_co = []
#         con_emb_16 = []
    
    con_emb_16 = []
    con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[0]]))
    con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[1]]))
    con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[2]]))
    con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[3]]))
    
    # 一次项
    fm1_input = sparse_emb_1 + dense_emb_1
    fm1_out = reduce_sum(input_tensor = Concatenate()(fm1_input), axis = -1, keep_dims = True)
    
    # 二次项
    fm2_input = sparse_emb_16 + dense_emb_16 + con_emb_16
    fm2_out = FM()(Concatenate(axis = 1)(fm2_input))
    
    # DNN
    dnn_input = sparse_emb_16 + dense_emb_16 + con_emb_16
    deep_input = Concatenate(axis=1)(sparse_emb_16)
    flatten_deep = Flatten()(deep_input)
    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)
    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)
    # DNN Embedding+卷积+DNN
    conv_input = Conv2D(32, 3)(Reshape([deep_input.shape[1],deep_input.shape[2],1])(deep_input))
    conv_flatten = Flatten()(conv_input)
    conv_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_flatten)
    conv_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_deep)
    # 卷积Embedding+卷积+DNN
    conv_input = sparse_emb_co + dense_emb_co
    conv_emb = Concatenate(axis = 3)(conv_input)
    conv1 = Conv2D(32,3,padding = 'same')(conv_emb)
    conv_t_flatten = Flatten()(conv1)
    conv_t_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_t_flatten)
    conv_t_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_t_deep)
    # 结果
    out = Add()([fm1_out, fm2_out, deep_out, conv_out, conv_t_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)
    model = Model(inputs = sparse_input,
                 outputs = [out])
    return model

def DFXX(sparse_features, dense_features = None, con_dense_features = None):
    sparse_features_names = [k for k in sparse_features]
    sparse_input = [Input(shape=(1,), name = 'sparse_'+sparse_features_names[x]) for x in range(len(sparse_features_names))]
    sparse_emb_1 = [Reshape([1,1])(Embedding(sparse_features[sparse_features_names[idx]], 1)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_16 = [(Embedding(sparse_features[sparse_features_names[idx]], 16)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_co = [Reshape([8,8,1])(Embedding(sparse_features[sparse_features_names[idx]], 64)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_8 = [(Embedding(sparse_features[sparse_features_names[idx]], 8)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    
    if dense_features != None:
        dense_features_names = [k for k in dense_features]
        dense_input = [Input(shape=(dense_features[dense_features_names[x]],), name = 'dense_'+dense_features_names[x]) for x in range(len(dense_features_names))]
        dense_emb_1 = [Reshape([1,1])(Dense(1)(x)) for x in dense_input]
        dense_emb_16 = [Reshape([1,16])(Dense(16)(x)) for x in dense_input]
        dense_emb_8 = [Reshape([1,8])(Dense(8)(x)) for x in dense_input]
        dense_emb_co = [Reshape([8,8,1])((Dense(64))(x)) for x in dense_input]
    else:
        dense_emb_1 = []
        dense_emb_16 = []
        dense_emb_8 = []
        dense_emb_co = []
#         con_emb_16 = []
    
    con_emb_16 = []
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[0]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[1]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[2]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[3]]))
    
    # 一次项
    fm1_input = sparse_emb_1 + dense_emb_1
    fm1_out = reduce_sum(input_tensor = Concatenate()(fm1_input), axis = -1, keep_dims = True)
    
    # 二次项
    fm2_input = sparse_emb_16 + dense_emb_1 + con_emb_16
    fm2_out = FM()(Concatenate(axis = -1)(fm2_input))
    
    # DNN
    dnn_input = sparse_emb_16 + dense_emb_1 + con_emb_16
    deep_input = Concatenate(axis=-1)(dnn_input)
    flatten_deep = Flatten()(deep_input)
    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)
    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)
    # DNN Embedding+卷积+DNN
    dnn_conv_input = Concatenate(axis=1)(sparse_emb_16)
    conv_input = Conv2D(32, 3)(Reshape([dnn_conv_input.shape[1],dnn_conv_input.shape[2],1])(dnn_conv_input))
    conv_flatten = Flatten()(conv_input)
    conv_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_flatten)
    conv_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_deep)
    # 卷积Embedding+卷积+DNN
    conv_input = sparse_emb_co
    conv_emb = Concatenate(axis = 3)(conv_input)
    conv1 = Conv2D(32,3,padding = 'same')(conv_emb)
    conv_t_flatten = Flatten()(conv1)
    conv_t_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_t_flatten)
    conv_t_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_t_deep)
    # 结果
    out = Add()([fm1_out, fm2_out, deep_out, conv_out, conv_t_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)
    if dense_features != None:
        model = Model(inputs = sparse_input + dense_input,
                     outputs = [out])
    else:
        model = Model(inputs = sparse_input,
                     outputs = [out])
    return model

def DFXXX(sparse_features, dense_features = None, con_dense_features = None):
    sparse_features_names = [k for k in sparse_features]
    sparse_input = [Input(shape=(1,), name = 'sparse_'+sparse_features_names[x]) for x in range(len(sparse_features_names))]
    sparse_emb_1 = [Reshape([1,1])(Embedding(sparse_features[sparse_features_names[idx]], 1)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_16 = [(Embedding(sparse_features[sparse_features_names[idx]], 16)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_co = [Reshape([8,8,1])(Embedding(sparse_features[sparse_features_names[idx]], 64)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_8 = [(Embedding(sparse_features[sparse_features_names[idx]], 8)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    
    if dense_features != None:
        dense_features_names = [k for k in dense_features]
        dense_input = [Input(shape=(dense_features[dense_features_names[x]],), name = 'dense_'+dense_features_names[x]) for x in range(len(dense_features_names))]
        dense_input_reshape = [Reshape([1, dense_features[dense_features_names[x]]])(dense_input[x]) for x in range(len(dense_features_names))]
        dense_emb_1 = [Reshape([1,1])(Dense(1)(x)) for x in dense_input]
        dense_emb_16 = [Reshape([1,16])(Dense(16)(x)) for x in dense_input]
        dense_emb_8 = [Reshape([1,8])(Dense(8)(x)) for x in dense_input]
        dense_emb_co = [Reshape([8,8,1])((Dense(64))(x)) for x in dense_input]
    else:
        dense_emb_1 = []
        dense_input_reshape = []
        dense_emb_16 = []
        dense_emb_8 = []
        dense_emb_co = []
#         con_emb_16 = []
    
    con_emb_16 = []
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[0]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[1]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[2]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[3]]))
    
    # 一次项
    fm1_input = sparse_emb_1 + dense_input_reshape
    fm1_out = reduce_sum(input_tensor = Concatenate()(fm1_input), axis = -1, keep_dims = True)
    
    # 二次项
    fm2_input = sparse_emb_16 + dense_input_reshape + con_emb_16
    fm2_out = FM()(Concatenate(axis = -1)(fm2_input))
    
    # DNN
    dnn_input = sparse_emb_16 + dense_input_reshape + con_emb_16
    deep_input = Concatenate(axis=-1)(dnn_input)
    flatten_deep = Flatten()(deep_input)
    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)
    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)
    # DNN Embedding+卷积+DNN
    dnn_conv_input = Concatenate(axis=1)(sparse_emb_16)
    conv_input = Conv2D(32, 3)(Reshape([dnn_conv_input.shape[1],dnn_conv_input.shape[2],1])(dnn_conv_input))
    conv_flatten = Flatten()(conv_input)
    conv_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_flatten)
    conv_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_deep)
    # 卷积Embedding+卷积+DNN
    conv_input = sparse_emb_co
    conv_emb = Concatenate(axis = 3)(conv_input)
    conv1 = Conv2D(32,3,padding = 'same')(conv_emb)
    conv_t_flatten = Flatten()(conv1)
    conv_t_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_t_flatten)
    conv_t_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_t_deep)
    # 结果
    out = Add()([fm1_out, fm2_out, deep_out, conv_out, conv_t_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)
    if dense_features != None:
        model = Model(inputs = sparse_input + dense_input,
                     outputs = [out])
    else:
        model = Model(inputs = sparse_input,
                     outputs = [out])
    return model

def DFX_TK(sparse_features, dense_features = None, con_dense_features = None, tag_feature = None, keyword_feature = None):
    print('use model DFX_TK...')
    print('{}:{}'.format('sparse_features', [k for k in sparse_features]))
    if dense_features != None:
        print('{}:{}'.format('dense_features', [k for k in dense_features]))
    if tag_feature != None:
        print('tag_feature')
    if keyword_feature != None:
        print('keyword_feature')
    sparse_features_names = [k for k in sparse_features]
    sparse_input = [Input(shape=(1,), name = 'sparse_'+sparse_features_names[x]) for x in range(len(sparse_features_names))]
    sparse_emb_1 = [Reshape([1,1])(Embedding(sparse_features[sparse_features_names[idx]], 1)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_16 = [(Embedding(sparse_features[sparse_features_names[idx]], 16)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_co = [Reshape([8,8,1])(Embedding(sparse_features[sparse_features_names[idx]], 64)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_8 = [(Embedding(sparse_features[sparse_features_names[idx]], 8)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    
    if dense_features != None:
        dense_features_names = [k for k in dense_features]
        dense_input = [Input(shape=(dense_features[dense_features_names[x]],), name = 'dense_'+dense_features_names[x]) for x in range(len(dense_features_names))]
        dense_input_reshape = [Reshape([1, dense_features[dense_features_names[x]]])(dense_input[x]) for x in range(len(dense_features_names))]
        dense_emb_1_sing = [Reshape([1,1])(Dense(1)(x)) for x in dense_input if x.shape[1] == 1]
        dense_emb_1_mul = [Reshape([1,1])(Dense(1)(x)) for x in dense_input if x.shape[1] > 1]
        dense_emb_1 = dense_emb_1_sing + dense_emb_1_mul
        
        dense_emb_16 = [Reshape([1,16])(Dense(16)(x)) for x in dense_input if x.shape[1] >= 16]
        
        dense_emb_8 = [Reshape([1,8])(Dense(8)(x)) for x in dense_input]
        dense_emb_co = [Reshape([8,8,1])((Dense(64))(x)) for x in dense_input]
    else:
        dense_emb_1 = []
        dense_input_reshape = []
        dense_emb_16 = []
        dense_emb_8 = []
        dense_emb_co = []
    if tag_feature != None:
        sparse_tag_input = Input(shape=(tag_feature[0],), name = 'sparse_tag')
        sparse_input.append(sparse_tag_input)
        tag_emb = Embedding(tag_feature[1], 8, mask_zero = True, input_length = tag_feature[0])(sparse_tag_input)
        tag_emb = Reshape([tag_feature[0], 8, 1])(tag_emb)
        tag_conv1 = Conv2D(16,3,padding = 'same')(tag_emb)
        tag_conv2 = Conv2D(32,3,padding = 'same',strides = 2)(tag_conv1)
        tag_conv3 = Conv2D(16,3,padding = 'same', strides = 2)(tag_conv2)
        tag_flatten = Flatten()(tag_conv3)
        tag_emb_16 = Reshape([1,16])(Dense(16, activation = 'relu')(tag_flatten))
        tag_emb_1 = Reshape([1,1])(Dense(1, activation = 'relu')(tag_flatten))
        sparse_emb_16.append(tag_emb_16)
        sparse_emb_1.append(tag_emb_1)
        
    if keyword_feature != None:
        sparse_keyword_input = Input(shape=(keyword_feature[0],), name = 'sparse_keyword')
        sparse_input.append(sparse_keyword_input)
        keyword_emb = Embedding(keyword_feature[1], 8, mask_zero = True, input_length = keyword_feature[0])(sparse_keyword_input)
        keyword_emb = Reshape([keyword_feature[0], 8, 1])(keyword_emb)
        keyword_conv1 = Conv2D(16,3,padding = 'same')(keyword_emb)
        keyword_conv2 = Conv2D(32,3,padding = 'same',strides = 2)(keyword_conv1)
        keyword_conv3 = Conv2D(16,3,padding = 'same', strides = 2)(keyword_conv2)
        keyword_flatten = Flatten()(keyword_conv3)
        keyword_emb_16 = Reshape([1,16])(Dense(16, activation = 'relu')(keyword_flatten))
        keyword_emb_1 = Reshape([1,1])(Dense(1, activation = 'relu')(keyword_flatten))
        sparse_emb_16.append(keyword_emb_16)
        sparse_emb_1.append(keyword_emb_1)
        
        
#         con_emb_16 = []
    
    con_emb_16 = []
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[0]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[1]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[2]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[3]]))
    
    # 一次项
    fm1_input = sparse_emb_1 + dense_emb_1
    fm1_out = reduce_sum(input_tensor = Concatenate()(fm1_input), axis = -1, keep_dims = True)
    
    # 二次项
    fm2_input = sparse_emb_16 + dense_emb_16 + con_emb_16 + dense_emb_1_sing
    fm2_out = FM()(Concatenate(axis = -1)(fm2_input))
    
    # DNN
    dnn_input = sparse_emb_16 + dense_input_reshape + con_emb_16
    deep_input = Concatenate(axis=-1)(dnn_input)
    flatten_deep = Flatten()(deep_input)
    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)
    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)
    # DNN Embedding+卷积+DNN
    dnn_conv_input = Concatenate(axis=1)(sparse_emb_16 + dense_emb_16)
    conv_input = Conv2D(32, 3)(Reshape([dnn_conv_input.shape[1],dnn_conv_input.shape[2],1])(dnn_conv_input))
    conv_flatten = Flatten()(conv_input)
    conv_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_flatten)
    conv_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_deep)
    # 卷积Embedding+卷积+DNN
    conv_input = sparse_emb_co + dense_emb_co
    conv_emb = Concatenate(axis = 3)(conv_input)
    conv1 = Conv2D(32,3,padding = 'same')(conv_emb)
    conv_t_flatten = Flatten()(conv1)
    conv_t_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_t_flatten)
    conv_t_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_t_deep)
    # 结果
    out = Add()([fm1_out, fm2_out, deep_out, conv_out, conv_t_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)
    if dense_features != None:
        model = Model(inputs = sparse_input + dense_input,
                     outputs = [out])
    else:
        model = Model(inputs = sparse_input,
                     outputs = [out])
    return model

def conv(sparse_features):
    print('ready conv model...')
    sparse_features_names = [k for k in sparse_features]
    sparse_input = [Input(shape=(1,), name = 'sparse_'+sparse_features_names[x]) for x in range(len(sparse_features_names))]
    sparse_emb_co = [Reshape([16,16,1])(Embedding(sparse_features[sparse_features_names[idx]], 256)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    conv_emb = Concatenate(axis = 3)(sparse_emb_co)
    
    conv1 = Conv2D(64,7,padding = 'same')(conv_emb)
    pool1 = MaxPool2D(strides = 2, padding = 'same')(conv1)
    
    conv2_1 = Conv2D(64,3,padding = 'same', activation = 'relu')(pool1)
    conv2_2 = Conv2D(64,3,padding = 'same', activation = 'relu')(conv2_1)
    conv2_3 = pool1 + conv2_1
    conv2_4 = Conv2D(64,3,padding = 'same', activation = 'relu')(conv2_3)
    conv2_5 = Conv2D(64,3,padding = 'same', activation = 'relu')(conv2_4)
    conv2 = conv2_3 + conv2_5
    
    conv3_1 = Conv2D(128,3,strides = (2,2), padding = 'same', activation = 'relu')(conv2)
    conv3_1_1 = Conv2D(128,1,strides = (2,2), padding = 'same', activation = None)(conv2)
    conv3_2 = Conv2D(128,3,padding = 'same', activation = 'relu')(conv3_1)
    conv3_3 = conv3_1_1 + conv3_2
    conv3_4 = Conv2D(128,3,padding = 'same', activation = 'relu')(conv3_3)
    conv3_5 = Conv2D(128,3,padding = 'same', activation = 'relu')(conv3_4)
    conv3 = conv3_4 + conv3_5
    
    conv4_1 = Conv2D(256,3,strides = (2,2), padding = 'same', activation = 'relu')(conv3)
    conv4_1_1 = Conv2D(256,1,strides = (2,2), padding = 'same', activation = None)(conv3)
    conv4_2 = Conv2D(256,3,padding = 'same', activation = 'relu')(conv4_1)
    conv4_3 = conv4_1_1 + conv4_2
    conv4_4 = Conv2D(256,3,padding = 'same', activation = 'relu')(conv4_3)
    conv4_5 = Conv2D(256,3,padding = 'same', activation = 'relu')(conv4_4)
    conv4 = conv4_3 + conv4_5
    
#     conv5_1 = Conv2D(512,3,strides = (2,2), padding = 'same', activation = 'relu')(conv4)
#     conv5_1_1 = Conv2D(512,1,strides = (2,2), padding = 'same', activation = None)(conv4)
#     conv5_2 = Conv2D(512,3,padding = 'same', activation = 'relu')(conv5_1)
#     conv5_3 = conv5_1_1 + conv5_2
#     conv5_4 = Conv2D(512,3,padding = 'same', activation = 'relu')(conv5_3)
#     conv5_5 = Conv2D(512,3,padding = 'same', activation = 'relu')(conv5_4)
#     conv5 = conv5_3+conv5_5
    
    pool2 = AveragePooling2D(pool_size = (3,3), strides = 2, padding = 'same')(conv4)
    out = Conv2D(1,1,padding = 'same', activation = None)(pool2)
    outs = Reshape([1])(out)
    
    model = Model(inputs = sparse_input, outputs = [outs])
    return model
    
def DFX_TK2(sparse_features, dense_features = None, con_dense_features = None, tag_feature = None, keyword_feature = None):
    print('use model DFX_TK2...')
    print('{}:{}'.format('sparse_features', [k for k in sparse_features]))
    if dense_features != None:
        print('{}:{}'.format('dense_features', [k for k in dense_features]))
    if tag_feature != None:
        print('tag_feature')
    if keyword_feature != None:
        print('keyword_feature')
    sparse_features_names = [k for k in sparse_features]
    sparse_input = [Input(shape=(1,), name = 'sparse_'+sparse_features_names[x]) for x in range(len(sparse_features_names))]
    sparse_emb_1 = [Reshape([1,1])(Embedding(sparse_features[sparse_features_names[idx]], 1)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_16 = [(Embedding(sparse_features[sparse_features_names[idx]], 16)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_co = [Reshape([8,8,1])(Embedding(sparse_features[sparse_features_names[idx]], 64)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    sparse_emb_8 = [(Embedding(sparse_features[sparse_features_names[idx]], 8)(sparse_input[idx])) for idx in range(len(sparse_features_names))]
    
    if dense_features != None:
        dense_features_names = [k for k in dense_features]
        dense_input = [Input(shape=(dense_features[dense_features_names[x]],), name = 'dense_'+dense_features_names[x]) for x in range(len(dense_features_names))]
        dense_input_reshape = [Reshape([1, dense_features[dense_features_names[x]]])(dense_input[x]) for x in range(len(dense_features_names))]
        dense_emb_1_sing = [Reshape([1,1])(Dense(1)(x)) for x in dense_input if x.shape[1] == 1]
        dense_emb_1_mul = [Reshape([1,1])(Dense(1)(x)) for x in dense_input if x.shape[1] > 1]
        dense_emb_1 = dense_emb_1_sing + dense_emb_1_mul
        
        dense_emb_16 = [Reshape([1,16])(Dense(16)(x)) for x in dense_input if x.shape[1] >= 16]
        
        dense_emb_8 = [Reshape([1,8])(Dense(8)(x)) for x in dense_input]
        dense_emb_co = [Reshape([8,8,1])((Dense(64))(x)) for x in dense_input]
    else:
        dense_emb_1 = []
        dense_input_reshape = []
        dense_emb_16 = []
        dense_emb_8 = []
        dense_emb_co = []
    if tag_feature != None:
        sparse_tag_input = Input(shape=(tag_feature[0],), name = 'sparse_tag')
        sparse_input.append(sparse_tag_input)
        
        tag_emb = Embedding(tag_feature[1], 8, mask_zero = True, input_length = tag_feature[0])(sparse_tag_input)
        tag_emb = Reshape([tag_feature[0], 8, 1])(tag_emb)
        tag_conv1 = Conv2D(16,3,padding = 'same')(tag_emb)
        tag_conv2 = Conv2D(32,3,padding = 'same',strides = 2)(tag_conv1)
        tag_conv3 = Conv2D(16,3,padding = 'same', strides = 2)(tag_conv2)
        tag_flatten = Flatten()(tag_conv3)
        tag_emb_16 = Reshape([1,16])(Dense(16, activation = 'relu')(tag_flatten))
        tag_emb_1 = Reshape([1,1])(Dense(1, activation = 'relu')(tag_flatten))
        
        tag_emb_2 = Embedding(tag_feature[1], 64, mask_zero = True, input_length = tag_feature[0])(sparse_tag_input)
        tag_emb_2 = Reshape([8,8,tag_feature[0]])(tag_emb_2)
        tag_conv1_2 = Conv2D(32,3,padding = 'same')(tag_emb_2)
        tag_conv2_2 = Conv2D(64,3,padding = 'same')(tag_conv1_2)
        tag_conv3_2 = Conv2D(32,3,padding = 'same')(tag_conv2_2)
        tag_flatten_2 = Flatten()(tag_conv3_2)
        tag_emb_16_2 = Reshape([1,16])(Dense(16, activation = 'relu')(tag_flatten_2))
        tag_emb_1_2 = Reshape([1,1])(Dense(1, activation = 'relu')(tag_flatten_2))
        
        sparse_emb_16.append(tag_emb_16)
        sparse_emb_16.append(tag_emb_16_2)
        sparse_emb_1.append(tag_emb_1)
        sparse_emb_1.append(tag_emb_1_2)
        
    if keyword_feature != None:
        sparse_keyword_input = Input(shape=(keyword_feature[0],), name = 'sparse_keyword')
        sparse_input.append(sparse_keyword_input)
        
        keyword_emb = Embedding(keyword_feature[1], 8, mask_zero = True, input_length = keyword_feature[0])(sparse_keyword_input)
        keyword_emb = Reshape([keyword_feature[0], 8, 1])(keyword_emb)
        keyword_conv1 = Conv2D(16,3,padding = 'same')(keyword_emb)
        keyword_conv2 = Conv2D(32,3,padding = 'same',strides = 2)(keyword_conv1)
        keyword_conv3 = Conv2D(16,3,padding = 'same', strides = 2)(keyword_conv2)
        keyword_flatten = Flatten()(keyword_conv3)
        keyword_emb_16 = Reshape([1,16])(Dense(16, activation = 'relu')(keyword_flatten))
        keyword_emb_1 = Reshape([1,1])(Dense(1, activation = 'relu')(keyword_flatten))
        
        keyword_emb_2 = Embedding(keyword_feature[1], 64, mask_zero = True, input_length = keyword_feature[0])(sparse_keyword_input)
        keyword_emb_2 = Reshape([8,8,keyword_feature[0]])(keyword_emb_2)
        keyword_conv1_2 = Conv2D(32,3,padding = 'same')(keyword_emb_2)
        keyword_conv2_2 = Conv2D(64,3,padding = 'same')(keyword_conv1_2)
        keyword_conv3_2 = Conv2D(32,3,padding = 'same')(keyword_conv2_2)
        keyword_flatten_2 = Flatten()(keyword_conv3_2)
        keyword_emb_16_2 = Reshape([1,16])(Dense(16, activation = 'relu')(keyword_flatten_2))
        keyword_emb_1_2 = Reshape([1,1])(Dense(1, activation = 'relu')(keyword_flatten_2))
        
        
        sparse_emb_16.append(keyword_emb_16)
        sparse_emb_16.append(keyword_emb_16_2)
        sparse_emb_1.append(keyword_emb_1)
        sparse_emb_1.append(keyword_emb_1_2)
        
        
#         con_emb_16 = []
    
    con_emb_16 = []
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[0]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[1]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[2]]))
#     con_emb_16.append(Concatenate()([sparse_emb_8[4], sparse_emb_8[3]]))
    
    # 一次项
    fm1_input = sparse_emb_1 + dense_emb_1
    fm1_out = reduce_sum(input_tensor = Concatenate()(fm1_input), axis = -1, keep_dims = True)
    
    # 二次项
    fm2_input = sparse_emb_16 + dense_emb_16 + con_emb_16 + dense_emb_1_sing
    fm2_out = FM()(Concatenate(axis = -1)(fm2_input))
    
    # DNN
    dnn_input = sparse_emb_16 + dense_input_reshape + con_emb_16
    deep_input = Concatenate(axis=-1)(dnn_input)
    flatten_deep = Flatten()(deep_input)
    hid_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(flatten_deep)
    deep_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(hid_deep)
    # DNN Embedding+卷积+DNN
    dnn_conv_input = Concatenate(axis=1)(sparse_emb_16 + dense_emb_16)
    conv_input = Conv2D(32, 3)(Reshape([dnn_conv_input.shape[1],dnn_conv_input.shape[2],1])(dnn_conv_input))
    conv_flatten = Flatten()(conv_input)
    conv_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_flatten)
    conv_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_deep)
    # 卷积Embedding+卷积+DNN
    conv_input = sparse_emb_co + dense_emb_co
    conv_emb = Concatenate(axis = 3)(conv_input)
    conv1 = Conv2D(32,3,padding = 'same')(conv_emb)
    conv_t_flatten = Flatten()(conv1)
    conv_t_deep = DNN(hidden_units=(128,128), activation='relu', l2_reg=0.00001, dropout_rate=0, use_bn=False, seed=1024)(conv_t_flatten)
    conv_t_out = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(conv_t_deep)
    # 结果
    out = Add()([fm1_out, fm2_out, deep_out, conv_out, conv_t_out])
    #     out = tf.sigmoid(out)
    out = PredictionLayer('binary')(out)
    if dense_features != None:
        model = Model(inputs = sparse_input + dense_input,
                     outputs = [out])
    else:
        model = Model(inputs = sparse_input,
                     outputs = [out])
    return model