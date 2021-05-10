"""
Created on May 9, 2021

Core layers built by myself.

@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Embedding, Layer
from tensorflow.keras.regularizers import l2


class MLP(Layer):
    """Multilayer Perceptron Structure.

    Arguments:
      hidden_units: A list like [64, 32, 16]. List of mlp's hidden units.
      activation: A string. Activation function to use.
      dropout_prob: A scalar(0~1). Number of dropout probability.
      use_batch_norm: A boolean. Whether to use batch normalization.

    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(batch_size, ..., hidden_units[-1])`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, hidden_units[-1])`.

    """
    def __init__(self, hidden_units=None, activation='relu', dropout_prob=0., use_batch_norm=True):
        super(MLP, self).__init__()
        self.use_batch_norm = use_batch_norm
        if hidden_units is None:
            hidden_units = [64, 32, 16]
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout_prob)
        self.batch_norm = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        return x


def build_input_embedding_layers(sparse_feature_columns, embed_reg):
    """
    This function can build embedding layers through sparse feature columns.

    :param sparse_feature_columns: A list contains dictionaries of the sparse feature.
    :param embed_reg: A scalar. Regularizer for embedding layers.
    :return: A dictionary of embedding layers like `{'embed_name1': Embedding1, 'embed_name2': Embedding 2,...}`.
    """
    embed_layers = {
        feat['feat_name']: Embedding(input_dim=feat['feat_num'],
                                     input_length=1,
                                     output_dim=feat['embed_dim'],
                                     embeddings_initializer='random_normal',
                                     embeddings_regularizer=l2(embed_reg))
        for i, feat in enumerate(sparse_feature_columns)
    }
    return embed_layers