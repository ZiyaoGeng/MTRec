"""
Created on May 9, 2021

model: MMoE
paper: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts

@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from collections import OrderedDict
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

from mtrec.layers.core import build_input_embedding_layers, MLP


def MMoE(task_names, num_experts, sparse_feature_columns, expert_units=None, gate_units=None, tower_units=None,
         expert_activation='relu', expert_dropout=0., expert_use_bn=True, gate_activation='relu', gate_dropout=0.,
         gate_use_bn=True, tower_activation='relu', tower_dropout=0., tower_use_bn=True, embed_reg=1e-6):
    """Multi-gate Mixture-of-Experts model.
    The model can solve the problem of multi-task learning (binary classification),
    and the Input must be the sparse features.

    :param task_names: A list like `[task1, task2,...]`. List of task name.
    :param num_experts: A scalar. Number of experts.
    :param sparse_feature_columns: A list contains dictionaries of the sparse feature.
    :param expert_units: A list. List of mlp's hidden units of the expert module.
    :param gate_units: A list. List of mlp's hidden units of the gate module.
    :param tower_units: A list. List of mlp's hidden units of the tower module.
    :param expert_activation: A string. Activation function of the expert module.
    :param expert_dropout: A scalar. Number of dropout of the expert module.
    :param expert_use_bn: A boolean. Whether to use batch normalization on the expert module.
    :param gate_activation: A string. Activation function of the gate module.
    :param gate_dropout: A scalar. Number of dropout of the gate module.
    :param gate_use_bn: A boolean. Whether to use batch normalization on the gate module.
    :param tower_activation: A string. Activation function of the tower module.
    :param tower_dropout: A scalar. Number of dropout of the tower module.
    :param tower_use_bn: A boolean. Whether to use batch normalization on the tower module.
    :param embed_reg: A scalar. Regularizer for embedding layers.
    :return: MMoE Model
    """
    feature_layer = OrderedDict()
    for feat in sparse_feature_columns:
        feature_layer[feat['feat_name']] = Input(shape=(1,), dtype=tf.int32, name=feat['feat_name'])
    # embedding
    embed_layers = build_input_embedding_layers(sparse_feature_columns, embed_reg)
    sparse_embed = tf.concat([
        embed_layers[feat_name](tf.squeeze(val, axis=-1)) for feat_name, val in feature_layer.items()], axis=-1)
    # multi gates module
    multi_gates = []
    for _ in range(len(task_names)):
        # We use the MLP structure, can also be replaced by other modules.
        hidden = MLP(gate_units, gate_activation, gate_dropout, gate_use_bn)(sparse_embed)
        if hidden.shape[-1] != num_experts:
            hidden = Dense(num_experts, activation='relu')(hidden)
        gate = tf.nn.softmax(hidden)
        multi_gates.append(gate)  # (num_task, batch_size, num_experts)
    # multi experts module
    multi_experts = []
    for _ in range(num_experts):
        # We use the MLP structure, can also be replaced by other modules.
        expert = MLP(expert_units, expert_activation, expert_dropout, expert_use_bn)(sparse_embed)
        multi_experts.append(expert)  # (num_experts, batch_size, last_hidden_unit)
    multi_experts = tf.transpose(tf.convert_to_tensor(multi_experts), perm=[1, 0, 2])
    final_outputs = []
    # multi towers module
    for idx, gate in enumerate(multi_gates):
        expand_gate = tf.expand_dims(gate, axis=-1)
        weight_experts = tf.reduce_sum(multi_experts * expand_gate, axis=1)  # (batch_size, last_hidden_unit)
        # We use the MLP structure, can also be replaced by other modules.
        output = MLP(tower_units, tower_activation, tower_dropout, tower_use_bn)(weight_experts)
        output = Dense(1, activation="sigmoid", name=task_names[idx])(output)  # (batch_size, 1)
        final_outputs.append(output)  # (batch_size, num_task)
    # model
    model = Model(inputs=feature_layer, outputs=final_outputs)
    return model


def test():
    fea_col = [{'feat_name': 'item_id', 'feat_num': 1000, 'embed_dim': 16},
               {'feat_name': 'item_cate_id', 'feat_num': 200, 'embed_dim': 8}]
    task_names = ['task1', 'task2']
    num_experts = 3
    model = MMoE(task_names, num_experts, fea_col)
    model.summary()
    tf.keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
