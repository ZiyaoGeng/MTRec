import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

from mtrec.models import MMoE
from utils import build_census

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    """
        ========================= Hyper Parameters =======================
    """
    train_file = 'data/census/census-income.data.gz'
    test_file = 'data/census/census-income.test.gz'

    embed_dim = 4
    num_experts = 8
    expert_units = [16, 8]
    gate_units = [16, 8, 2]
    tower_units = [8]

    epochs = 10
    batch_size = 1024
    learning_rate = 0.001
    """
        ========================= Create dataset =======================
    """
    sparse_feature_columns, train, test = build_census(train_file, test_file, embed_dim)
    train_X, train_y = train
    test_X, test_y = test
    task_names = list(train_y.keys())
    """
        ========================= Build Model =======================
    """
    model = MMoE(task_names, num_experts, sparse_feature_columns, expert_units,
                 gate_units, tower_units)
    model.summary()
    # tf.keras.utils.plot_model(model, "model_with_shape_info.png", show_shapes=True)
    """
        ============================Compile============================
    """
    model.compile(loss={'income': 'binary_crossentropy', 'marital': 'binary_crossentropy'},
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    """
        ==============================Fit==============================
    """
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_marital_loss', patience=2, restore_best_weights=True)],  # checkpoint,
        batch_size=batch_size,
        validation_split=0.1
    )
    """
        ===========================Test==============================
    """
    test_metric = model.evaluate(test_X, test_y, batch_size=batch_size)
    print('test income AUC: %f, marital AUC: %f' % (test_metric[3], test_metric[4]))


if __name__ == '__main__':
    main()
