import os
import matplotlib

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from time import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Input, Embedding, LSTM

from utils import make_w2v_embeddings, split_and_zero_padding, ManhattanDistance

matplotlib.use('Agg')

ROOT_PATH = os.path.abspath('')
TRAIN_CSV = 'data/train.csv'

gpus = 2
batch_size = 256 * gpus
n_epoch = 50
n_hidden = 256
embedding_dim = 300
max_seq_length = 25


def prepare_data():
    train_df = pd.read_csv(os.path.join(ROOT_PATH, TRAIN_CSV))
    for q in ['question1', 'question2']:
        train_df[q + '_n'] = train_df[q]

    train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=embedding_dim)
    validation_size = int(len(train_df) * 0.1)

    X = train_df[['question1_n', 'question2_n']]
    Y = train_df['is_duplicate']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    X_train = split_and_zero_padding(X_train, max_seq_length)
    X_validation = split_and_zero_padding(X_validation, max_seq_length)

    Y_train = Y_train.values
    Y_validation = Y_validation.values

    assert X_train['q1'].shape == X_train['q2'].shape
    assert len(X_train['q1']) == len(Y_train)

    return X_train, X_validation, Y_train, Y_validation, embeddings


def prepare_model(embeddings):
    shared_model = Sequential()
    shared_model.add(Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
    shared_model.add(LSTM(n_hidden))

    q1_input = Input(shape=(max_seq_length,), dtype='int32')
    q2_input = Input(shape=(max_seq_length,), dtype='int32')

    malstm_distance = ManhattanDistance()([shared_model(q1_input), shared_model(q2_input)])
    model = Model(inputs=[q1_input, q2_input], outputs=[malstm_distance])

    # if gpus >= 2:
    #     model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)

    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    shared_model.summary()

    return model


def train_model(X_train, X_validation, Y_train, Y_validation, model):
    training_start_time = time()

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(ROOT_PATH, 'model/weights.{epoch:02d}.h5'),
        verbose=1, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', period=1)
    
    malstm_trained = model.fit(
        [X_train['q1'], X_train['q2']], Y_train, batch_size=batch_size, epochs=n_epoch,
        validation_data=([X_validation['q1'], X_validation['q2']], Y_validation), 
        callbacks=[checkpointer])
    
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))

    model.save(os.path.join(ROOT_PATH, 'model/SiameseLSTM.h5'))

    return malstm_trained


def plot_accuracy_and_loss(malstm_trained):
    plt.subplot(211)
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig(os.path.join(ROOT_PATH, 'model/history-graph.png'))

    print(str(malstm_trained.history['val_acc'][-1])[:6] + "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")


if __name__ == '__main__':
    X_train, X_validation, Y_train, Y_validation, embeddings = prepare_data()
    model = prepare_model(embeddings)
    # malstm_trained = train_model(X_train, X_validation, Y_train, Y_validation, model)
    malstm_trained = model.load_weights(os.path.join(ROOT_PATH, 'model/siamese-lstm-weights.h5'))
    plot_accuracy_and_loss(malstm_trained)
