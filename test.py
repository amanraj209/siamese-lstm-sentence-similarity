import os

import pandas as pd
import tensorflow as tf

from utils import make_w2v_embeddings, split_and_zero_padding

ROOT_PATH = os.path.abspath('')
TEST_CSV = 'data/test-20.csv'


def prepare_data():
    test_df = pd.read_csv(os.path.join(TEST_CSV))
    for q in ['question1', 'question2']:
        test_df[q + '_n'] = test_df[q]

    embedding_dim = 300
    max_seq_length = 25
    test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim)

    X_test = split_and_zero_padding(test_df, max_seq_length)

    assert X_test['q1'].shape == X_test['q2'].shape

    return X_test


def test_data(X_test):
    model = tf.keras.models.load_model(os.path.join(ROOT_PATH, 'model/siamese-lstm-weights.h5'))
    model.summary()

    prediction = model.predict([X_test['q1'], X_test['q2']])
    print(prediction)

    with open(os.path.join(ROOT_PATH, 'model/predictions.txt'), 'w') as f:
        f.write(prediction)
        f.close()


if __name__ == '__main__':
    X_test = prepare_data()
    test_data(X_test)
