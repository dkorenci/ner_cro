'''
Implements neural model of the BiLSTM sequence tagger component.
'''

import spacy
from typing import List

from thinc.api import Model, chain, with_array, TensorFlowWrapper
from spacy.ml import CharacterEmbed
from spacy.tokens import Doc
from thinc.types import Floats2d

from tensorflow import keras
from tensorflow.keras import layers

from scripts.nercro_utils import logger as L


# https://spacy.io/usage/processing-pipelines#trainable-components
@spacy.registry.architectures("bilstm.model.v1")
def build_model(tok2vec: Model[List[Doc], List[Floats2d]], layers, width, num_labels) \
        -> Model[List[Doc], List[Floats2d]]:
    return thinc_bilstm_model1(tok2vec, num_labels=num_labels, num_layers=layers, width=width)

def tf_bilstm_model1(num_labels, emb_size = 128, state_size = 32, num_layers = 2):
    # https://keras.io/examples/nlp/bidirectional_lstm_imdb/
    # https://keras.io/examples/nlp/lstm_seq2seq/
    # Input for variable-length (!?) sequences of embeddings
    L.info('tf_bilstm_model1')
    inputs = keras.Input(shape=(None, emb_size), dtype="float64")
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(state_size, return_sequences=True))(inputs)
    for _ in range(1, num_layers):
        x = layers.Bidirectional(layers.LSTM(state_size, return_sequences=True))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_labels, activation="relu"))(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model

def thinc_bilstm_model1(tok2vec: Model[List[Doc], List[Floats2d]], num_labels: int,
                        width: int = 64, num_layers = 2, state_size = 64) \
        -> Model[List[Doc], List[Floats2d]]:
    '''
    :param tok2vec: spacy-compliant token 2 embedding encoder
    :param width: size of the embedding vector
    :param num_layers: number of BiLSTM layers
    :param state_size: size of LSTM state vector
    :return: Model that accepts a batch of Docs and returns a matrix
     with scores, for each token (x-axis), probabilities of NER tag labels.
    '''
    wrapped_tf_model = TensorFlowWrapper(tf_bilstm_model1(num_labels, width, state_size, num_layers))
    #char_embed = CharacterEmbed(width, embed_size, nM, nC, include_static_vectors=False)
    model = chain(tok2vec, with_array(wrapped_tf_model))
    return model

if __name__ == '__main__':
    #tf_bilstm_model1()
    thinc_bilstm_model1()
