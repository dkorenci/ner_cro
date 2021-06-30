'''
Implements neural model of the BiLSTM sequence tagger component.
'''

import spacy
from typing import List

from thinc.api import Model, chain, TensorFlowWrapper, PyTorchLSTM
from thinc.layers import list2padded, list2array, with_padded, Softmax, with_list, with_debug
from thinc.api import strings2arrays, with_array

from spacy.ml import CharacterEmbed
from spacy.tokens import Doc
from thinc.types import Floats2d

from tensorflow import keras
from tensorflow.keras import layers

from scripts.nercro_utils import L as L, model_debug_fw


# https://spacy.io/usage/processing-pipelines#trainable-components
@spacy.registry.architectures("bilstm.model.v1")
def build_model(tok2vec: Model[List[Doc], List[Floats2d]], layers, num_labels, emb_width, lstm_width) \
        -> Model[List[Doc], List[Floats2d]]:
    return thinc_bilstm_torch(tok2vec, num_labels=num_labels, emb_width=emb_width,
                              lstm_width=lstm_width, num_layers=layers, debug=True)

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

def thinc_bilstm_tf(tok2vec: Model[List[Doc], List[Floats2d]], num_labels: int,
                    width: int = 64, num_layers = 2, state_size = 64) \
        -> Model[List[Doc], List[Floats2d]]:
    '''
    Create BiLSTM model based on TF implementation
    :param tok2vec: spacy-compliant token 2 embedding encoder
    :param width: size of the embedding vector
    :param num_layers: number of BiLSTM layers
    :param state_size: size of LSTM state vector
    :return: Model that accepts a batch of Docs and returns a matrix
     with scores, for each token (x-axis), probabilities of NER tag labels.
    '''
    wrapped_tf_model = TensorFlowWrapper(tf_bilstm_model1(num_labels, width, state_size, num_layers))
    #char_embed = CharacterEmbed(width, embed_size, nM, nC, include_static_vectors=False)
    #model = chain(tok2vec, list2padded(), wrapped_tf_model, padded2list())
    model = chain(tok2vec, with_padded(wrapped_tf_model))
    return model

def thinc_bilstm_torch(tok2vec: Model[List[Doc], List[Floats2d]], num_labels: int, emb_width:int,
                       lstm_width: int = 64, num_layers: int = 2, debug:bool = False) \
    -> Model[List[Doc], List[Floats2d]]:
    '''
    Create a BiLSTM model based on pyTorch implementation
    :return:
    '''
    bilstm = PyTorchLSTM(nO=lstm_width*2, nI=emb_width, bi=True, depth=num_layers)
    classification = Softmax(nO=num_labels, nI=lstm_width)
    if debug:
        model = chain(with_debug(tok2vec, on_forward=model_debug_fw),
                  with_debug(bilstm, on_forward=model_debug_fw),
                  with_debug(with_array(classification), on_forward=model_debug_fw))
    else:
        model = chain(tok2vec, bilstm, with_array(classification))
    return model

if __name__ == '__main__':
    #tf_bilstm_model1()
    thinc_bilstm_tf()
