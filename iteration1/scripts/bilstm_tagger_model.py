'''
Implements neural model of the BiLSTM sequence tagger component.
'''

from tensorflow import keras
from tensorflow.keras import layers

from thinc.api import chain, with_array, TensorFlowWrapper
from spacy.ml import CharacterEmbed

def tf_bilstm_model1(emb_size = 128, state_size = 32, num_layers = 2, num_tags = 10):
    # Input for variable-length (!?) sequences of embeddings
    inputs = keras.Input(shape=(None, emb_size), dtype="float64")
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(state_size, return_sequences=True))(inputs)
    for _ in range(1, num_layers):
        x = layers.Bidirectional(layers.LSTM(state_size, return_sequences=True))(x)
    outputs = layers.TimeDistributed(layers.Dense(num_tags, activation="relu"))(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model

def thinc_bilstm_model1(width=10, embed_size=20, nM=10, nC=10):
    wrapped_tf_model = TensorFlowWrapper(tf_bilstm_model1())
    char_embed = CharacterEmbed(width, embed_size, nM, nC, include_static_vectors=False)
    model = chain(char_embed, with_array(wrapped_tf_model))
    print(model)

if __name__ == '__main__':
    #tf_bilstm_model1()
    thinc_bilstm_model1()
