import spacy
from spacy.language import Language
from thinc.api import Model, chain

from typing import List, Tuple, Callable
from spacy.tokens import Doc, Span
from thinc.types import Floats2d, Ints1d, Ragged, cast

#from iteration1.scripts.bilstm_tagger import BilstmTagger
from scripts.bilstm_tagger import bilstm_tagger
#from iteration1.scripts.bilstm_tagger_model import *
from scripts.bilstm_tagger_model import *

# https://spacy.io/usage/processing-pipelines#trainable-components

@spacy.registry.architectures("bilstm.model.v1")
def build_model(tok2vec: Model[List[Doc], List[Floats2d]], layers, width) \
        -> Model[List[Doc], List[Floats2d]]:
    return thinc_bilstm_model1(tok2vec, num_layers=layers, width=width)
