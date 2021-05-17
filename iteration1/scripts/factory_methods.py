import spacy
from spacy.language import Language
from thinc.api import Model, chain

from typing import List, Tuple, Callable
from spacy.tokens import Doc, Span
from thinc.types import Floats2d, Ints1d, Ragged, cast

#from iteration1.scripts.bilstm_tagger import BilstmTagger
from scripts.bilstm_tagger import BilstmTagger
#from iteration1.scripts.bilstm_tagger_model import *
from scripts.bilstm_tagger_model import *

# https://spacy.io/usage/processing-pipelines#trainable-components

@Language.factory("bilstm_tagger")
def bilstm_tagger(nlp, name, model):
   return BilstmTagger(nlp.vocab, model, name)

@spacy.registry.architectures("bilstm.model.v1")
def build_model(layers: int, tok2vec: Model[List[Doc], List[Floats2d]]
                ) -> Model[List[Doc], List[List[Floats2d]]]:
    return Model(name="mockmodel", forward=lambda x:x)
