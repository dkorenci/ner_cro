'''
Implements BiLSTM sequence tagger as spaCy pipeline component.
'''

from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any
import numpy as np

from spacy.pipeline import TrainablePipe
from spacy.language import Language
from spacy.training.example import Example
from spacy.vocab import Vocab
from spacy.tokens.doc import Doc
from thinc.model import Model
from thinc.types import Floats2d

from scripts.nercro_utils import *

@Language.factory("bilstm_tagger")
def bilstm_tagger(nlp, name, model):
   return BilstmTagger(nlp.vocab, model, name)

class BilstmTagger(TrainablePipe):

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        **cfg,
    ) -> None:
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = cfg
        self.cfg['labels'] = []

    @property
    def labels(self) -> Tuple[str]:
        return tuple(self.cfg["labels"])

    def initialize(self, get_examples: Callable[[], Iterable[Example]], *,
            nlp: Language = None, labels: Optional[List[str]] = None):
        from itertools import islice
        L.info('bilstm.initialize')
        # extract ner labels
        labels = set()
        for i, ex in enumerate(get_examples()):
            add_iob_labels(ex.reference, labels)
        self._labels = labels
        self._label_map = create_label_map(labels)
        L.info(self._labels)
        L.info(self._label_map)
        sample = list(islice(get_examples(), 10))
        docsample = [ex.reference for ex in sample]
        labels = self._examples_to_ner_labels(sample)
        self.model.initialize(X=docsample, Y=labels)

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        L.info('predict')
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs, Doc=None, *args, **kwargs):
        L.info('set_annotations')
        pass

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        from thinc.api import CategoricalCrossentropy
        loss_calc = CategoricalCrossentropy()
        truth = self._examples_to_ner_labels(examples, debug=True)
        scores_flat = self.model.ops.flatten(scores)
        L.info(f'truth.shape {truth.shape}')
        L.info(f'scores.shape {scores_flat.shape}')
        return loss_calc(scores_flat, truth)

    def _examples_to_ner_labels(self, examples: List[Example], debug:bool = False) \
            -> Optional[np.ndarray]:
        '''
        Convert reference documents in the examples to matrix of one-hot NER labels
        '''
        if len(examples) == 0: return None
        labels = [self._document_ner_labels(ex.reference, debug) for ex in examples]
        return np.concatenate(labels)

    def _document_ner_labels(self, doc:Doc, debug:bool = False) -> Optional[np.ndarray]:
        '''
        Convert a document with NER annotations into a matrix of one-hot per-token labels
        '''
        if debug: print_doc_info(doc)
        labels = np.zeros((len(doc), len(self._labels)), float)
        for i, tok in enumerate(doc):
            labels[i, self._label_map[token_iob_label(tok)]] = 1.0
        return labels