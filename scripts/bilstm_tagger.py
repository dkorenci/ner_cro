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
            #add_iob_labels(ex.reference, labels)
            add_ner_labels(ex, labels)
        self._labels = labels
        self._label_map, self._inv_label_map = create_label_mappings(labels)
        L.info(self._labels)
        L.info(self._label_map)
        sample = list(islice(get_examples(), 10))
        docsample = [ex.reference for ex in sample]
        labels = self._examples_to_ner_labels(sample)
        self.model.initialize(X=docsample, Y=labels)

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs, scores, Doc=None, *args, **kwargs):
        from spacy.training import biluo_tags_to_spans
        L.info('set_annotations')
        ti = 0 # token index
        for doc in docs:
            biluo_tags = []
            for tok in doc:
                ner_tag_probs = scores[ti]; ti += 1
                maxi = np.argmax(ner_tag_probs)
                ner_label = self._inv_label_map[maxi]
                biluo_tags.append(ner_label)
            try:
                spans = biluo_tags_to_spans(doc, biluo_tags)
            except ValueError as error:
                L.info(f'biluo_tags_to_spans error: {error}')
                spans = []  # fallback to no entities if bilou tags as wrongly formatted
            try:
                doc.set_ents(spans)
            except ValueError as error:
                L.info(f'doc.set_ents error: {error}')
                doc.set_ents([]) # fallback to no entities
            if doc.ents:
                L.info(f'doc.text: {doc.text}')
                L.info(f'doc.ents: {doc.ents}')
                L.info(','.join([t.ent_iob_+('-'+t.ent_type_ if t.ent_type_ else '') for t in doc]))

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        from thinc.api import CategoricalCrossentropy
        loss_calc = CategoricalCrossentropy()
        truth = self._examples_to_ner_labels(examples, debug=False)
        grad, loss = loss_calc(scores, truth)
        return float(loss), grad

    def _examples_to_ner_labels(self, examples: List[Example], debug:bool = False) \
            -> Optional[np.ndarray]:
        '''
        Convert reference documents in the examples to matrix of one-hot NER labels
        '''
        if len(examples) == 0: return None
        #labels = [self._document_ner_labels(ex.reference, debug) for ex in examples]
        labels = [self._example_ner_labels_aligned(ex, debug) for ex in examples]
        return np.concatenate(labels)

    def _example_ner_labels_aligned(self, ex: Example, debug:bool = False) \
            -> Optional[np.ndarray]:
        '''
        Convert single example to ground-truth one-hot NER labels, using ex.get_aligned_ner()
        to get BILOU tags of ex.predicted, aligned to ex.reference
        '''
        aner = ex.get_aligned_ner()
        if debug:
            L.info('aligned ner:')
            L.info(aner)
        # without explicitly setting dtype="float32", there is a type mismatch error
        labels = np.zeros((len(aner), len(self._labels)), dtype="float32")
        for i, nerLabel in enumerate(ex.get_aligned_ner()):
            labels[i, self._label_map[nerLabel]] = 1.0
        return labels

    def _document_ner_labels(self, doc:Doc, debug:bool = False) -> Optional[np.ndarray]:
        '''
        Convert a document with NER annotations into a matrix of one-hot per-token labels
        '''
        if debug: print_doc_info(doc)
        labels = np.zeros((len(doc), len(self._labels)), float)
        for i, tok in enumerate(doc):
            labels[i, self._label_map[token_iob_label(tok)]] = 1.0
        return labels