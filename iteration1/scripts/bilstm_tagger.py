'''
Implements BiLSTM sequence tagger as spaCy pipeline component.
'''

from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

from spacy.pipeline import TrainablePipe
from spacy.language import Language
from spacy.training.example import Example
from spacy.vocab import Vocab
from thinc.model import Model

from scripts.nercro_utils import logger as L

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

    def initialize(self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        L.info('bilstm.initialize')
        # extract ner labels
        for i, ex in enumerate(get_examples()):
            if i == 10: break
            #L.info(f'doc: {ex.reference}')
            #for ent in ex.reference.ents:
            #    L.info(f'{ent}, {ent.label_}')
            #for tok in ex.reference:
            #    L.info(f'token: {tok}, iob: {tok.ent_iob_}, type: {tok.ent_type_}')
        # init labels
        # label == iob + type (lowercased?)
        # init model (output size)

    def predict(self, docs, Doc=None):
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs, Doc=None, *args, **kwargs):
        print('bilstm.set_annotations')
        pass