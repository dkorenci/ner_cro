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
        from scripts.nercro_utils import add_iob_labels, create_label_map
        L.info('bilstm.initialize')
        # extract ner labels
        labels = set()
        for i, ex in enumerate(get_examples()):
            add_iob_labels(ex.reference, labels)
            #L.info(f'doc: {ex.reference}')
            #for ent in ex.reference.ents:
            #   L.info(f'{ent}, {ent.label_}')
        self._labels = labels
        self._label_map = create_label_map(labels)
        L.info(self._labels)
        L.info(self._label_map)

    def predict(self, docs, Doc=None):
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs, Doc=None, *args, **kwargs):
        print('bilstm.set_annotations')
        pass