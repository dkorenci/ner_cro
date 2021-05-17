'''
Implements BiLSTM sequence tagger as spaCy pipeline component.
'''

from spacy.pipeline import TrainablePipe

class BilstmTagger(TrainablePipe):
    def predict(self, docs, Doc=None): pass
    def set_annotations(self, docs, Doc=None, *args, **kwargs): pass