import logging
from spacy.tokens import Token, Doc

logger = logging.getLogger("spacy")

def add_iob_labels(doc:Doc, labels = None):
    '''
    Add tokens' IOB lables to the label set.
    :param labels: set of string IOB labels
    :param doc: Doc with labeled examples
    :return:
    '''
    if labels is None: labels = set()
    for tok in doc:
        labels.add(token_iob_label(tok))
    return labels

def token_iob_label(tok:Token) -> str:
    if tok.ent_iob_ == 'O': iob_lab = tok.ent_iob_
    else: iob_lab = f'{tok.ent_type_}-{tok.ent_iob_}'
    return iob_lab

def create_label_map(labels):
    '''
    Create label by lexicographical sort
    :param labels: set of IOB labels
    :return: map label -> integer index
    '''
    return {l:i for i, l in enumerate(sorted(labels))}

