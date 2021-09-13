import logging
from spacy.tokens import Token, Doc
from spacy.training import Example

L = logging.getLogger("spacy")

def add_ner_labels(ex:Example, labels = None):
    '''
    Add aligned NER labels of the training example to the label set.
    ex.get_aligned_ner() determines the nature of the labels
    :param labels: set of string labels
    :return:
    '''
    if labels is None: labels = set()
    for l in ex.get_aligned_ner(): labels.add(l)
    return labels

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

def create_label_mappings(labels):
    '''
    Index ner labels by lexicographically sorting them,
     return label->index and index->label maps
    :param labels: set of IOB labels
    :return: map label -> integer index, map index -> label
    '''
    slabels = sorted(labels)
    l2i = {l:i for i, l in enumerate(slabels)}
    i2l = {i:l for i, l in enumerate(slabels)}
    return l2i, i2l

def print_doc_info(doc: Doc):
    L.info(f'doc.length: {len(doc)}')
    L.info(f'doc text: {str(doc)}')
    L.info(f'doc tokens:[{"|".join([str(tok) for tok in doc])}]')

def model_debug_fw(model, X, Y):
    '''
    Forward callback for the thinc model debug logger.     
    '''
    L.info(f"Model.name: {model.name})")
    L.info(f"Input - length: {len(X)}, type: {type(X)}")
    for e in X:
        if isinstance(e, Doc): print_doc_info(e)
        else: # generic element info
            L.info(f"  element - length: {len(e)}, type: {type(e)}")

