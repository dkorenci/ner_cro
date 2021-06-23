import logging

logger = logging.getLogger("spacy")

def add_iob_labels(doc, labels = None):
    '''
    Add tokens' IOB lables to the label set.
    :param labels: set of string IOB labels
    :param doc: Doc with labeled examples
    :return:
    '''
    if labels is None: labels = set()
    for tok in doc:
        if tok.ent_iob_ == 'O': iob_lab = tok.ent_iob_
        else: iob_lab = f'{tok.ent_type_}-{tok.ent_iob_}'
        labels.add(iob_lab)
    return labels

def create_label_map(labels):
    '''
    Create label by lexicographical sort
    :param labels: set of IOB labels
    :return: map label -> integer index
    '''
    return {i:l for i, l in enumerate(sorted(labels))}

