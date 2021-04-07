'''
Prepare the hr500k corpus of annotated sentences to be used by spaCy
'''

import spacy
from spacy.tokens import Doc, DocBin
from pathlib import Path

hr500k_path = '/data/resources/cro.lang.resources/hr500k.conll/hr500k.conll'
hr500k_trunc_path = '/data/resources/cro.lang.resources/hr500k.conll/hr500k.trunc.conll'

def analyze_hr500k(path=hr500k_path):
    '''
    Print properties of the data hr500k
    '''
    numfields = set(); # numbers of fields per record
    maxnes = set() # indexes of maximum non-empty fields
    nertags = set() # NER token tags
    for line in open(path, 'r').readlines():
        l = line.strip()
        if l.startswith('#') or l == '': # comment or empty line
            pass
        else:
            fields = l.split('\t')
            maxne = max(i for i, f in enumerate(fields) if f != '_')
            numfields.add(len(fields)); maxnes.add(maxne)
            nertags.add(fields[10])
    print(f'field lengths: {numfields}')
    print(f'non-empty fields: {maxnes}')
    nertags = sorted(nertags, key=lambda tag:tag[::-1])
    print(f'ner tages: {nertags}')

def trunc_hr500k(inpath=hr500k_path, outpath=hr500k_trunc_path):
    '''
    Truncate rows in hr500k, to 10 standard CoNLL-U fields,
     move 11th field (NER tags) to 10th 'misc' field.
    '''
    outf = open(outpath, 'w')
    for line in open(inpath, 'r').readlines():
        l = line.strip()
        if l.startswith('#') or l == '': # comment or empty line
            outf.write(line)
        else:
            fields = l.split('\t')
            trunc_fields = fields[:10]
            # empty fields that are not basic data, lemma or pos
            for i in range(5, 10): trunc_fields[i] = '_'
            trunc_fields[9] = fields[10] # copy NER tag
            outf.write('\t'.join(trunc_fields)+'\n')

class Sentence():
    def __init__(self, words=[], lemmas=None, pos=None, ner=None):
        self.words = words; self.lemmas = lemmas; self.pos = pos; self.ner = ner

def read_hr500k(inpath=hr500k_path):
    '''
    Read sentences as lists of words and word-level annotations.
    :return: list of Sentence objects
    '''
    sentences = []
    sentence_started = False
    for line in open(inpath, 'r').readlines():
        l = line.strip()
        if l.startswith('#') or l == '': # comment or empty line
            if sentence_started: # ending a sentence
                # assuption is that there is an empty line after last sentence
                sent = Sentence(words, lemmas, pos, ner)
                sentences.append(sent)
                sentence_started = False
        else:
            if not sentence_started: # starting new sentence
                words, lemmas, pos, ner = [], [], [], []
                sentence_started = True
            fields = l.split('\t')
            word, lemma, postag, nertag = fields[1], fields[2], fields[3], fields[10]
            words.append(word); lemmas.append(lemma);
            pos.append(postag); ner.append(nertag)
    return sentences

def create_spacy_corpus(inpath=hr500k_path, outpath='corpus',
                        train=0.8, test=0.1, rseed=88103):
    '''
    Create spacy data in DocBin format.
    Create train/dev/test splits according to proportion params.
    :param outpath: save splits to this folder
    '''
    from random import seed, shuffle
    # load data and create splits
    sents = read_hr500k(inpath)
    N = len(sents)
    print(f'{N} sentences in total')
    train_sz = int(N*train); test_sz = int(N*test); dev_sz = N - train_sz - test_sz
    print(f'splitting into train({train_sz}), dev({dev_sz}), test({test_sz})')
    seed(rseed); shuffle(sents)
    train_set = sents[:train_sz]; assert(train_sz==len(train_set))
    test_set = sents[train_sz:train_sz+test_sz]; assert(test_sz==len(test_set))
    dev_set = sents[train_sz+test_sz:]; assert(dev_sz==len(dev_set))
    # save each split as DocBin of individual sentences
    nlp = spacy.blank("hr")
    splits = {'train':train_set, 'dev':dev_set, 'test':test_set}
    outpath = Path(outpath)
    for name, sents in splits.items():
        docbin = DocBin()
        print(f'processing {name} ...')
        for i, s in enumerate(sents):
            #sent = Sentence(words, lemmas, pos, ner)
            spaces = [True]*(len(s.words)-1)+[False]
            assert(len(s.words) == len(spaces))
            doc = Doc(nlp.vocab, words=s.words, spaces=spaces,
                      lemmas=s.lemmas, pos=s.pos, ents=s.ner)
            docbin.add(doc)
            if ((i+1)%1000 == 0): print(f'.. processed {i+1} sentences')
        docbin.to_disk(outpath / f'{name}.spacy')
        print(f'{name} done.')

if __name__ == '__main__':
    analyze_hr500k()
    #trunc_hr500k()
    #create_spacy_corpus()