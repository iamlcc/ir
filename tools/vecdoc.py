import os
import re
import math
from nltk.stem.porter import PorterStemmer

HOME = os.getcwd()

ps = PorterStemmer()

# read the stopwords
# sw is a module-global variable (sw is not reassigned on each
# instantiation of VecDoc)
sw = set()
swpath = os.path.join(HOME, "data/rawdata/stopwords/stopwords.txt")
with open(swpath, 'r', encoding='utf-8') as sw_file:
    for line in sw_file:
        sw.add(line[:-1]) # remove the '\n' at the end

class VecDoc:
    """
    Class for representing vectorized version of a text file.
    
    Parameters
    ----------
    docpath : string
        absolute path to the text file which has to be vectorized 
    log_weight : boolean
        whether or not the raw frequencies should be log weighted
    """
    def __init__(self, docpath, log_weight):

        self.docpath = docpath
        self.log_weight = log_weight

        self.term_weight = {}
        self.doclen = 0
        self.max_termfreq = 0

        self.vectorize()


    def vectorize(self):

        with open(self.docpath, 'r', encoding='windows-1251') as rawdoc:
            # replace special chars with space
            sc_removed = re.sub(r'[^a-zA-Z\'\s]+', ' ', rawdoc.read())

        # trim, lower case, split by whitespace
        tokens = sc_removed.strip().lower().split()
        # Stem the tokens, slows down quite a lot
        for i, t in enumerate(tokens):
            tokens[i] = ps.stem(t)

        # "table" the tokens, do not include stopwords
        for t in tokens:
            if t in sw:
                continue
            if t in self.term_weight:
                self.term_weight[t] = self.term_weight[t] + 1
            else:
                self.term_weight[t] = 1

            # find the frequency of the most frequent term
            if self.term_weight[t] > self.max_termfreq:
                self.max_termfreq = self.term_weight[t]

        self.doclen = len(self.term_weight)

        # if the max_termfreq is 1 it doesn't make sense to loop and
        # recalculate since all frequencies are equal to 1
        if self.log_weight and self.max_termfreq > 1:
            for term, weight in self.term_weight.items():
                # take the log from the raw freqs and also divide by the
                # frequency of the most frequent term
                self.term_weight[term] = ((1 + math.log10(weight)) /
                                          (1 + math.log10(self.max_termfreq)))
