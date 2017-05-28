import math
import pickle
import os
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from tools import vecdoc

HOME = os.getcwd()

class SearchCore:
    """
    Parameters
    ----------
    docpaths : list
        list of absolute paths to all files from the collection
    log_weight : boolean
        whether or not the raw frequencies should be log weighted
    min_freq: integer
        minimum document frequency

    Returns
    -------
    LSICore
    
    """
    def __init__(self, docpaths, log_weight, min_freq, k):

        self.docpaths = docpaths
        self.log_weight = log_weight
        self.min_freq = min_freq
        self.N = len(docpaths)
        self.svdk = k

        self.vecdoc_list = [] # hmm, think about this
        self.vocabulary_full = {}
        self.vocabulary = {}
        self.tdmatrix = None
        self.svd = None

        self.build_vecdoc_list()
        self.build_vocabulary()
        self.build_tdmatrix()

    def build_vecdoc_list(self):
        for docpath in self.docpaths:
            self.vecdoc_list.append(vecdoc.VecDoc(docpath, self.log_weight))
        print("vecdoc_list_ready")


    # get 'tdm_row_index', 'document frequency' and 'doc_indices' which contain
    # it for each term: ['tdm_row_index, 'df', [doc_indices]]
    def build_vocabulary(self):
        counter = 0
        for docindex, vdoc in enumerate(self.vecdoc_list):
            for term in vdoc.term_weight.keys():
                if term not in self.vocabulary_full:
                    self.vocabulary_full[term] = [counter, 1, [docindex]]
                    counter += 1
                else:
                    # if the current term is in 'vocabulary_full', increment
                    # and append the additional doc_index
                    self.vocabulary_full[term][1] += 1
                    self.vocabulary_full[term][2].append(docindex)
        # filter the term with document frequency less than 'min_freq'
        self.vocabulary = {k: v for k, v in self.vocabulary_full.items() 
                                 if v[1] > self.min_freq}

        # create new indices for the terms to remove "holes" caused by the
        # filtering
        i = 0
        for k, v in self.vocabulary.items():
            v[0] = i
            i += 1

        # convert the document frequencies to inverse document frequencies
        # the raw frequencies are still available ('len(vocabulary[2])'')
        for term, v in self.vocabulary.items():
            self.vocabulary[term][1] = math.log10(self.N / v[1])

        print("vocabulary ready")

    # build the term-document matrix, it is quite sparse so it has
    # to be stored as sparse.csr_matrix
    def build_tdmatrix(self):

        i = []
        j = []
        v = []

        if len(self.vocabulary_full) < 1:
            raise Exception("build the full vocabulary first")
        
        # loop through the list of VecDocs and through their term_weight
        # dicts to build the sparse term-document matrix
        for docindex, vdoc in enumerate(self.vecdoc_list):
            for term, weight in vdoc.term_weight.items():

                # skip the term which were filtered
                if term not in self.vocabulary:
                    continue

                # Add the current i (the row index for the current term)
                i.append(self.vocabulary[term][0])

                # Add the current j (the column index for the current term)
                j.append(docindex)
                
                # Add the current value ()
                cv = weight * self.vocabulary[term][1]
                v.append(cv)

        self.tdmatrix = csr_matrix((v, (i, j)))

        self.tdmatrix_col_norms = np.squeeze(np.asarray(
                                  self.tdmatrix.power(2).sum(0)))
        self.tdmatrix_col_norms = np.sqrt(self.tdmatrix_col_norms)
        print("term-document matrix ready")

        self.svd = svds(self.tdmatrix, k=self.svdk)
        self.u = self.svd[0]
        self.sigmavt           = np.dot(np.diag(self.svd[1]), self.svd[2])
        self.sigmavt_col_norms = np.sqrt(np.sum(self.sigmavt ** 2, 0))
        print("svd ready")

    def export(self):
        # do not export the vecdoc_list, only wastes memory
        self.vecdoc_list = None
        self.svd = None
        with open("/home/boyangeor/datasets/SearchCore.pkl", 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    def summary(self):
        print("vocabulary_full: ", len(self.vocabulary_full))
        print("vocabulary: ", len(self.vocabulary))
        print("docs : ", self.N)
        print("entries: ", self.tdmatrix.nnz)
