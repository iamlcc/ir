import re
import numpy as np
import operator as op

from scipy.sparse import csc_matrix
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def retrieve(SearchCore, query, top):
    # assume the query has only unique terms
    # TO DO: improve the query preprocessing

    vq = vectorize_query(SearchCore, query)

    vsm = retrieve_vsm(SearchCore, 
                       tdm_row_indices = vq[0],
                       idfs            = vq[1],
                       qterms          = vq[2], top=top)

    lsi = retrieve_lsi(SearchCore,
                       tdm_row_indices = vq[0],
                       idfs            = vq[1], top = top)

    return [vsm, lsi]

def vectorize_query(SearchCore, query):
    """
    Vectorizes the query.
    
    Parameters
    ----------
    SearchCore : SearchCore object
        'tdmatrix', 'u', 'sigmavt', 'vocabulary'
    query : string
        single string wit the query
    """

    # do the same pre-processing as in the VecDocs
    sc_removed = re.sub(r'[^a-zA-Z\'\s]+', ' ', query)
    terms = sc_removed.strip().lower().split()
    for i, t in enumerate(terms):
        terms[i] = ps.stem(t)

    tdm_row_indices = []
    idfs            = []
    qterms          = []
    for t in terms:

        if t not in SearchCore.vocabulary:
            continue

        tdm_row_indices.append(SearchCore.vocabulary[t][0])
        idfs.append(SearchCore.vocabulary[t][1])
        qterms.append(t)

    if len(tdm_row_indices) != len(idfs):
        raise Exception("the query was not vectorized properly")

    return [tdm_row_indices, idfs, qterms]

def retrieve_lsi(SearchCore, tdm_row_indices, idfs, top):

    # if the none of the query terms is in the vocabulary
    if len(tdm_row_indices) < 1:
        return []

    urows = SearchCore.u[tdm_row_indices, ]
    dense_q = np.dot(np.transpose(urows), idfs)

    fake_cosines = np.dot(dense_q, SearchCore.sigmavt)
    fake_cosines = fake_cosines / SearchCore.sigmavt_col_norms

    ranking = op.itemgetter(*np.argsort(-fake_cosines))(SearchCore.docpaths)

    return ranking[0:top]

def retrieve_vsm(SearchCore, tdm_row_indices, idfs, qterms, top):

    if len(tdm_row_indices) < 1:
        return []

    # get the relevant columns from the term-document matrix
    # it does not make sense to calculate cosines for document
    # which do not contain any of the query terms
    # the relevant rows are extracted by 'vectorize_query'
    a_cols = set()
    for t in qterms:
        a_cols = a_cols.union(set(SearchCore.vocabulary[t][2]))
    a_cols = np.array(list(a_cols))

    a_submatrix = (SearchCore.tdmatrix[tdm_row_indices, ].tocsc()
                   [:, a_cols].todense())

    fake_cosines = np.dot(idfs, a_submatrix)
    fake_cosines = fake_cosines / SearchCore.tdmatrix_col_norms[a_cols]

    o = np.argsort(-fake_cosines)
    ranking = op.itemgetter(*np.squeeze(a_cols[o]).tolist())(SearchCore.docpaths)

    return ranking[0:top]
