from itertools import chain
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import pickle
import sys, os
import copy
import itertools
import operator
from stemming import stemmer, naive_stemmer
from pprint import pprint

DEPTH = 4 # Computed: 2 and 4
STEMMING = stemmer.stem # False / naive_stemmer / stemmer.stem


path = f"depth{DEPTH}"
path2 = "nostemming" if not STEMMING else "naive_stemming" if STEMMING=="naive" else "stemming"

try:
    with open(os.path.join(path, path2, "tfidf.dict"),'rb') as f:
        tfidf = pickle.load(f)

    with open(os.path.join(path, path2, "tokInfo.dict"),'rb') as f:
        tokInfo = pickle.load(f)

    with open(os.path.join(path, "pageRank.dict"),'rb') as f:
        pageRankDict = pickle.load(f)
except FileNotFoundError:
    sys.exit("This depth has not been computed yet!")


def normalize(vector: dict) -> dict:
    """ vector must be a dictionnary with numerical values. """
    vec = np.array(list(vector.values()))
    if np.linalg.norm(vec, 2)!=0.0:
        normvec = vec / np.linalg.norm(vec, 2)
    else:
        normvec = [0] * len(vec)
    normVecDict = {key: value for key, value in zip(vector.keys(), normvec)}
    return normVecDict

print("Normalizing tf idf...\n")

tfidfNorm = copy.deepcopy(tfidf)
doctok = defaultdict(dict) # will store the tfidf indexed by documents.
for token, docdict in tfidf.items():
    for doc, value in docdict.items():
        doctok[doc][token] = value

NormVec = defaultdict(dict)
for doc, tokdict in doctok.items():
    NormVec[doc] = normalize(tokdict)
    for token, value in NormVec[doc].items():
        tfidfNorm[token][doc] = value


# Returns the topN documents by token relevance (vector model)
def getBestResults(queryStr, topN, NormDocs=False, NormQuery=False):
    query = queryStr.split(" ")
    if STEMMING:
        for idx, word in enumerate(query):
            query[idx] = STEMMING(word)

    tfidfMatrix = tfidfNorm if NormDocs else tfidf

    res = defaultdict(float)
    querycounter = Counter(query)
    tfidf_query = dict()
    for token, occ in querycounter.items():
        tfidf_query[token] = (occ / sum(querycounter.values())) * tokInfo[token] # Because tokInfo is a defaultdict, no need to check if token exists as a key.
    tfidf_queryNorm = normalize(tfidf_query)


    # Compute dot product between query and tfidfMatrix
    product = defaultdict(float)
    cosim = defaultdict(float)
    for token in tfidf_query.keys():
        for doc in tfidfMatrix.get(token, []):
            product[doc] += tfidf_query[token] * tfidfMatrix[token][doc]
            cosim[doc] += tfidf_queryNorm[token] * tfidfNorm[token][doc] # Will be equal to product if both query and docs are already normalized.
    return OrderedDict(itertools.islice(sorted(product.items(), key = operator.itemgetter(1), reverse = True), topN)), cosim


# Sorts a list of results according to their pageRank
def rankResults(results:dict):
    ranked = {doc: pageRankDict[doc] for doc in results.keys()}
    return OrderedDict(sorted(ranked.items(), key = operator.itemgetter(1), reverse = True))

def printResults(rankedResults):
    if len(rankedResults)==0:
        print("Your query returned no document.")
    sumcosimtot = sum(cosim.values()) # Stores the total cosimilarity between all indexed documents and the query.
    sumcosimtop = 0 # Stores the total cosimilarity between the topN documents displayed and the query.
    sumPRtot = sum(pageRankDict.values()) # Stores the total PR for all pages.
    sumPRtop = 0 # Stores the total PR for the topN pages.

    for idx, page in enumerate(rankedResults):
        print(f"{idx+1}. {page} ({cosim[page]:.4f} / {sumcosimtot:.2f} = {cosim[page]/sumcosimtot * 100:.2f}%)")
        sumcosimtop += cosim[page]
        sumPRtop += pageRankDict[page]
    print(60*"-")
    print(f"Total relevance of those {top} documents with respect to your query: {sumcosimtop:.2f} / {sumcosimtot:.2f} = {sumcosimtop/sumcosimtot * 100:.2f}%")
    #print(f"Share of total PageRank: {sumPRtop} / {sumPRtot:.2f} = {sumPRtop / sumPRtot * 100}%")



try:
    query = sys.argv[1]
except IndexError:
    query = "darwin"
try:
    top = int(sys.argv[2])
except IndexError:
    top = 10

print("Results for:", query, "\n", 60*"=")
results, cosim = getBestResults(query, top, NormDocs=False)
printResults(results)

print("\n\nResults after normalization for:", query, "\n", 60*"=")
results, cosim = getBestResults(query, top, NormDocs=True, NormQuery=True)
printResults(results)

print("\n\nResults after ranking for:", query, "\n", 60*"=")
ranked_results = rankResults(results)
printResults(ranked_results)


#bestPageSimilarity = list(reversed([ searchRes[i] for i in numpy.argsort(searchRes)[-10:] ]))
#bestPageSimilarity


