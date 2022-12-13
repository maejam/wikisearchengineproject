import sys
import xml.etree.ElementTree
from collections import defaultdict, Counter
import numpy
import re
import pickle
import glob
from itertools import chain
from math import log

from time import process_time

from stemming import stemmer, naive_stemmer

STEMMER = naive_stemmer # lambda x:x / stemmer / naive_stemmer


xmlFiles = list(chain(*[ glob.glob(globName)  for globName in sys.argv[1:] ]))
print("Files as input:", xmlFiles)

docs = dict()

##############################
print("Parsing XML...")
##############################
for xmlFile in xmlFiles:
	pages = xml.etree.ElementTree.parse(xmlFile).getroot()

	for page in pages.findall("{http://www.mediawiki.org/xml/export-0.10/}page"):
		titles = page.findall("{http://www.mediawiki.org/xml/export-0.10/}title")
		revisions = page.findall("{http://www.mediawiki.org/xml/export-0.10/}revision")

		if titles and revisions:
			revision = revisions[0] # last revision
			contents = revision.findall("{http://www.mediawiki.org/xml/export-0.10/}text")
			if contents:
				docs[titles[0].text] = contents[0].text

# Some regEx for parsing
cleanExtLinks = "({{.*?}}|\[http.?:\/\/.*\])"
linkRe = "\[\[(.*?)\]\]"
removeLinkRe = "\[\[[^\]]+\|([^\|\]]+)\]\]"
removeLink2Re =  "\[\[([^\|\]]+)\]\]"
wordRe = "[a-zA-Z\-]+"
stopWords = ["-"]


def extract_links():
    """ Builds a dictionnary indexed by documents with links to other documents. """
    print("Extracting links...")
    links = dict()
    docs_keys = tuple(str(key).lower() for key in docs.keys()) # necessary for "[[organism]]" to match doc "Organism" in "Z-value (temperature)" as an example.
    for idx,doc in enumerate(docs):
        if idx%(len(docs)//20) == 0:
            print("Progress " + str(int(idx*100/len(docs)))  +"%", end="\r")
        links[doc] = list()
        for link in re.finditer(linkRe,docs[doc]):
            target = link.group(1).split('|')[0]
            if target.lower() in docs_keys:
                #print(doc + " --> " + target)
                links[doc] += [target]
    return links


def clean_docs():
    """ Cleans the documents by removing external links. """
    print("Cleaning documents...")
    for doc in docs:
        cleanDoc = re.sub(cleanExtLinks, "", docs[doc])

        # transform links to text
        docs[doc] = re.sub(removeLinkRe, r"\1", cleanDoc)
        docs[doc] = re.sub(removeLink2Re, r"\1", docs[doc])
    return docs


def create_doctok_matrix(docs):
    """ Creates a dictionnary of all tokens indexed by documents."""
    print("Filling a doc-tok matrix...")
    doctok = dict()

    for doc in docs:
        # fill the doctok matrix
        doctok[doc] = list()
        for wordre in re.finditer(wordRe,docs[doc]):
            word = wordre.group(0).lower()
            if word not in stopWords:
                word = STEMMER(word)
                doctok[doc] += [word]
        doctok[doc] = Counter(doctok[doc]) # aggregating duplicates
    return doctok


def transpose_doctok_matrix(matrix):
    print("Transposing doctok to tokdoc...")
    tokdoc = defaultdict(dict)
    for doc, token_count in matrix.items():
        for token, nb in token_count.items():
            tokdoc[token].update({doc: nb})
    return tokdoc


def create_tokdoc_matrix(docs):
    print("Filling a tok-doc matrix...")
    tokdoc = defaultdict(list)

    for doc in docs:
        # fill the tokdoc matrix
        for wordre in re.finditer(wordRe, docs[doc]):
            word = wordre.group(0).lower()
            if word not in stopWords:
                word = STEMMER(word)
                tokdoc[word].append(doc)
    tokdoc = {tok: Counter(tokdoc[tok]) for tok in tokdoc} # aggregating duplicates
    return tokdoc


def build_tf_idf_table():
    print("Building tf-idf table...")
    docList = doctok.keys()
    Ndocs = len(docList)

    tokInfo = defaultdict(float) # tokInfo[tok] contains the information in bits of the token
    tf = dict() # tf[doc][tok] contains the frequency of the token tok in document doc
    tfidf = dict()

    for tok in tokdoc:
        tfidf[tok] = dict()
        tokInfo[tok] = log(Ndocs / len(tokdoc[tok]), 2)
    for doc in docList:
        tf[doc] = dict()
        tottokindoc = sum(doctok[doc].values()) #total number of tokens in a given document. Trying to win the prize for funniest sounding variable name :)
        for tok in doctok[doc]:
            tokindoc = tokdoc[tok][doc] #nb of times a given token appears in a document.

            tf[doc][tok] = tokindoc/tottokindoc

            #Compute tf-idf
            tfidf[tok][doc] = tf[doc][tok] * tokInfo[tok]

    return tfidf, tokInfo


links = extract_links()
cleanDocs = clean_docs()

print("========================================================================")

t = process_time()
doctok = create_doctok_matrix(cleanDocs)
elapsed_time_doctok = process_time() - t
print(f"Time taken to build Doctok matrix : {elapsed_time_doctok:.4f} seconds.")

t = process_time()
tokdoc_t = transpose_doctok_matrix(doctok)
elapsed_time_transpose = process_time() - t
print(f"Time taken to transpose Doctok matrix to Tokdoc matrix : {elapsed_time_transpose:.4f} seconds.")

t = process_time()
tokdoc = create_tokdoc_matrix(cleanDocs)
elapsed_time_tokdoc = process_time() - t
print(f"Time taken to build Tokdoc matrix : {elapsed_time_tokdoc:.4f} seconds.")


print("========================================================================")

tfidf, tokInfo = build_tf_idf_table()


print("Saving the links and the tfidf as pickle objects...")
with open("links.dict",'wb') as fileout:
	pickle.dump(links, fileout, protocol=pickle.HIGHEST_PROTOCOL)

with open("tfidf.dict",'wb') as fileout:
	pickle.dump(tfidf, fileout, protocol=pickle.HIGHEST_PROTOCOL)

with open("tokInfo.dict",'wb') as fileout:
	pickle.dump(tokInfo, fileout, protocol=pickle.HIGHEST_PROTOCOL)
