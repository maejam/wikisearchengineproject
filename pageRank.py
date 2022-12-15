from itertools import chain
import numpy
import pickle
from pprint import pprint
import sys, os

CONVERGENCE_LIMIT = 0.001
DEPTH = 2
PERSONALIZED_INTEREST = None #"RNA"


path = f"depth{DEPTH}"


# Load the link information
try:
    with open(os.path.join(path, "links.dict"),'rb') as f:
        links = pickle.load(f)
except FileNotFoundError:
    sys.exit(f'DEPTH {DEPTH} has not been computed yet!')

# List of page titles
allPages = list(set().union(chain(*links.values()), links.keys()))

# For simplicity of coding we give an index to each of the pages.
linksIdx = [[allPages.index(target) for target in links.get(source, list())] for source in allPages]
print("done")

# Remove redundant links (i.e. same link in the document)
for l in links:
	links[l] = list(set(links[l]))


# One click step in the "random surfer model"
# origin = probability distribution of the presence of the surfer (list of numbers) on each of the page
def surfStep(origin, links):
    dest = [0.0] * len(origin)
    for idx, proba in enumerate(origin):
        if len(links[idx]):
            w = 1.0 / len(links[idx])
        else:
            w = 0.0
        for linkid in links[idx]:
            dest[linkid] += proba * w
    return dest # proba distribution after a click

# Init of the pageRank algorithm
pageRanks = [1.0/len(allPages)] * len(allPages) # will contain the page ranks
sourceVector = [0.0] * len(allPages)
if not PERSONALIZED_INTEREST:
    sourceVector = [1.0/len(allPages)] * len(allPages) # default source

# Or use a personalized source vector :
else:
    personalized_pages = [page for page in allPages if PERSONALIZED_INTEREST in page]
    for idx, page in enumerate(allPages):
        sourceVector[idx] = 1/len(personalized_pages) if page in personalized_pages else 0.0


delta = float("inf")
iteration = 0
while delta > CONVERGENCE_LIMIT: #TO COMPLETE (1 expression)
    iteration += 1
    print(iteration, ") Convergence delta: ", delta, sum(pageRanks), len(pageRanks))
    # with open("convergence.txt", "a") as f:
        # print(iteration, "\t", delta, sep="", file=f)
    pageRanksNew = surfStep(pageRanks, linksIdx) #TO COMPLETE (1 expression)
    jumpProba = sum(pageRanks) - sum(pageRanksNew) # what effect is detected here?
    if jumpProba < 0: # Technical artifact due to numerical errors
        jumpProba = 0
    # Correct for this effect:
    pageRanksNew = [pageRank + jump for pageRank, jump in zip(pageRanksNew, (p*jumpProba for p in sourceVector))]
    # Compute the delta:
    delta = numpy.linalg.norm(numpy.subtract(pageRanksNew, pageRanks), 1)
    pageRanks = pageRanksNew

bestPages = [allPages[i] for i in numpy.argsort(pageRanks)[-20:]]
bestPageRanks = [pageRanks[i] for i in numpy.argsort(pageRanks)[-20:]]

# Name the entries of the pageRank vector
pageRankDict = dict()
for idx, pageName in enumerate(allPages):
    pageRankDict[pageName] = pageRanks[idx]

# Rank of some pages:
print('\n------------------------------------------------------------')
print("Pages with the highest PageRank: \n")
pprint(list(reversed(list(zip(bestPages, bestPageRanks)))))
#print("PageRank of 'DNA': " + pageRanks)


# Save the ranks as pickle object
with open("pageRank.dict",'wb') as fileout:
	pickle.dump(pageRankDict, fileout, protocol=pickle.HIGHEST_PROTOCOL)

