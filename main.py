import os
import glob
import pickle

from tools import searchcore
from tools import retdoc

HOME = os.getcwd()

col_path = os.path.join(HOME, "data/rawdata/20news-18828/**/*")
docpaths = glob.glob("/home/boyangeor/datasets/20news-18828/**/*", recursive=True)
# The first 20 paths are the subdirs (not files)
docpaths = docpaths[20:] 

x = searchcore.SearchCore(docpaths, log_weight=True, min_freq=1, k=1000)
x.summary()
x.export()

# with open("/home/boyangeor/datasets/SearchCore.pkl", 'rb') as infile:
#     x = pickle.load(infile)

# those terms are from '67211'
q = "environment layout unix xterm pageup pagedown"

ret = retdoc.retrieve(x, q, top=10)

for e in ret[0]:
    print(e)

print()

for e in ret[1]:
    print(e)
