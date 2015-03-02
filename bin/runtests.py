#A proper unit test would automatically inspect these results
import wrapper
import shutil
import json
import os
#
#from scipy import misc
#other='/home/daniel/Desktop/Fantasy_Fudge.jpg'
#print(misc.imread(other).shape)
#
#cells='ScanTest.json'
#cells='bloottest.json'
#cells='celltest.json'
cells='bigbone.json'
proto=json.load(open(cells))
outdir=proto.get("outDir")
#
if os.path.exists(outdir):
    shutil.rmtree(outdir)
wrapper.stitching(cells)

#D:\ScanTest2
