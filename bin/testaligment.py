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
odir = proto["outDir"]
for x in range(95,140,5):
    params = ("_%d" % x)
    proto["maxPeakXY"] = x
    proto["maxPeakX"] = x
    proto["maxPeakY"] = x
    #
    proto["outDir"]=odir+params
    outdir=proto["outDir"]
    if os.path.exists(outdir):
        shutil.rmtree(outdir) 
    useme=params+".json"
    with open(useme,'w') as outfile:
        json.dump(proto,outfile)
    wrapper.stitching(useme)
    inkto='/var/www/localhost/htdocs/foo'+params
    print(outdir)
    if not os.path.exists(inkto):
        os.symlink(outdir,inkto)
    print(inkto)

#wrapper.stitching(cells)
#D:\ScanTest2

