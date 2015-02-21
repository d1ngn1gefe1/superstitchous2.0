#A proper unit test would automatically inspect these results
import wrapper
import shutil
import json
import os
#

#
#cells='bloottest.json'
cells='celltest.json'
proto=json.load(open(cells))
outdir=proto.get("outDir")
#
skipalign=True
switchposlist=True

if os.path.exists(outdir):
    shutil.rmtree(outdir)
wrapper.stitching(cells)
