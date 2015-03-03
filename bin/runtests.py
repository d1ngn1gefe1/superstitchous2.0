#A proper unit test would automatically inspect these results
import wrapper
import shutil
import json
import os
#
cells='morebone.json'
#cells='celltest.json'
proto=json.load(open(cells))
outdir=proto.get("outDir")
#
if os.path.exists(outdir):
    shutil.rmtree(outdir)
wrapper.stitching(cells)