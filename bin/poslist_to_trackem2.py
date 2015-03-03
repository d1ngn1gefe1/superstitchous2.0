#Mikhail Kandel on 9/30, huutan86@gmail.com
from lxml import etree as ET
import os
import re
from scipy import misc
#Make sure to choose the height and width
srcfile='testo.poslist'
prefile=srcfile+"pre";
finalfile=prefile+".xml"
stub='metadata.xml'
minv=str(-.5)
maxv=str(1.0)
#
filenames = [stub,prefile]
root = ET.Element("trakem2")
#Do project heading
proj = ET.SubElement(root, "project")
proj.set("id","0");
unuid=str(102930128301823);
proj.set("unuid",unuid);
#proj.set("mipmaps_folder","/mipmaps/");#Triggers a rejen in current directory and now we wait
#proj.set("storage_folder","/");
proj.set("mipmaps_format",str(4));
proj.set("image_resizing_mode","Area downsampling")
proj.set("mipmaps_regen","true");

def oid():
    oid.counter += 1
    return oid.counter
oid.counter=10;
    
def mat(x,y): return "matrix(1.0,0.0,0.0,1.0,%s,%s)" %(x,y)

layer_set=ET.SubElement(root, "t2_layer_set")
layer_set.set("oid","3")
layer_set.set("width","20.0")
layer_set.set("height","20.0")
layer_set.set("transform",mat(0,0))
layer_set.set("title","Top Level")
layer_set.set("links","")
layer_set.set("layer_width","12") # sets later?
layer_set.set("layer_height","12") # sets later?
layer_set.set("rot_x","0") # sets later?
layer_set.set("rot_y","0") # sets later?
layer_set.set("rot_z","0") # sets later?
layer_set.set("snapshots_quality","true") # sets later?
layer_set.set("snapshots_mode","Outlines") # sets later?
layer_set.set("color_cues","true")
layer_set.set("area_color_cues","true")
layer_set.set("avoid_color_cue_colors","false")
layer_set.set("n_layers_color_cue","0")
layer_set.set("paint_arrows","true")
layer_set.set("paint_tags","true")
layer_set.set("paint_edge_confidence_boxes","true")
layer_set.set("prepaint","false")
layer_set.set("preload_ahead","0")
#
t2_calibration=ET.SubElement(layer_set, "t2_calibration")
t2_calibration.set("pixelWidth","1.0")
t2_calibration.set("pixelHeight","1.0")
t2_calibration.set("pixelDepth","1.0")
t2_calibration.set("xOrigin","0.0")
t2_calibration.set("yOrigin","0.0")
t2_calibration.set("zOrigin","0.0")
t2_calibration.set("info","null")
t2_calibration.set("valueUnit","Gray Value")
t2_calibration.set("timeUnit","sec")
t2_calibration.set("unit","pixel")
#
t2_layer = ET.SubElement(layer_set, "t2_layer")
t2_layer.set("oid",str(oid()))
t2_layer.set("thickness","1.0")
t2_layer.set("z","0.0")
t2_layer.set("title","")
#get image size
shape = []
rego=re.compile(r'(.*?.tif)[^0-9\r\n\-\+]+([-+]?[0-9]*\.?[0-9]+)[^0-9\r\n\-\+]+([-+]?[0-9]*\.?[0-9]+)')
#Let the patching begin
with open(srcfile) as f:
    first = True
    width=0
    height=0
    for line in f:
        parts=rego.findall(line)[0]# Todo should only get the end of the file
        t2_patch = ET.SubElement(t2_layer, "t2_patch")
        t2_patch.set("oid",str(oid()))
        if(first):
            shape=misc.imread(parts[0]).shape
            width=str(shape[1])
            height=str(shape[0])
            first=False
        t2_patch.set("width",width)
        t2_patch.set("height",height)
        t2_patch.set("transform",mat(parts[1],parts[2]))
        t2_patch.set('title',parts[0])
        t2_patch.set('links','')
        t2_patch.set('type','2')
        t2_patch.set('file_path',parts[0])
        t2_patch.set('style','fill-opacity:1.0;stroke:#ffff00;')
        t2_patch.set("o_width",width)
        t2_patch.set("o_height",height)
        t2_patch.set("min",minv)
        t2_patch.set("max",maxv)
        t2_patch.set("mres","32")

#taken from SO
tree = ET.ElementTree(root)
tree.write(prefile,pretty_print=True)
with open(finalfile, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
