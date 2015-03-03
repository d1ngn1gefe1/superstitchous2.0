'''Helper script to run main.exe using directory containing images and .poslist file'''
import sys
import math
import json
import os
import shutil 
import platform
import re
from glob import glob
import distutils.core
from subprocess import call
import tempfile
import numpy as np
import scipy.signal as sig
from scipy import misc
from PIL import Image, ImageFile
from common import *
import argparse    
    
STACK_NAME = 'stack1'

# zoom level to use for the small image to segment cores with
SMALL_ZOOM = 7

def getWinSize(im, padFactor=4):
    h, w = im.shape
    im = np.pad(im, ((0, h * (padFactor - 1)), (0, w * (padFactor - 1))), mode='constant')

    spec = abs(np.fft.fft2(im))

    horizIs = sig.argrelmax(spec[0, :])[0]
    vertIs = sig.argrelmax(spec[:, 0])[0]
    maxHorizI = max(horizIs, key=lambda i: spec[0, i])
    maxVertI = max(vertIs, key=lambda i: spec[i, 0])
    return round(float(im.shape[1]) / maxHorizI), round(float(im.shape[0]) / maxVertI)

def dirtyWriteImage(name,image):
    #Wrapper around an OpenCV bug
    #http://stackoverflow.com/questions/6788398/how-to-save-progressive-jpeg-using-python-pil-1-1-7
    img=Image.fromarray(image)
    q=80
    try:
        img.save(name, "JPEG", quality=q, optimize=True, progressive=True)
    except IOError:
        ImageFile.MAXBLOCK = img.size[0] * img.size[1]
        img.save(name, "JPEG", quality=q, optimize=True, progressive=True)   
           
def saveZoomedImsCatmaid(folder, zoomLvls, verbose=True):
    #CATMAID in ROW_COL_ZOOM.jpg, zoom = 0 is zoomed in
    def getPath(name):
        return join(folder, name)

    def printV(*args):
        if verbose:
            print( ' '.join(str(a) for a in args))
        
    def filenames(row,col,zoom):
            namestring='%d_%d_%d.jpg'
            return namestring % (row,col,zoom)
    
    def count(dim,zoom):
            selector=['0_*_%d.jpg','*_0_%d.jpg']
            return len(glob(getPath(selector[dim] % zoom)))
            return len(glob(what))
    
    zoomrange = range(0,zoomLvls-1,1)#Zoom 1 is already done
    for zoom in zoomrange:
        printV('--- generating zoom %d ---' % (zoom))
        gridW =count(0,zoom)
        gridH =count(1,zoom)
        printV('gridW:', gridW, 'gridH:', gridH)
        size = None
        for x in range(0, gridW, 2):
            for y in range(0, gridH, 2):
                ims = []
                for dy in (0, 1):
                    for dx in (0, 1):
                        name = getPath(filenames(y + dy, x + dx, zoom))
                        printV('read', name)
                        try:
                            im = misc.imread(name)
                        except IOError:
                            printV('not there, fill empty')
                            ims.append(np.zeros((size[0] * 0.5, size[1] * 0.5)))
                        else:
                            if size is None:
                                size = im.shape
                            ims.append(misc.imresize(im, 0.5, 'nearest'))

                scaled = np.concatenate((np.concatenate((ims[0], ims[1]), 1), np.concatenate((ims[2], ims[3]), 1)), 0)
                scaled = scaled.astype('uint8')

                name = getPath(filenames(y / 2, x / 2, zoom + 1))
                printV('saving', name)
                dirtyWriteImage(name,scaled)
                
    #misc.imsave(name, scaled)
    gridW =count(0,0)
    gridH =count(1,0)
    first=getPath(filenames(0, 0, 0))
    pxH, pxW = misc.imread(first).shape
    pxW,pxH = pxW * gridW, pxH * gridH
    # write small.jpg: icon image for catmaid
    im = misc.imread(join(folder, '0_0_%d.jpg' % zoomLvls))
    if len(im) > len(im[0]):
        # height > width
        im = misc.imresize(im, 256. / len(im))
        padLen = (256 - len(im[0])) / 2
        im = np.pad(im, ((0, 0), (padLen, padLen)), mode='constant')
    else:
        im = misc.imresize(im, 256. / len(im[0]))
        padLen = (256 - len(im)) / 2
        im = np.pad(im, ((padLen, padLen), (0, 0)), mode='constant')
        print ('writing small.jpg')
        misc.imsave(join(cfg['outDir'], STACK_NAME, '0', 'small.jpg'), im)

    # write project.yaml file
    projStr = open('project.yaml').read()
    projStr = projStr.replace('{NAME}', cfg['projName'])
    print ('total pixel width/height: %d/%d' % (pxW, pxH))
    projStr = projStr.replace('{W_PX}', str(pxW))
    projStr = projStr.replace('{H_PX}', str(pxH))
    projStr = projStr.replace('{ZOOMS}', str(zoomLvls))
    open(join(cfg['outDir'], 'project.yaml'), 'w').write(projStr)

    if zoomLvls >= SMALL_ZOOM:
        ## save cropped image for core segmentation
        im = misc.imread(join(folder, '0_0_%d.jpg' % SMALL_ZOOM))
        info = open(join(cfg['outDir'], STACK_NAME, '0', 'info.txt')).read()
        w, h = re.findall(r'total size: (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)', info)[0]
        w = float(w)
        h = float(h)
        print ('actual dims: (%s, %s)' % (w, h))
        w = round(w / (2 ** SMALL_ZOOM))
        h = round(h / (2 ** SMALL_ZOOM))
        print ('cropped dims: (%s, %s)' % (w, h))
        im = im[:h, :w]
        print ('saving cropped.jpg')
        misc.imsave(join(folder, 'cropped.jpg'), im)
    
def saveZoomedImsZoomify(folder,outdir,verbose=True):
    #Zoomify in ZOOM-ROW-COL.jpg, zoom = 0 is zoomed out 
    #Also Zoomify tilegroups are in 256 packages
    #Mostly stolen from OpenLayers, blame them not me!
    def getPath(name):
        return join(folder, name)
    #Build data structures for weird zoomify tilegroups
    defaultselector=['0_*_%d.jpg','*_0_%d.jpg']
    gridW= len(glob(getPath(defaultselector[0] % 0)))
    gridH = len(glob(getPath(defaultselector[1] % 0)))
    imgH,imgW = misc.imread(getPath('0_0_0.jpg')).shape #Yes, its in row,cols
    print("Grid Width: %d Grid Height: %d"%(gridW,gridH))
    assert(imgH==imgW)
    tilesize = imgH
    tiles = (gridW,gridH)
    tierSizeInTiles = [tiles]
    imageSize=imgW*gridW,imgH*gridH
    tierImageSize = [imageSize]
    while (imageSize[0]>tilesize  or imageSize[1] > tilesize):
        imageSize = math.floor(imageSize[0]/2),math.floor(imageSize[1]/2)#Why floor?
        tiles = math.ceil(imageSize[0]/tilesize),math.ceil(imageSize[1]/tilesize)
        tierImageSize.append(imageSize)
        tierSizeInTiles.append(tiles)    
    tierImageSize=tierImageSize[::-1]
    imgcount = 0    
    for t in tierSizeInTiles:
        imgcount =imgcount+ t[0]*t[1]
    tierSizeInTiles=tierSizeInTiles[::-1]
    numberOfTiers=len(tierSizeInTiles)
    tileCountUpToTier = [0]
    resolutions = [1]
    for i in range(1,numberOfTiers):
        resolutions.insert(0, math.floor(math.pow(2,i)))
        sz = tierSizeInTiles[i-1]        
        val=sz[0]*sz[1]+tileCountUpToTier[i-1]
        tileCountUpToTier.append(val)
    
    def printV(*args):
        if verbose:
            print( ' '.join(str(a) for a in args))
            
    def filenames(row,col,zoom):
            namestring='%d_%d_%d.jpg'
            return namestring % (row,col,zoom)
    
    def count(dim,zoom):
            selector=['0_*_%d.jpg','*_0_%d.jpg']
            return len(glob(getPath(selector[dim] % zoom)))
            return len(glob(what))
    
    zoomrange = range(0,numberOfTiers-1,1)#Zoom 1 is already done
    print(zoomrange)
    for zoom in zoomrange:
        printV('--- generating zoom %d ---' % (zoom))
        gridW =count(0,zoom)
        gridH =count(1,zoom)
        printV('gridW:', gridW, 'gridH:', gridH)
        size = None
        for x in range(0, gridW, 2):
            for y in range(0, gridH, 2):
                ims = []
                for dy in (0, 1):
                    for dx in (0, 1):
                        name = getPath(filenames(y + dy, x + dx, zoom))
                        printV('read', name)
                        try:
                            im = misc.imread(name)
                        except IOError:
                            printV('not there, fill empty')
                            ims.append(np.zeros((size[0] * 0.5, size[1] * 0.5)))
                        else:
                            if size is None:
                                size = im.shape
                            ims.append(misc.imresize(im, 0.5, 'nearest'))

                scaled = np.concatenate((np.concatenate((ims[0], ims[1]), 1), np.concatenate((ims[2], ims[3]), 1)), 0)
                scaled = scaled.astype('uint8')

                name = getPath(filenames(y / 2, x / 2, zoom + 1))
                printV('saving', name)
                dirtyWriteImage(name,scaled)
        
    #Fix the names
    imageroot = '%s/image'%outdir 
    if not os.path.exists(imageroot):    
        os.mkdir(imageroot)
        
    def groupPath(group):
        return '%s/TileGroup%d' %(imageroot,group)
        
    def getTile(row,col,zoom):
        w=tierSizeInTiles[zoom]
        idx = col+row*w[0]+tileCountUpToTier[zoom]
        group = int(math.floor(idx/256))
        return (getPath('%s/%d-%d-%d.jpg' %(groupPath(group),zoom,col,row)),group)
    
    safe={}
    def checkFolder(group):
        if not group in safe:
            dirpath=groupPath(group)
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
            safe[group]=True
    
    tierSizeInTilesIn=tierSizeInTiles[::-1]
    for z in range(0,numberOfTiers):
        gridW,gridH = tierSizeInTilesIn[z]
        for x in range(0,gridW):
                for y in range(0,gridH):
                    src=getPath(filenames(y,x ,z))
                    zout = numberOfTiers-z-1
                    dst,group=getTile(y,x,zout)
                    checkFolder(group)
                    print("%s->%s" %(src,dst))
                    shutil.move(src,dst)

    srcZ=toAbsolutePath("../zoomify")
    distutils.dir_util.copy_tree(srcZ,outdir,verbose=False)
    lazyxml=r'<IMAGE_PROPERTIES WIDTH="%f" HEIGHT="%f" NUMTILES="%d" NUMIMAGES="1" VERSION="1.8" TILESIZE="%d" />'        
    dstLabel="%s//ImageProperties.xml" %imageroot
    pxW=tierImageSize[-1][0]
    pxH=tierImageSize[-1][1]
    with open(dstLabel, "w") as text_file:
        text_file.write(lazyxml %(pxW,pxH,imgcount,tilesize))     
    
    stackextra = '%s/%s'%(outdir,STACK_NAME )
    poslist = '%s/0/stitch_poslist' % stackextra
    poslistdest='%s/stitch_poslsit'% imageroot
    if os.path.exists(poslist):
        shutil.move(poslist,poslistdest)
        
    if os.path.exists(stackextra):    
        shutil.rmtree(stackextra)      
                


def makeOrderedImData(dat, snakeDir, szInIms):
    """Rearrange items in dat, whose elements are in a snake shape, into row-major order. Return new array."""
    xDIM = szInIms[0]
    yDIM = szInIms[1]
    arr = []
    for i in range(yDIM):
        for j in range(xDIM):
            if snakeDir == 'col':
                ind = j * yDIM
                if j % 2:
                    ind += yDIM - i - 1
                else:
                    ind += i
            elif snakeDir == 'row':
                ind = i * xDIM
                if i % 2:
                    ind += xDIM - j - 1
                else:
                    ind += j;
            else:
                print('invalid snakeDir')
                sys.exit(1)
            arr.append(dat[ind])
    return arr

def fixJsonPaths(cfg):
    convert = ['inDir','outDir','poslist']
    for name in convert:
        if (name in cfg):
            what = cfg[name]
            if ('names' not in what):
                cfg[name]=toAbsolutePath(what)
#            if os.path.exists(abspath):
#                cfg[name]=abspath
#            else:
#                err("Can't find %s" % abspath)
    outdir=cfg['outDir']
    if os.path.exists(outdir):
        err("Output path already exists, please clear before continuing:\n%s" % outdir)
        
    return cfg

def stitching(fin):
    cfg = json.load(open(fin))
    cfg = fixJsonPaths(cfg)
    skipAlign = cfg.get('skipAlign', False)
    raster = cfg.get("rasterFormat","CATMAID")
    imFiles, coords, snakeDir, cols, rows, xOff, yOff = parsePoslistDir(cfg['inDir'], cfg.get('poslist', None),cfg.get('pixelRatio',1))
    im = misc.imread(imFiles[0])
    imH, imW = im.shape #Yes, its rows first
    outdir = cfg['outDir']
    imDir = join(outdir, STACK_NAME, '0')
    mkdir(outdir)
    mkdir(imDir)#Necissary?
    imH, imW = im.shape
    imDir = join(outdir, STACK_NAME, '0')
    mkdir(cfg['outDir'])
    mkdir(imDir)
    zoomlevels = cfg.get('zoomLvls',6)
    if skipAlign:
        maxPeakX = -1
        maxPeakY = -1
        maxPeakXY = -1
    else:
        maxPeakX = cfg.get('maxPeakX', float(imW - xOff) / imW * 0.2 * math.sqrt(imW * imH))
        maxPeakY = cfg.get('maxPeakY', float(imH - yOff) / imH * 0.2 * math.sqrt(imW * imH))
        maxPeakXY = cfg.get('maxPeakXY', float((imW - xOff) * (imH - yOff)) / (imW * imH) * 0.2 * math.sqrt(imW * imH))
        maxPeakX = max(int(round(maxPeakX)), 1)
        maxPeakY = max(int(round(maxPeakY)), 1)
        maxPeakXY = max(int(round(maxPeakXY)), 1)
    
    outWidth = cfg.get('outWidth',1024)
    outHeight = cfg.get('outHeight',1024)
    if ((outWidth != outHeight) and (raster=="ZOOMIFY")):
        err("Zoomify requires squre output tiles:")
                        
    weightPwr = cfg.get('weightPwr', 1)
    weightPwr = cfg.get('weightPwr', 1)    
    
    weightPwr = cfg.get('weightPwr', 1)
    peakRadius = cfg.get('peakRadius', 1)
    fixNaN = int(cfg.get('fixNaN', 1))
    bgSub = cfg.get('bgSub', 0)

    print('rows:', rows, 'cols:', cols, 'xOff:', xOff, 'yOff:', yOff, 'snakeDir:', snakeDir)
    print('im sz:', (imW, imH))
    print('input dir:', cfg['inDir'])
    print('CATMAID dir:', cfg['outDir'])
    print('image dir:', imDir)
    print('maxPeakX, Y, XY:', maxPeakX, maxPeakY, maxPeakXY)
    print('weight power:', weightPwr)
    print('peak radius:', peakRadius)
    print('fix NaNs?', fixNaN)
    print('background subtraction ', bgSub)
    print('im out sz:', (outWidth, outHeight))
    
    imData = [(f, c[0] - coords[0][0], c[1] - coords[0][1]) for f, c in zip(imFiles, coords)]
    if skipAlign==False: 
        if(snakeDir=='none'):
            err("All poslists for aligment should be ordered in snake fashion")
        imData = makeOrderedImData(imData, snakeDir, (cols, rows))
    else:
        print('Using a known position list')
    
    # write image data
    tmpFd, tmpPath = tempfile.mkstemp()
    imglist = '\n'.join(' '.join(map(str, dat)) for dat in imData)
    os.write(tmpFd, bytes(imglist,'UTF-8'))
    args = []
    if platform.system() == 'Windows':
        args.append(r'..\vsproject\superstitchous2013\x64\Release\stitching.exe')
    else:
        args.append(r'./stitching')
#        args.append(r'../src/stiching')
    args += [
            tmpPath,
            imDir,
            xOff,
            yOff,
            cols,
            rows,
            maxPeakX,
            maxPeakY,
            maxPeakXY,
            cfg.get('cacheSz', 50),
            imW,
            imH,
            skipAlign,
            weightPwr,
            peakRadius,
            fixNaN,
            bgSub,
            outWidth,
            outHeight,
            int(cfg.get('writeJPG', 1)),
            int(cfg.get('writeTIF', 0)),
            float(cfg.get('colormapMin', -.3)),
            float(cfg.get('colormapMax', 0.7)),
            ]
    args = list(map(str, args)) #for Python 3
##  args = ['gdb', '--args'] + args
    print( ' '.join(args))
    if(cfg.get('debug_cpp',False)):
        debugname='debug_cpp2.txt'
        with open(debugname,'w') as f:
            f.write(imglist)
        sys.exit(0)  

    codes = call(args)
    if  codes != 0:
        print("Stitching returned an error: %d" %codes)
        sys.exit(1)

    if (raster=="CATMAID"):        
        saveZoomedImsCatmaid(imDir, zoomlevels)
    if (raster=="ZOOMIFY"):
        saveZoomedImsZoomify(imDir,outdir)
    #
    print("Done!")

def segmentation(fin):
    print (fin)
    cfg = json.load(open(fin))

    inDir = cfg['inDir']+('/%s/0' %STACK_NAME)
    outDir = cfg['outDir']
    with open(cfg['inDir']+'/project.yaml') as yaml:
        for line in yaml:
            if re.match('.*name: ".+"', line):
                projName = line.split('"')[1]
            if re.match('.*zoomlevels: .+', line):
                zoomLvls = line.split('zoomlevels: ')[1].rstrip()
    files = glob(inDir+'/0_0_0.tiff')
    if not files:
        sys.exit(2)
    im = misc.imread(files[0])
    tileH, tileW = im.shape
    gridW = len(glob(inDir+'/0_*_0.tiff'))
    gridH = len(glob(inDir+'/*_0_0.tiff'))
    coreW = cfg['coreW']
    coreH = cfg['coreH']
    winW = cfg.get('winW',55)
    winH = cfg.get('winH',55)
    crop = cfg.get('crop',0)
    
    print ('inDir:', inDir)
    print ('outDir:', outDir)
    print ('projName:', projName)
    print ('zoomLvls:', zoomLvls)
    print ('tileW:', tileW)
    print ('tileH:', tileH)
    print ('gridW:', gridW)
    print ('gridH:', gridH)
    print ('coreW:', coreW)
    print ('coreH:', coreH)
    print ('winW:', winW)
    print ('winH:', winH)
    print ('crop:', crop)
    
    args = []
    args.append(r'..\vsproject\superstitchous2013\x64\Release\segmentation.exe')
    args += [
            inDir,
            outDir,
            projName,
            zoomLvls,
            tileW,
            tileH,
            gridW,
            gridH,
            coreW,
            coreH,
            winW,
            winH,
            crop
            ]
    args = map(str, args)
    print (' '.join(args))

    if call(args) != 0:
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stitch", help="do stitching", metavar="<config file for stitch>")
    parser.add_argument("-c", "--core", help="do core segmentation", metavar="<config file for seg>")
    args = parser.parse_args()
    if args.stitch:
        stitching(args.stitch)
    if args.core:
        segmentation(args.core)
