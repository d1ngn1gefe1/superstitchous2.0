'''Helper script to run main.exe using directory containing images and .poslist file'''
import sys
import math
import json
import os
import platform
import re
from glob import glob
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
    #http://stackoverflow.com/questions/6788398/how-to-save-progressive-jpeg-using-python-pil-1-1-7
    img=Image.fromarray(image)
    q=80
    try:
        img.save(name, "JPEG", quality=q, optimize=True, progressive=True)
    except IOError:
        ImageFile.MAXBLOCK = img.size[0] * img.size[1]
        img.save(name, "JPEG", quality=q, optimize=True, progressive=True)              

def saveZoomedIms(folder, zoomLvls=6, verbose=True):
    """zoomLvls is the highest x in "*_*_x.jpg"."""
    def getPath(name):
        return join(folder, name)

    def printV(*args):
        if verbose:
            print( ' '.join(str(a) for a in args))

    for zoom in range(zoomLvls):
        printV('--- generating zoom %d ---' % (zoom + 1))
        gridW = len(glob(getPath('0_*_%d.jpg' % zoom)))
        gridH = len(glob(getPath('*_0_%d.jpg' % zoom)))

        printV('gridW:', gridW, 'gridH:', gridH)

        size = None
        for x in range(0, gridW, 2):
            for y in range(0, gridH, 2):
                ims = []
                for dy in (0, 1):
                    for dx in (0, 1):
                        name = getPath('%d_%d_%d.jpg' % (y + dy, x + dx, zoom))
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

                name = getPath('%d_%d_%d.jpg' % (y / 2, x / 2, zoom + 1))
                printV('saving', name)
                dirtyWriteImage(name,scaled)
#                misc.imsave(name, scaled)

def getPxSize(folder):
    gridW = len(glob(join(folder, '0_*_0.jpg')))
    gridH = len(glob(join(folder, '*_0_0.jpg')))
    pxH, pxW = misc.imread(join(folder, '0_0_0.jpg')).shape
    return pxW * gridW, pxH * gridH

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

def stitching(fin):
    cfg = json.load(open(fin))
    skipAlign = cfg.get('skipAlign', False)
    imFiles, coords, snakeDir, cols, rows, xOff, yOff = parsePoslistDir(cfg['inDir'], cfg.get('poslist', None))
    im = misc.imread(imFiles[0])
    imH, imW = im.shape
    imDir = join(cfg['outDir'], STACK_NAME, '0')
    mkdir(cfg['outDir'])
    mkdir(imDir)
    imH, imW = im.shape
    imDir = join(cfg['outDir'], STACK_NAME, '0')
    mkdir(cfg['outDir'])
    mkdir(imDir)
    if skipAlign:
        maxPeakX = cfg.get('maxPeakX', float(imW - xOff) / imW * 0.2 * math.sqrt(imW * imH))
        maxPeakY = cfg.get('maxPeakY', float(imH - yOff) / imH * 0.2 * math.sqrt(imW * imH))
        maxPeakXY = cfg.get('maxPeakXY', float((imW - xOff) * (imH - yOff)) / (imW * imH) * 0.2 * math.sqrt(imW * imH))

        maxPeakX = max(int(round(maxPeakX)), 1)
        maxPeakY = max(int(round(maxPeakY)), 1)
        maxPeakXY = max(int(round(maxPeakXY)), 1)
    else:
        maxPeakX = -1
        maxPeakY = -1
        maxPeakXY = -1

    outWidth = cfg.get('outWidth',1024)
    outHeight = cfg.get('outHeight',1024)
                        
    weightPwr = cfg.get('weightPwr', 1)
    weightPwr = cfg.get('weightPwr', 1)    
    
    weightPwr = cfg.get('weightPwr', 1)
    peakRadius = cfg.get('peakRadius', 0)
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
            outHeight
            ]
    args = list(map(str, args)) #for Python 3
##  args = ['gdb', '--args'] + args
    print( ' '.join(args))
    quit()
#    print(args[0])
    if call(args) != 0:
        os.remove(tmpPath)
        sys.exit(1)

    os.remove(tmpPath) #Should use RAII mechanism

    saveZoomedIms(imDir, cfg['zoomLvls'])

    pxW, pxH = getPxSize(imDir)

    # write small.jpg: icon image for catmaid
    im = misc.imread(join(imDir, '0_0_%d.jpg' % cfg['zoomLvls']))
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
    projStr = projStr.replace('{ZOOMS}', str(cfg['zoomLvls']))
    open(join(cfg['outDir'], 'project.yaml'), 'w').write(projStr)

    if cfg['zoomLvls'] >= SMALL_ZOOM:
        ## save cropped image for core segmentation
        im = misc.imread(join(imDir, '0_0_%d.jpg' % SMALL_ZOOM))
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
        misc.imsave(join(imDir, 'cropped.jpg'), im)

##        print 'window size:', getWinSize(im)

def segmentation(fin):
    print (fin)
    cfg = json.load(open(fin))

    inDir = cfg['inDir']+'/stack1/0'
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
