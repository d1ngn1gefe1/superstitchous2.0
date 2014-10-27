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
            print ' '.join(str(a) for a in args)

    for zoom in xrange(zoomLvls):
        printV('--- generating zoom %d ---' % (zoom + 1))
        gridW = len(glob(getPath('0_*_%d.jpg' % zoom)))
        gridH = len(glob(getPath('*_0_%d.jpg' % zoom)))

        printV('gridW:', gridW, 'gridH:', gridH)

        size = None
        for x in xrange(0, gridW, 2):
            for y in xrange(0, gridH, 2):
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
    arr = []
    for i in xrange(szInIms[1]):
        for j in xrange(szInIms[0]):
            if snakeDir == 'col':
                ind = j * szInIms[1]
                if j % 2:
                    ind += szInIms[1] - i - 1
                else:
                    ind += i
            elif snakeDir == 'row':
                ind = i * szInIms[0];
                if i % 2:
                    ind += szInIms[0] - j - 1
                else:
                    ind += j;
            else:
                print 'invalid snakeDir'
                sys.exit(1)
            arr.append(dat[ind])
    return arr
      
def stitching(fin):
    print fin
    cfg = json.load(open(fin))
    imFiles, coords, snakeDir, cols, rows, xOff, yOff = parsePoslistDir(cfg['inDir'], cfg.get('poslist', None))

    #sanity checks
    im = misc.imread(glob(join(cfg['inDir'], '*.tif'))[0])
    imH, imW = im.shape

    imDir = join(cfg['outDir'], STACK_NAME, '0')
    mkdir(cfg['outDir'])
    mkdir(imDir)

    if 'usePoslist' not in cfg:
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

    outWidth = cfg.get('outWidth',4096)
    outHeight = cfg.get('outHeight',4096)
                        
    weightPwr = cfg.get('weightPwr', 1)
    weightPwr = cfg.get('weightPwr', 1)    
    
    weightPwr = cfg.get('weightPwr', 1)
    peakRadius = cfg.get('peakRadius', 0)
    fixNaN = int(cfg.get('fixNaN', 1))
    bgSub = cfg.get('bgSub', 0)

    print 'rows:', rows, 'cols:', cols, 'xOff:', xOff, 'yOff:', yOff, 'snakeDir:', snakeDir
    print 'im sz:', (imW, imH)
    print 'input dir:', cfg['inDir']
    print 'CATMAID dir:', cfg['outDir']
    print 'image dir:', imDir
    print 'maxPeakX, Y, XY:', maxPeakX, maxPeakY, maxPeakXY
    print 'weight power:', weightPwr
    print 'peak radius:', peakRadius
    print 'fix NaNs?', fixNaN
    print 'background subtraction ', bgSub
    print 'im out sz:', (outWidth, outHeight)
    
    imData = [(f, c[0] - coords[0][0], c[1] - coords[0][1]) for f, c in zip(imFiles, coords)]
    if 'usePoslist' not in cfg:
        imData = makeOrderedImData(imData, snakeDir, (cols, rows))

    # write image data
    tmpFd, tmpPath = tempfile.mkstemp()
    debuggles = '\n'.join(' '.join(map(str, dat)) for dat in imData)
    os.write(tmpFd, debuggles)
    os.close(tmpFd)

    args = []
    if platform.system() == 'Windows':
        args.append(r'..\vsproject\superstitchous2013\x64\Release\stitching.exe')
    else:
        args.append(r'./main.exe')
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
            1 if ('usePoslist' in cfg) else 0,
            weightPwr,
            peakRadius,
            fixNaN,
            bgSub,
            outWidth,
            outHeight
            ]
    args = map(str, args)
##    args = ['gdb', '--args'] + args
    print ' '.join(args)
    if call(args) != 0:
        os.remove(tmpPath)
        sys.exit(1)

    os.remove(tmpPath)

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
    print 'writing small.jpg'
    misc.imsave(join(cfg['outDir'], STACK_NAME, '0', 'small.jpg'), im)

    # write project.yaml file
    projStr = open('project.yaml').read()
    projStr = projStr.replace('{NAME}', cfg['projName'])
    print 'total pixel width/height: %d/%d' % (pxW, pxH)
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
        print 'actual dims: (%s, %s)' % (w, h)
        w = round(w / (2 ** SMALL_ZOOM))
        h = round(h / (2 ** SMALL_ZOOM))
        print 'cropped dims: (%s, %s)' % (w, h)
        im = im[:h, :w]
        print 'saving cropped.jpg'
        misc.imsave(join(imDir, 'cropped.jpg'), im)

##        print 'window size:', getWinSize(im)

def segmentation(fin):
    print fin
    cfg = json.load(open(fin))

    inDir = cfg['inDir']+'/stack1/0'
    outDir = cfg['outDir']
    with open(cfg['inDir']+'/project.yaml') as yaml:
        for line in yaml:
            if re.match('.*name: ".+"', line):
                projName = line.split('"')[1]
            if re.match('.*zoomlevels: .+', line):
                zoomLvls = line.split('zoomlevels: ')[1].rstrip()
    gridW = len(glob(inDir+'/0_*_0.tiff'))
    gridH = len(glob(inDir+'/*_0_0.tiff'))
    coreW = cfg['coreW']
    coreH = cfg['coreH']
    winW = cfg.get('winW',55)
    winH = cfg.get('winH',55)
    crop = cfg.get('crop',0)
	
    print 'inDir:', inDir
    print 'outDir:', outDir
    print 'projName:', projName
    print 'zoomLvls:', zoomLvls
    print 'gridW:', gridW
    print 'gridH:', gridH
    print 'coreW:', coreW
    print 'coreH:', coreH
    print 'winW:', winW
    print 'winH:', winH
    print 'crop:', crop

    sys.exit(0)
	
    args = []
    args.append(r'..\vsproject\superstitchous2013\x64\Release\segmentation.exe')
    args += [
    		inDir,
    		outDir,
    		projName,
			zoomLvls,
    		gridW,
    		gridH,
    		coreW,
    		coreH,
    		winW,
    		winH,
			crop
    		]
    args = map(str, args)
    print ' '.join(args)

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