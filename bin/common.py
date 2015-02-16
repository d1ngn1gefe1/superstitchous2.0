import os
import sys
import re
import errno
from glob import glob
from itertools import takewhile
import functools #for reduce
import math #For ceil
def join(*args):
    return functools.reduce(os.path.join, args)

def mkdir(d):
    try:
        os.makedirs(d)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
    
def err(s, code=1):
    print( s)
    sys.exit(code)

def parsePoslist(poslistLines, poslistDir,ratio):
    """
    Parse a poslist file and return:
    
    (list of image paths, list of (x, y) positions, snakedir, cols, rows, x offset, y offset)
    """
    imFiles = []
    coords = []
    for l in poslistLines:
        #print(l)
        if l.strip().startswith('#') or l.strip().startswith('dim = '):
            continue
        imName = re.search(r'\w+\.tiff?', l).group(0)
        imFiles.append(os.path.abspath(join(poslistDir, imName)))
        coMatch = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(?:,|\s)+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', l)
        coords.append((float(coMatch.group(1))*ratio, ratio*float(coMatch.group(2))))
    
    #Todo add support stitching a regularly spaced list 
    '''
    posA=coords[0][0]
    posB=coords[1][0]
    posC=coords[2][0]
    posAA=coords[0][1]
    posBB=coords[1][1]
    posCC=coords[2][1]
    print(posA,posB,posC)
    print(posAA,posBB,posCC)
#   print(posA,posB)
#   print(posC,posD)
    '''
    #snakedir = 'col' means 1st dim is constant for a row
    #snakedir = 'row' means 2nd dim is constant for a row
    rowFirstDim = (coords[0][0] == coords[1][0])
    rowSecondDim = (coords[0][1] == coords[1][1])
    if rowFirstDim:
        snakeDir = 'col'
        rows = len(list(takewhile(lambda c: c[0] == coords[0][0], coords)))
        cols = math.ceil(len(coords) / rows) #Change for python3
        yOff = coords[1][1] - coords[0][1]
        if cols == 1:
            xOff = 0
        else:
            xOff = coords[rows][0] - coords[0][0]
    elif rowSecondDim:
        snakeDir = 'row'
        cols = len(list(takewhile(lambda c: c[1] == coords[0][1], coords)))
        rows = math.ceil(len(coords) / cols)
        xOff = coords[1][0] - coords[0][0]
        if rows == 1:
            yOff = 0
        else:
            yOff = coords[cols][1] - coords[0][1]
    else:
        snakeDir = 'none'
        cols = -1
        rows = -1
        xOff = -1
        yOff = -1
    return imFiles, coords, snakeDir, cols, rows, xOff, yOff

def parsePoslistDir(poslistDir, poslistPath,ratio):
    if poslistPath is None:
        files = glob(join(poslistDir, '*poslist'))
        if len(files) > 1:
            err('err: more than one poslist file')
        poslistPath = files[0]
    else:
        poslistPath = poslistPath

    return parsePoslist(open(poslistPath).read().splitlines(), poslistDir,ratio)

def getImgI(x, y, rows):
    i = x * rows
    if x % 2:
        i += rows - y - 1
    else:
        i += y
    return i

def toAbsolutePath(partialfile):
    fullpath=os.path.abspath(partialfile)
    #Somekind of error correction goes here
    return fullpath
    
#class TileFormatCatmaid(object):
#     #provides format specific tile information
#    
#    def filenames(row,col,zoom):        
#        return '%d_%d_%d.jpg' % (row,col,zoom)
#    
#    def count(dim,zoom):
#        selector=['0_*_%d.jpg','*_0_%d.jpg']   
#        return len(glob(getPath(selector[dim] % zoom)));
#    