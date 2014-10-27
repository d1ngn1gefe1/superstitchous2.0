import os
import sys
import re
import errno
from glob import glob
from itertools import takewhile

def join(*args):
    return reduce(os.path.join, args)

def mkdir(d):
    try:
        os.makedirs(d)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
    
def err(s, code=1):
    print s
    sys.exit(code)

def parsePoslist(poslistLines, poslistDir):
    """
    Parse a poslist file and return:
    
    (list of image paths, list of (x, y) positions, snakedir, cols, rows, x offset, y offset)
    """
    imFiles = []
    coords = []
    for l in poslistLines:
        if l.strip().startswith('#') or l.strip().startswith('dim = '):
            continue

        imName = re.search(r'\w+\.tiff?', l).group(0)
        imFiles.append(os.path.abspath(join(poslistDir, imName)))

        coMatch = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(?:,|\s)+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', l)
        coords.append((float(coMatch.group(1)), float(coMatch.group(2))))
    
    if coords[0][0] == coords[1][0]:
        snakeDir = 'col'
        rows = len(list(takewhile(lambda c: c[0] == coords[0][0], coords)))
        cols = len(coords) / rows
        yOff = coords[1][1] - coords[0][1]
        if cols == 1:
            xOff = 0
        else:
            xOff = coords[rows][0] - coords[0][0]
    elif coords[0][1] == coords[1][1]:
        snakeDir = 'row'
        cols = len(list(takewhile(lambda c: c[1] == coords[0][1], coords)))
        rows = len(coords) / cols
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

def parsePoslistDir(poslistDir, poslistPath=None):
    if poslistPath is None:
        files = glob(join(poslistDir, '*poslist'))
        if len(files) > 1:
            err('err: more than one poslist file')
        poslistPath = files[0]
    else:
        poslistPath = join(poslistDir, poslistPath)

    return parsePoslist(open(poslistPath).read().splitlines(), poslistDir)

def getImgI(x, y, rows):
    i = x * rows
    if x % 2:
        i += rows - y - 1
    else:
        i += y
    return i
