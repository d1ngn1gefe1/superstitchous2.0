#Superstitchous 2.1
------------------
**Purpose:** Image stitching and core segmentation software for large scale digital holography (Quantitative Phase Imaging). You must cite this work by providing a reference to this page.

**Authors:** Kevin Han, Zelun Luo

**Mentor:** Mikhail Kandel

**Requirement:** 

* Visual Studio 2013 (Visual Studio 2012 will not work)

* [Python 3](https://www.python.org/downloads/)
    * Extension package: NumPy, SciPy, PIL
* [OpenCV 3.0 alpha](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0-alpha/) (OpenCV 2.x will not work)

* [Boost 1.56.0 ](http://www.boost.org/)

* [fftw](http://www.fftw.org/download.html) (provided)

* [libtiff](http://download.osgeo.org/libtiff/) (provided)

* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (provided)

* [tminres](https://code.google.com/p/tminres/) (provided)

* CMake for GCC (YMMV)

**Usage:**
For stitching and core segmentation, run

`python wrapper.py [-s <config file for stitch>] [-c <config file for seg>]`

Output of stitching is in a format suitable for uploading to [CATMAID](http://catmaid.org/index.html) or [Zoomify] (http://www.zoomify.com). 

Additionally, TIFF files are provided for inspection in ImageJ. 

The config file is JSON, consisting of a single object with these keys:

###Config file parameters for wrapper.py:
####Stitching:
Parameter                           | Description
----------------------------------- | ------------------------------------------
inDir                               | directory with poslist file; file name must end in "poslist"
outDir                              | output directory 
projName                            | project name for
zoomLvls                            | number of zoom levels

Optional parameter                  | Description | Default
----------------------------------- | ----------- | -------
pixelRatio							| Conversion from stage coordinates to pixels | 1 pixel/micron (off) 
maxPeakX                            | Max pixel offset of phase correlation peak from stage value for horizontally offset images. | dx / w \* 0.2 sqrt(w\*h), where dx is x overlap.
maxPeakY                            | Max pixel offset of phase correlation peak from stage value for vertically offset images. | dy / h \* 0.2 sqrt(w\*h), where dy is y overlap.
maxPeakXY                           | Max pixel offset of phase correlation peak from stage value for diagonally offset images. | (dx\*dy) / (w\*h) \* 0.2 sqrt(w\*h)
cacheSz                             | # images to cache when reading. | 50
poslist                             | Path of poslist relative to inDir. | Any file ending in "poslist" (can only be one)
skipAlign                           | `true` to skip aligment, and raster the "poslist" as-is | false
weightPwr                           | During least squares optimization, the error for each offset is weighted by `peak sum ^ weightPwr`. So, 0 = no weighting. | 1
peakRadius                          | Amount on each side of the peak to sum up. 0 = 1 pixel total (just the peak); 1 = 9 pixels total, etc. | 1
fixNaN                              | `true` to replace NaN values in image with zeros. | `true`
bgSub                               | percentage of input images used to generate the background. (0 if not doing background subtraction) | 0
outWidth                            | Rasterized tile width | 1024
outHeight                         	 | Rasterized tile height | 1024
writeJPG			    		 | Write JPGs? | `true`
writeTIF			   				| Write floating point TIF files? | `false`
colormapMin							| Color for pure black (0,0,0) |-0.3
colormapMax							| Color for pure white (1,1,1) |0.7
rasterFormat                        | Naming convention for files, choose from `CATMAID` or `ZOOMIFY` | `CATMAID`

The defaults for peak offset linearly extrapolate between 0% image overlap (0 peak offset; trust stage completely), and 100% overlap (peak offset is 0.2 * "image dimension"). (Maybe this will work?)

####Segmentation:
Parameter                           | Description
----------------------------------- | ------------------------------------------
inDir                               | directory with `project.yaml`; output directory of stitching.
outDir                              | output directory of segmented cores.
coreW                               | number of cores in each row.
coreH                               | number of cores in each column.

Optional parameter                  | Description | Default
----------------------------------- | ----------- | -------
winW                                | width of segmenting window in `cropped.jpg`. | 55
winH                                | height of segmenting window in `cropped.jpg`. | 55
crop                                | 1 = generate the output; 0 = only view the segmentation result in downsampled image. | 0
