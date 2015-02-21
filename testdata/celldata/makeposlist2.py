#For DPM  2015
#From Points.xml
#xSize     ->offY*ratio
#qsbXSteps ->rows
#ySize->   ->offX*ratio
#qsbYSteps ->cols
rows = 4
cols = 4
#ratio = 15.6;
ratio=-6.2
#ratio=1
xstep = ratio*30 # in pixels
ystep = ratio*30 #
outfile='testo.poslist'
#templ = '%d_BEST.tif'
templ='0_0_0_%d_SLIM.tif'
#
def snake(i,c,r):
    row = int(i/c)
    rem = i%c
    col= rem if (row %2==0) else (c-1)-rem
    return (row,col)
#
#
lines=[]
n=cols*rows
for n in range(n):
    row,col=snake(n,cols,rows)
    name=templ %n
    lines.append('%s,%d,%d,0'%(name,xstep*col,ystep*row))
#    lines.append('%s,%d,%d,0'%(name,ystep*row,xstep*col))

f = open(outfile,'w')
f.write('\n'.join(lines))
f.close()
print(outfile)
