rows = 30
cols = 40
ratio=-6.3
xstep = ratio*80 # in pixels
ystep = ratio*100 #
outfile='testo.poslist'
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
