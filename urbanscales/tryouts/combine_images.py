import os, numpy, PIL
from PIL import Image
from tqdm import tqdm 

# Access all PNG files in directory
allfiles=os.listdir(os.getcwd())
imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"] and "e25" in filename]

# Assuming all images are the same size, get dimensions of first image
w,h=Image.open(imlist[0]).size
N=len(imlist)
# N = 100

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((h,w,3),numpy.float)

# Build up average pixel intensities, casting each image as an array of floats
for i in tqdm(range(N)):
    im = imlist[i]
    imarr=numpy.array(Image.open(im),dtype=numpy.float)
    imarr = imarr[:,:,:3]
    arr=arr+imarr/N
arrcopy = numpy.array(arr)

# Round values in array and cast as 8-bit integer

arr = numpy.array(arrcopy)

arr=numpy.array(numpy.round(arr*275*N)/N,dtype=numpy.uint8)

# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
out.save("Average.png")
out.show()
