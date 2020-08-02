import numpy as np
from scipy import misc
import cv2
from argparse import ArgumentParser

# a and b are numpy arrays
def phase_correlation(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    return r

def main():
    
    parser = ArgumentParser(description="Set parameters phase correlation calculation")
    
    parser.add_argument("infile1", metavar="in1", help="input image 1")    
    parser.add_argument("infile2", metavar="in2", help="input image 2")
    parser.add_argument("outfile", metavar="out", help="output image file name")
    
    args = parser.parse_args()
    
    infile1 = open(args.infile1)
    infile2 = open(args.infile2)
    outfile = args.outfile
    newfile = open(outfile, 'w')
    
    road1 = cv2.imread("./4.png")
    road2 = cv2.imread("./5.png")
    result = phase_correlation(road1, road2)
    cv2.imshow("s",result*1000)
    cv2.waitKey(0)
    infile1.close()
    infile2.close()
    newfile.close()

if __name__=="__main__":
    main()
