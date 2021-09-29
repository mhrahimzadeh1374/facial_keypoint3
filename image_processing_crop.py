

import cv2
import numpy as np
import argparse
import csv

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='../source.jpg', help='path to image')
    opt = parser.parse_args()
    return(opt)

if __name__ == "__main__":
    opt = parse_opt()
    img=cv2.imread(opt.source) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = 2)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    
    edged = cv2.Canny(dilation, 30, 200)
    
    x=[]
    y=[]
    for yp in range(edged.shape[0]):
        for xp in range(edged.shape[1]):
            if edged[yp,xp]==255:
                x.append(xp)
                y.append(yp)
                          
    with open('results.csv','w',newline='') as csvf:
        csvw=csv.writer(csvf)
        csvw.writerow([opt.source,min(x),min(y),max(x),max(y),1])        
