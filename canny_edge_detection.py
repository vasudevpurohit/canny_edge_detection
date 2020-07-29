import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('exit-ramp.jpg')

import cv2
grayscale_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
       ##imshow expects a 3 channel image by default and hence the cmap=grey would have to be defined to display a grayscale image



kernel_size = 5

smoothed_grayscale_image = cv2.GaussianBlur(grayscale_image,(kernel_size,kernel_size),0)



"""
grad_x = (cv2.Sobel(smoothed_grayscale_image,ddepth=-1,dx=1,dy=0,ksize=3))
grad_y = (cv2.Sobel(smoothed_grayscale_image,ddepth=-1,dx=0,dy=1,ksize=3))
"""

#exporting the gradients to 'gradients.csv'


low = 50
high = 150

edge = cv2.Canny(smoothed_grayscale_image, low, high)
mask = np.zeros_like(edge)
vertices = np.array([[(50,539),(900,539),(500,300),(420,300)]])
mask = cv2.fillPoly(mask,vertices,255)
masked_image = cv2.bitwise_and(edge,mask)

rho = 1
theta = np.pi/180
threshold = 25
max_length = 50
min_gap = 5

lines = cv2.HoughLinesP(masked_image,rho,theta,threshold,np.array([]),max_length,min_gap)

line_image=np.copy(image)*0
a=(np.size(lines))/4.0

for i in range(int(a)):
    j=0
    point1 = (lines[i][0][j],lines[i][0][j+1])
    point2 = (lines[i][0][j+2],lines[i][0][j+3])
    cv2.line(line_image,point1,point2,(255,0,0),3)
    
edge_3 = np.zeros((edge.shape[0],edge.shape[1],3),dtype="uint8")

edge_3[:,:,0] = edge
edge_3[:,:,1] = edge
edge_3[:,:,2] = edge

lines_edges = cv2.addWeighted(edge_3, 0.8, line_image, 1,0)
plt.imshow(lines_edges)
