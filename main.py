import cv2
import numpy as np

# read nura profile(screen shot of nura app)
img = cv2.imread("nura profile.png")

# convert image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("graytest.png", gray)

# find center of the circle profile
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, 
                           param1 = 50, param2 = 50, minRadius=150, maxRadius=500)

i = circles[0][0]
center = (int(i[0]), int(i[1]))
radius = int(i[2])

h, w = gray.shape

# cut img
cutimg = img[center[1]-center[0]:center[1]+center[0],:,:]
cv2.imwrite("cuttest.png", cutimg)

# find edge with multiple color space
edges_img = cv2.Canny(cutimg, 20, 100)
cutgray = cv2.cvtColor(cutimg, cv2.COLOR_BGR2GRAY)
edges_gray = cv2.Canny(cutgray, 20, 100)
cutblue = cutimg[:,:,0]
edges_blue = cv2.Canny(cutblue, 20, 100)
cutred = cutimg[:,:,2]
edges_red = cv2.Canny(cutred, 20, 100)
cutgreen = cutimg[:,:,1]
edges_green = cv2.Canny(cutgreen, 20, 100)
cutvalue = cv2.cvtColor(cutimg, cv2.COLOR_BGR2HSV)[:,:,2]
edges_value = cv2.Canny(cutvalue, 20, 100)
cutvalue = cv2.cvtColor(cutimg, cv2.COLOR_BGR2HSV)[:,:,1]
edges_saturation = cv2.Canny(cutvalue, 20, 100)
cutvalue = cv2.cvtColor(cutimg, cv2.COLOR_BGR2HSV)[:,:,0]
edges_hue = cv2.Canny(cutvalue, 20, 100)
cuthsv = cv2.cvtColor(cutimg, cv2.COLOR_BGR2HSV)
edges_hsv = cv2.Canny(cuthsv, 20, 100)

edges = (edges_img + edges_gray + edges_blue + edges_red + edges_green + 
         edges_value + edges_saturation + edges_hue + edges_hsv)
drawedges = cutimg.copy()
drawedges[edges != 0] = (0, 0, 0)
cv2.imwrite("drawedgestest.png", drawedges)


# draw the circle
for i in circles[0]:
    cv2.circle(gray, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 0), 1)


