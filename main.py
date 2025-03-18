import cv2
import numpy as np

# read nura profile(screen shot of nura app)
img = cv2.imread("nura profile.png")

# convert image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite("graytest.png", gray)

# find center of the circle profile
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, 
                           param1 = 50, param2 = 50, minRadius=150, maxRadius=500)

i = circles[0][0]
center = (int(i[0]), int(i[1]))
radius = int(i[2])

h, w = gray.shape

# cut img
cutimg = img[center[1]-center[0]:center[1]+center[0],:,:]
#cv2.imwrite("cuttest.png", cutimg)

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

# draw edge
draw = cutimg.copy()
draw[edges != 0] = (0, 0, 0)

# draw line for 31 band eq
theta = np.pi*0.0625 # 0.0625 = 2pi/32
eq_31band = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 
             1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
for n in range(1,32):
    linemask = np.zeros_like(edges)
    x, y = np.array([w/2,w/2]) + np.array([np.sin(theta*n), -np.cos(theta*n)])*w*0.45
    x, y = int(x), int(y)
    cv2.line(linemask, (w//2, w//2), (x, y), 200, 2)
    draw[linemask != 0] = (0, 0, 0)

    # find intersection of edge and line
    intersection = (edges != 0) * (linemask != 0)

    intersection_x, intersection_y = np.where(intersection)
    max_l = 0
    abs_max_l = 0
    point_x, point_y = np.array([w/2,w/2]) + np.array([np.sin(theta), -np.cos(theta)])*radius
    for i in range(len(intersection_x)):
        l = np.sqrt((intersection_x[i]-w/2)**2 + (intersection_y[i]-w/2)**2)
        # find distance from radius to intersection
        l_from_radius = (l-radius)/radius*10
        if abs(l_from_radius) > abs_max_l:
            max_l = l_from_radius
            abs_max_l = abs(l_from_radius)
            point_x, point_y = intersection_x[i], intersection_y[i]
            
    cv2.circle(draw, (point_y, point_x), radius=5, color=(0, 0, 200), thickness=-1)
    cv2.putText(draw, str((eq_31band[n-1], str(round(max_l, 1)))), (x-int(radius*0.12), y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200))
    eq_31band[n-1] = [eq_31band[n-1], max_l]

#cv2.imwrite("drawedgestest.png", draw)

# make txt for eq
txt_31eq = "GraphicEQ: "
for i in range(31):
    num = str(round(eq_31band[i][1]*-2, 2))
    txt_31eq += str(eq_31band[i][0]) + " " + num + "; "

f = open("equalizer apo 31 band.txt", "w")
f.write(txt_31eq[:-2])
f.close()
