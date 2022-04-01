import cv2
import numpy as np
import queue
from math import pow, sqrt
from imutils.perspective import four_point_transform

def dist(p1, p2):
    delta_y = (p1[0] - p2[0]) ** 2
    delta_x = (p1[1] - p2[1]) ** 2
    return sqrt(delta_y + delta_x)

def get_neighbors(img, p):
    neighbors = []
    y, x = p
    height, width = img.shape

    if y-1 > 0:
        neighbors.append((y-1,x))
    if x-1 > 0:
        neighbors.append((y,x-1))
    if x+1 < width:
        neighbors.append((y,x+1))
    if y+1 < height:
        neighbors.append((y+1,x))
    return neighbors

def nearest_barrier(img, p):
	searched = [p]
	toSearch = get_neighbors(img, p)
	while len(toSearch) != 0:
		pix = toSearch.pop(0)
		if img[pix[0]][pix[1]] == 0:
			return 0 - dist(pix, p)
		searched.append(pix)
		neighbors = get_neighbors(img, pix)
		for n in neighbors:
			if not n in searched:
				toSearch.append(n)
	return 0

def a_star(img, start, goal):
    """
    This function finds the shortest path between a start and goal point, where each point
    is a tuple of values in the form (y, x)
    """
    visited_nodes = []
    frontier = queue.PriorityQueue()

    start_cost = dist(start, goal) #+ nearest_barrier(img, start)
    frontier.put((start_cost, [start]))

    while not frontier.empty():
        (current_cost, current_path) = frontier.get()
        current_node = current_path[-1]

        if current_node not in visited_nodes:
            # Record we've been to the current node
            visited_nodes.append(current_node)

            # Check if we're at the goal
            if current_node == goal:
                return current_path

            # Goal not found, continue searching
            new_paths = []

            # Cycle through each of the surrounding nodes
            for node in get_neighbors(img, current_node):
                if img[node[0]][node[1]] == 255:
                    # Create a path including the new node, calculate cost
                    new_path = current_path + [node]
                    new_cost = len(new_path) + dist(node, goal) #+ nearest_barrier(img, node)
                    
                    # Save the path as a new path
                    new_paths.append((new_cost, new_path))

            # Add each of the new paths to the frontier
            for (new_cost, new_path) in new_paths:
                frontier.put((new_cost, new_path))

    # If no path is found, return nothing
    return []

def draw_path(img,path):
    for p in path:
        cv2.circle(img, (p[1], p[0]), 5, (125), -1)
	


image = cv2.imread('actual3.png')
orig = image.copy()

# Add Black bar buffer
buffer = np.zeros((400,orig.shape[1],orig.shape[2]), dtype=np.uint8)
new_img = np.concatenate((orig, buffer), axis=0)
# cv2.imwrite("aaa.png", new_img)

# Warp image
points = np.array([[426,339], [1629,0], [242,800], [1757,1471]])
warped = four_point_transform(new_img, points)
# cv2.imshow("aa", warped)
cv2.imwrite("aaaaaaaaaaa.png", warped) #TESTING

# Erode image
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
kernel = np.ones((7,7), np.uint8)
(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ero = cv2.dilate(im_bw, kernel, iterations=5)
cv2.imwrite("bbbb.png", ero) #TESTING

# Resize image
(height,width) = ero.shape
scale = 12
resize = cv2.resize(ero,(int(width/scale),int(height/scale)))
cv2.imwrite("ccccccc.png", resize) #TESTING

(height,width) = resize.shape
print((height-5, 5), (5, width-5))
path = a_star(resize, (100, 126), (2, 2))
print(path)
draw_path(resize,path)


cv2.imshow("aa", resize)
# cv2.imshow("Warped", warped)
cv2.waitKey(0)
