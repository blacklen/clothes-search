import cv2
import numpy as np

FRONTAL_FACE_HAAR_LOCATION = "haarcascade_frontalface_alt.xml"

def canny_grab(image):
	"""
	Run Canny Edge Detection on input image and find the bounding box that surrounds the edges
	input - cv2 Image
	output - grayscale mask for the edges, the top left corner of the bounding box, and the bottom right of the bounding box
	"""
	canny_out = cv2.Canny(image,0,image.flatten().mean())
	y,x = canny_out.nonzero()
	top_left = x.min(), y.min()
	bot_right = x.max(), y.max()
	return canny_out, top_left, bot_right

def grab_cut(image, top_left, bot_right):
	"""
	Utililize openCv's foreground detection algorithm
	input - cv2 image, top left and bottom right of a bounding box to focus on
	output - cv2 image of the estimated foreground
	"""
	mask = np.zeros(image.shape[:2],np.uint8)
	background = np.zeros((1,65),np.float64)
	foreground = np.zeros((1,65),np.float64)
	roi = (top_left[0],top_left[1],bot_right[0],bot_right[1])
	cv2.grabCut(image, mask, roi, background, foreground, 5, cv2.GC_INIT_WITH_RECT)
	new_mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	image = image*new_mask[:,:,np.newaxis]
	return image

def bg_removal(orig, grabbed):
	"""
	Remove additional background pixels after the initial grabcut
	input - cv2 image (the original image before the grabcut), the image after the grabcut
	output - a grayscale mask of background pixels
	"""
	kernel = np.ones((3,3),np.uint8)
	mean,std = cv2.meanStdDev(cv2.cvtColor(orig,cv2.COLOR_BGR2HLS), cv2.inRange(grabbed,0,0))
	min_thresh = mean - std
	max_thresh = mean + std
	grab_bg = cv2.inRange(cv2.cvtColor(grabbed,cv2.COLOR_BGR2HLS),min_thresh,max_thresh)
	dilated_bg = cv2.morphologyEx(grab_bg, cv2.MORPH_OPEN, kernel)
	return dilated_bg

def watershed(grabbed):
	"""
	Run the watershed algorithm to try and find connected components
	input - cv2 image
	output - cv2 image with the marked connected components
	"""
	gray = cv2.cvtColor(grabbed,cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=3)

	background = cv2.dilate(opening, kernel, iterations=2)

	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
	_, foreground = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

	foreground = np.uint8(foreground) 
	unknown = cv2.subtract(background,foreground)

	_, ccs = cv2.connectedComponents(foreground)
	ccs = ccs + 1
	ccs[unknown==255] = 0
	ccs = cv2.watershed(grabbed,ccs)
	return ccs

def get_skin_hair_mean_std(image, K = 2):
	"""
	Identify the skin pixel mean and standard deviation from a face image using Kmeans
	
	input - cv2 image of a human face, the number of clusters (K) for the Kmeans to reduce down to
	output - 
	"""
	data = image.reshape((-1,3))
	data = np.float32(data)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = K

	ret,label,center=cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	center = np.uint8(center)
	result = center[label.flatten()]
	result_2 = result.reshape((image.shape))

	one_count = np.sum(label)
	zero_count = label.size - one_count

	skin_label = 1 if one_count > zero_count else 0
	hair_label = 1 - skin_label

	skin_BGR = center[skin_label]
	hair_BGR = center[hair_label]
	skin_mean,skin_std = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), mask = cv2.inRange(result_2,skin_BGR,skin_BGR))
	hair_mean,hair_std = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), mask = cv2.inRange(result_2,hair_BGR,hair_BGR))

	return skin_mean, skin_std, hair_mean, hair_std

def subtract_skin(orig, face):
	"""
	Subtract the skin from an image given a facial crop
	
	input - cv2 image
	output - a grayscale mask of the skin in the image
	""" 
	skin_mean, skin_std, _, _ = get_skin_hair_mean_std(face,2)
	kernel = np.ones((3,3),np.uint8)
	min_thresh = skin_mean - (skin_std*2)
	max_thresh = skin_mean + (skin_std*2)
	min_thresh[2] = 0
	max_thresh[2] = 255
	grab_skin = cv2.inRange(cv2.cvtColor(orig,cv2.COLOR_BGR2HSV),min_thresh,max_thresh)
	dilated_skin = cv2.morphologyEx(grab_skin, cv2.MORPH_OPEN, kernel)
	return dilated_skin

def subtract_hair(orig, face): 
	"""
	Subtract the hair from an image given a facial crop
	input - cv2 image
	output - a grayscale mask of the hair in the image
	""" 
	_, _, hair_mean, hair_std = get_skin_hair_mean_std(face,2)
	kernel = np.ones((3,3),np.uint8)
	min_thresh = hair_mean - (hair_std*2)
	max_thresh = hair_mean + (hair_std*2)
	grab_hair = cv2.inRange(cv2.cvtColor(orig,cv2.COLOR_BGR2HSV),min_thresh,max_thresh)
	dilated_hair = cv2.morphologyEx(grab_hair, cv2.MORPH_OPEN, kernel)
	return dilated_hair

def find_face(image):
	"""
	Subtract the skin from an image
	input - cv2 image
	output - a grayscale image of the face
	""" 
	img2 = image.copy()
	gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier(FRONTAL_FACE_HAAR_LOCATION)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)

	the_face = None

	if len(faces) > 0:
	    face_imgs = list()
	    for (x,y,w,h) in faces:
	        face_imgs.append(img2[y:y+h, x:x+w])

	    face_sizes = np.array([face.size for face in face_imgs])
	    the_face = face_imgs[face_sizes.argmax()]

	return the_face