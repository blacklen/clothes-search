import cv2
from mySegmentation import canny_grab, grab_cut, find_face, bg_removal, subtract_skin, subtract_hair, watershed
import os

image_path = "input/multi.jpg"
OUTPUT_DIR = "output"
GRAB_DIR = "grabcut"


img = cv2.imread(image_path)
# run canny edge detection
canny_img, tl, br = canny_grab(img)
roi = img[tl[1]:br[1], tl[0]:br[0]]
			
i = 0
the_face = None # the face isn't always found on the first iteration, so we search 5 times
while (the_face is None) and (i < 5):
	i += 1
	grab = grab_cut(img, tl, br)

	output_filename = os.path.join(GRAB_DIR,image_path.split('/')[-1])
	cv2.imwrite(output_filename, grab)

	the_face = find_face(grab)

	# remove any left over background in the image
	bg_mask = bg_removal(img, grab)
	bg_removed = cv2.subtract(grab, cv2.cvtColor(bg_mask,cv2.COLOR_GRAY2BGR))
				
	# if there is a face, we subtract the skin and the hair
	if the_face is not None:
		subtracted_skin_mask = subtract_skin(grab,the_face)
		subtracted_hair_mask = subtract_hair(grab,the_face)
		skin_removed = cv2.subtract(bg_removed, cv2.cvtColor(subtracted_skin_mask,cv2.COLOR_GRAY2BGR))
		hair_removed = cv2.subtract(skin_removed, cv2.cvtColor(subtracted_hair_mask,cv2.COLOR_GRAY2BGR))
		grab = hair_removed.copy()
	else: # could be improved by using kmeans to try and identify the skin when a face isn't found
		grab = bg_removed.copy()
	    
	# run the watershed algorithm to find connected components
	grab = cv2.GaussianBlur(grab, (15,15),0)
	watershed_out = watershed(grab)

	# subtract everything out but the foreground watershed piece 
	final_piece = cv2.bitwise_and(img, cv2.cvtColor(cv2.inRange(watershed_out,1,1),cv2.COLOR_GRAY2BGR))
				
	# write the piece to the output folder
	output_filename = os.path.join(OUTPUT_DIR,image_path.split('/')[-1])
	cv2.imwrite(output_filename, final_piece)
			