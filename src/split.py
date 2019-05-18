import os
import cv2
from WordSegmentation import wordSegmentation, prepareImg


def split(path):
	"""reads images from data/ and outputs the word-segmentation to out/"""

	count = 0
	# read input images from 'in' directory
	imgFiles = os.listdir(path + '/data/')
	for (i,f) in enumerate(imgFiles):
		# print('Segmenting words of sample %s'%f)
		
		# read image, prepare it by resizing it to fixed height and converting it to grayscale
		img = prepareImg(cv2.imread(path + '/data/%s'%f), 50)
		
		# execute segmentation with given parameters
		# -kernelSize: size of filter kernel (odd integer)
		# -sigma: standard deviation of Gaussian function used for filter kernel
		# -theta: approximated width/height ratio of words, filter function is distorted by this factor
		# - minArea: ignore word candidates smaller than specified area
		res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
		
		# write output to 'out/inputFileName' directory
		if not os.path.exists(path + '/out/%s'%(f.replace(".png", ""))):
			os.mkdir(path + '/out/%s'%(f.replace(".png", "")))
		
		# iterate over all segmented words
		# print('Segmented into %d words'%len(res))
		count = count + len(res)
		for (j, w) in enumerate(res):
			(wordBox, wordImg) = w
			(x, y, w, h) = wordBox
			cv2.imwrite(path + '/out/%s/%d.png'%(f.replace(".png", ""), j), wordImg) # save word
			cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
		
		# # output summary image with bounding boxes around words
		# cv2.imwrite(path + '/out/%s/summary.png'%f, img)
	return count

if __name__ == '__main__':
	main()