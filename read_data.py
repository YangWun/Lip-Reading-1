import os
import os.path
import scipy.io as matio
import cv2
import numpy as np


def read_data(path_read):
	index=0
	a =  os.walk(path_read)
	#For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames)

	height = -1
	width = -1
	num_sequence = 0

	for i in a:
		for file_name in i[2]:
			if file_name[0]=="." or file_name[0]=="#":
				continue
			this_file = path_read+file_name
			this_video = matio.loadmat(this_file)
			img_size = this_video['siz'][0]
			num_sequence += int (img_size[2])
			if height <0 and width <0:
				height = int (img_size[0])
				width = int (img_size[1])
			if (height >0 and height != img_size[0]) or (width >0 and width != img_size[1]):
				print ('image size should be equal'+'\n')
				return 0

	data_out = np.zeros((num_sequence,height,width))

	b =  os.walk(path_read)
	for j in b:
		for file_name in j[2]:								#process every video sequence in the folder
			if file_name[0]=="." or file_name[0]=="#":
				continue
			this_file = path_read+file_name
			this_video = matio.loadmat(this_file)
			temp_data = this_video['vid']
			img_size = this_video['siz'][0]
			for slice_index in range(int(img_size[2])):		#process every slice in the video sequence
				ttt = temp_data[:,slice_index]
				ttt = ttt.reshape((width,height))
				kk = np.transpose(ttt)						# for data imported from matlab we need to transpose the matrix
				data_out[slice_index,:,:] = kk
				# cv2.namedWindow("test",0)
				# cv2.imshow("test",kk)
				# cv2.waitKey()
				# cv2.destroyWindow("test")

	#assign 'vid' element in "dict" structure to data
	# # first_img = data_out[:,1]
	# # first_img = first_img.reshape((80,60))
	# # first_img = np.transpose(first_img)
    # #
    # #
    # #
    # #
    # #
	# # cv2.namedWindow("test",0)
	# # cv2.imshow("test",first_img)
	# # cv2.waitKey()
	# # cv2.destroyWindow("test")
	return data_out