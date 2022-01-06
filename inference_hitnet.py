import numpy as np
import cv2
from time import time
from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img
from imread_from_url import imread_from_url


if __name__ == "__main__":

    # Select model type
	model_type = ModelType.middlebury
	# model_type = ModelType.flyingthings
	# model_type = ModelType.eth3d

	if model_type == ModelType.middlebury:
		model_path = "models/middlebury_d400/saved_model_240x320/model_float32.onnx"
	elif model_type == ModelType.flyingthings:
		model_path = "models/flyingthings_finalpass_xl/saved_model_240x320/model_float32.onnx"
	elif model_type == ModelType.eth3d:
		model_path = "models/eth3d/saved_model_240x320/model_float32.onnx"

	# Initialize model
	hitnet_depth = HitNet(model_path, model_type)

	# Load images
	left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Bicycle1-imperfect/im0.png")
	right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Bicycle1-imperfect/im1.png")

	
	# Estimate the depth
	start = time()
	disparity_map = hitnet_depth(left_img, right_img)

	color_disparity = draw_disparity(disparity_map)
	color_disparity = cv2.resize(color_disparity, (left_img.shape[1],left_img.shape[0]))
	print(f'Time taken to run: {time() - start} seconds')

	cobined_image = np.hstack((left_img, right_img, color_disparity))

	cv2.imwrite("output-4.jpg", cobined_image)