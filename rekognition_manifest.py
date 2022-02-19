import os
from lxml import etree
import re
import json
from datetime import datetime
import glob
import random
from sklearn.model_selection import train_test_split


'''
Creates the AWS Rekognition manigest file for training and testing
'''

#creates the frame obj from the .xml file (PASCAL VOC format)
def make_frame_object_from_file(file_path, IMG_SIZE=(None, None), scale_up=True):
	annotation_xml = etree.parse(file_path)

	img_height = IMG_SIZE[1]
	img_width = IMG_SIZE[0]
	if IMG_SIZE[0] is None:
		# Set the image size
		for node in annotation_xml.iter('size'):
			for sn in node.iter('width'):
				img_width = float(sn.text)
			for sn in node.iter('height'):
				img_height = float(sn.text)

	data_to_append = {
		'name': file_path,
		'width': img_width,
		'height': img_height,
		'tools': []
	}

	# If we do NOT want to normalize the image dimensions
	if not scale_up:
		img_height = 1
		img_width = 1

	for obj in annotation_xml.findall('object'):
		true_class = obj.find('name').text
		true_class = re.sub('muscle patch', 'muscle', true_class)
		true_class = re.sub('other', 'tool', true_class)

		true_coordinates = []
		for corner in ['xmin', 'ymin', 'xmax', 'ymax']:
			true_coordinates.append(float(obj.find('bndbox').find(corner).text))

		data_to_append['tools'].append({
			'type': true_class,
			'coordinates': [
				(true_coordinates[0], true_coordinates[1]),  # (X1, Y1) , (X2, Y2)
				(true_coordinates[2], true_coordinates[3])
			]
		})

	return data_to_append

#just gets all the .xml file in a directory to a list
def getXMLsInDir(dir_path):
	image_list = []
	for filename in glob.glob(dir_path + '/*.xml'):
		image_list.append(filename)

	return image_list

def get_trial_test_set():
	return [

		'S201T1', 'S201T2',
		'S202T1', 'S202T2',
		'S203T1', 'S203T2',
		'S204T1', 'S204T2',
		'S205T1', 'S205T2',
		'S206T1', 'S206T2',
		'S207T1', 'S207T2',

		'S502T1', 'S502T2',
		'S502T2',
		'S504T1', 'S504T2',
		'S505T1', 'S505T2',
		'S506T1',
		'S507T1', 'S507T2'
	]


"""Get the list of trial in the validation split"""
def get_trial_validation_set():
	return [

		# New validation data
		'S502T1', 'S502T2',
		'S502T2',
		'S504T1', 'S504T2',
		'S505T1', 'S505T2',
		'S506T1',
		'S507T1', 'S507T2'
	]


def main():

	# first get the PASCAL XML files directory
	# Then, convert the XMLs into frame objects temporarily
	# Then, for each frame, add to manifest file.

	# PASCAL_XML_DIR = "C:\\Users\\Ganesh Balu\\Documents\\SOCAL_frames\\frames\\Pascal Format XML Annotations"
	PASCAL_XML_DIR = "C:\\Users\\reach\\Documents\\SOCAL\\frames\\Pascal Format XML Annotations"

	#where the FOLDERS that contains the other data are (images, xmls etc)
	#BASE_DIR = "C:\\Users\\Ganesh Balu\\Documents\\SOCAL_frames\\frames"
	BASE_DIR = "C:\\Users\\reach\\Documents\\SOCAL\\frames"
	#DARKNET_IMGS_DIR = "C:\\Users\\reach\\Documents\\darknet-master\\darknet-master\\data\\obj" #contains all frames used with darknet

	train_filename = BASE_DIR + "\\train_rekg_split2.manifest"
	test_filename = BASE_DIR + "\\test_rekg_split2.manifest"

	classes = ["drill", "suction", "muscle", "grasper", "cottonoid", "string", "scalpel", "tool"]

	image_paths = getXMLsInDir(PASCAL_XML_DIR)
	print(len(image_paths))

	# random.seed(123456)
	# random.shuffle(image_paths)
	#
	# #create the train and test splits (80/20)
	# darknet_train_images, darknet_test_images = train_test_split(image_paths, test_size=0.2)
	#
	# #dict to map image file to the right split
	# split_dict = {}
	# for img in darknet_train_images:
	# 	split_dict[img] = train_filename
	# for img in darknet_test_images:
	# 	split_dict[img] = test_filename

	#open the test manifest file
	test_file = open(test_filename, "w")

	with open(train_filename, "w") as train_file:

		#write the JSON line to the manifest files for each image
		for image_path in image_paths:

			frame = make_frame_object_from_file(image_path)  # creates a frame obj from the xml file

			frame_dict = {}

			frame_name = os.path.split(frame["name"])  # get the trial + frame number

			#S3 bucket location for the images (the bucket created by rekognition)
			#***make sure to use the right extension based on the images on S3 (i.e. .jpg vs .jpeg)
			frame_dict["source-ref"] = "s3://custom-labels-console-us-east-1-acd05dc007/images/" + frame_name[1][:-4] + ".jpeg"

			frame_dict_bounding_box = {}  # for the annotations
			frame_dict_metadata = {}  # for the metadata
			frame_dict_metadata["class-map"] = {}

			for index, tool in enumerate(classes):
				frame_dict_metadata["class-map"][index] = str(tool)

			frame_dict_metadata["type"] = "groundtruth/object-detection"
			frame_dict_metadata["human-annotated"] = "yes"
			frame_dict_metadata["creation-date"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
			frame_dict_metadata["job-name"] = "my job"

			frame_dict_bounding_box["image_size"] = [
				{"width": int(frame["width"]), "height": int(frame["height"]), "depth": 3}]

			tool_annotations = []
			metadata_objects = []

			# if(len(frame["tools"]) == 0):  #basically skip the images without any labels.
			# 	continue

			for tool in frame["tools"]:

				tool_dict = {"class_id": classes.index(tool["type"]), "top": int(tool["coordinates"][0][1]),
							 "left": int(tool["coordinates"][0][0]),
							 "width": int(tool["coordinates"][1][0]) - int(tool["coordinates"][0][0]),
							 "height": int(tool["coordinates"][1][1]) - int(tool["coordinates"][0][1])}

				tool_annotations.append(tool_dict)

				metadata_objects.append({"confidence": 1})

			frame_dict_bounding_box["annotations"] = tool_annotations
			frame_dict_metadata["objects"] = metadata_objects

			frame_dict_metadata2 = {"objects": metadata_objects, "class-map": frame_dict_metadata["class-map"],
									"type": frame_dict_metadata["type"],
									"human-annotated": frame_dict_metadata["human-annotated"],
									"creation-date": frame_dict_metadata["creation-date"],
									"job-name": frame_dict_metadata["job-name"]}

			frame_dict["bounding-box"] = frame_dict_bounding_box
			frame_dict["bounding-box-metadata"] = frame_dict_metadata2

			#add the frame to the right manifest file based on the split

			trial_id = os.path.splitext(os.path.basename(image_path))[0][:-15]

			if(trial_id in get_trial_test_set()):
				json.dump(frame_dict, test_file)
				test_file.write("\n")
			else:
				json.dump(frame_dict, train_file)
				train_file.write("\n")
			

			'''
			if(split_dict[image_path] == train_filename):
				json.dump(frame_dict, train_file)
				train_file.write("\n")
			else:
				json.dump(frame_dict, test_file)
				test_file.write("\n")'''

main()

'''

AWS manifest file JSON line format:

{
	"source-ref": "s3://custom-labels-bucket/images/IMG_1186.png",
	"bounding-box": {
		"image_size": [{
			"width": 640,
			"height": 480,
			"depth": 3
		}],
		"annotations": [{
			"class_id": 1,
			"top": 251,  -> y1
			"left": 399,  -> x1
			"width": 155,
			"height": 101
		}, {
			"class_id": 0,
			"top": 65,
			"left": 86,
			"width": 220,
			"height": 334
		}]
	},
	"bounding-box-metadata": {
		"objects": [{
			"confidence": 1
		}, {
			"confidence": 1
		}],
		"class-map": {
			"0": "Echo",
			"1": "Echo Dot"
		},
		"type": "groundtruth/object-detection",
		"human-annotated": "yes",
		"creation-date": "2013-11-18T02:53:27",
		"job-name": "my job"
	}
}

'''
