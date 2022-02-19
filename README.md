# surgical-video-analysis


Bounding box video - Trial S102T2 - Green=ground truth, Blue=detected from YOLO model
https://drive.google.com/file/d/1t9Pp4kXhnOFigSyO-WkpxN-M_ONKwNDo/view?usp=sharing

Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@alanbalugu 
alanbalugu
/
surgical-video-analysis
Public
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
surgical-video-analysis/Steps.txt
@alanbalugu
alanbalugu Add files via upload
Latest commit 10a005c 13 seconds ago
 History
 1 contributor
75 lines (45 sloc)  3.78 KB
   
INPUTS:

- Ground truth detections file (ex. socal.csv) has the true annotations in the following format:

	frame				x1	y1	x2	y2	label

	S201T1_frame_00000001.jpeg	409	520	824	884	drill

	**frames with no labels have "N/A" in the columns
	
	- This information is used to create the ground truth annotation files for each frame used for training

- socal_trial_outcomes.csv contains the size of the frames for each trial with the following column names:

	trial_video_width	trial_video_height

	**default trial size is 1920 x 1280 if the file does not contain a valid size (i.e. missing or N/A)
	
	- This information is used to ensure that each frame is scaled properly when adjusting the bounding box coordinates

PROCESS:

1. Convert the .csv file of ground truth detections into a set of corresponding .xml files that correspond
   to the PASCAL VOC labeling format. Each frame gets one .xml file. Use: XML_from_detections_list.py
	
	- These .xml files are in a standardized format that is well-known

2. For yolov4-darknet training:

	a. Convert the .xml annotation files into the yolov4 annotation format. Use: pascal_to_yolov4_annotations.py

		- This format has each frame represented by one .txt file (with the same name as the image). Each line of a file
		  contains the info for one label that is present on that frame. Format is:

			[index of the tool class] [bbox center x] [bbox center y] [bbox width] [bbox height]

			** all the values here are normalized to the size of the frame. i.e. scaled to [0,1]

		**the script also creates the train.txt file with the paths to all training images. May need to adjust if you put images in a different path
		   within darknet

	b. Create the train.txt (and test.txt) files (IF NOT ALREADY DONE in Step a) with the proper image path within darknet directory. Use: pascal_to_yolov4_annotations.py
		**This step uses the directory of .xml annotation files (PASCAV VOC format)

	c. To train yolov4-darknet, use the following guide: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

		- Follow the instruction on that page. ***BUT there are some inconsistencies potentially:

			1. Put all the image files (HAS TO BE .JPG file format) and .txt annotation files in data/obj/ directory
			2. Make sure that obj.data and obj.names files are in the data/ directory
			3. Put train.txt and test.txt in build/darknet/x64/data/ directory
			4. Make sure yolo-obj.cfg file is altered properly. 
				- Change random=1 to random=0 and mosaic=1 to mosaic=0 for CPU only training (i.e. no GPU)

			(for some reason, putting images in build/darknet/x64/data/obj/ as the instructions say does not seem to work)


3. For AWS Rekognition training/testing:

	a. Create a new project in AWS Rekognition.
	
	b. Upload all .jpeg or .jpg (file extension doesn't matter here) to the aws s3 bucket created by the project (has a long name)
	
	c. Create the manifest files (for training and testing) that contains all the annotations for each frame as JSON lines. Use: rekognition_manifest.py
		**This step uses the directory of .xml annotation files (PASCAV VOC format) to create the manifest files
		
	d. Upload the manifest files to the same s3 bucket as used before
	
	e. Create the train and test datasets through the rekognition webpage (click SageMaker annotation option to use the manifest files from Step c)
	
	d. Navigate to "dataset" page and check to make sure images have no errors, correct number of labeled and unlabeled, bboxes are shown etc
	
	e. Train the model
		**Training cannot be stopped after started, so be sure that there are no issues before it starts. It will charge you ONLY if it succeeds.
		
	f. Look at the test set metrics after training is done.

4. (NOT DONE YET) For yolov4-darknet testing:

	a. Use: yolov4-test.py 

