# surgical-video-analysis

## Files and Scripts Used to Analyze SOCAL simulation data

### Referenced in: _Pilot Analysis of Surgeon Instrument Utilization Signatures Based on Shannon Entropy and Deep Learning for Surgeon Performance Assessment in a Cadaveric Carotid Artery Injury Control Simulation_

Example Bounding box video - Trial S102T2 - Green=ground truth, Blue=detected from YOLO model (using bbox_video_generator.py)
https://drive.google.com/file/d/1t9Pp4kXhnOFigSyO-WkpxN-M_ONKwNDo/view?usp=sharing
   
**INPUTS**:


a. Ground truth detections file (ex. socal.csv) has the true annotations in the following format:
	
	[frame] 			[x1]	[y1]	[x2]	[y2]	[label]

	S201T1_frame_00000001.jpeg	409	520	824	884	drill

  - frames with no labels have "N/A" in the columns

  - This information is used to create the ground truth annotation files for each frame used for training
	

b. socal_trial_outcomes.csv contains (along with other stuff) the size of the frames for each trial with the following column names:

	trial_video_width	trial_video_height

  - default trial size is 1920 x 1280 if the file does not contain a valid size (i.e. missing or N/A)
	
  - This information is used to ensure that each frame is scaled properly when adjusting the bounding box coordinates

**PROCESS**:

1. Convert the .csv file of ground truth detections into a set of corresponding .txt files that correspond
   to the yolo format. Each frame gets one .txt file. Use: csv_to_yolo.py

   - These .txt files are in a standardized format that is well-known
   - Alternatively, you can use XML_from_detections_list.py to convert the .csv file of ground truth detections into .xml files in the Pascal VOC XML format

3. For yolov4-darknet training:

	a. Convert the .xml annotation files into the yolov4 annotation format. Use: pascal_to_yolov4_annotations.py (if not already done)
	
	- This format has each frame represented by one .txt file (with the same name as the image). Each line of a file
		  contains the info for one label that is present on that frame. Format is:

   	`	[index of the tool class] [bbox center x] [bbox center y] [bbox width] [bbox height]  `

	- All the values here are normalized to the size of the frame. i.e. scaled to [0,1]

	b. Create the train.txt (and test.txt) files (IF NOT ALREADY DONE in Step a) with the proper image path within darknet directory.

	- This is just a list of all the image files (with full path name) needed for training or testing.

	c. To train yolov4-darknet, use the following guide: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

	- Follow the instruction on that page. ***BUT there are some inconsistencies potentially:

			1. Put all the image files (HAS TO BE .JPG file format) and .txt annotation files in data/obj/ directory
			2. Make sure that obj.data and obj.names files are in the data/ directory
			3. Put train.txt and test.txt in build/darknet/x64/data/ directory
			4. Make sure yolo-obj.cfg file is altered properly. 
				- Change random=1 to random=0 and mosaic=1 to mosaic=0 for CPU only training (i.e. no GPU)

			(for some reason, putting images in build/darknet/x64/data/obj/ as the instructions say does not seem to work)

5. For yolov4-darknet testing:

	a. Make sure that darknet was built/compiled properly and can perform inference (at least on the coco dataset)
	
	b. Ensure that the darknet directory has all the .cfg, .names, .data., .weights, and .jpg files all in the proper locations for training/testing. You can train on a different machine/AWS instance and simply copy the weights file into your local darknet directory. (Just make sure you know the path to it relative to darknet.exe)
	
	c. If using Windows machine **WITHOUT GPU** for testing darknet inference **WITH Python 3.8+** (older python3 version may not need this change):
	
	- Edit the "...Python3\Lib\ctypes\__init__.py file to change the default value for winmode to 0 (zero). 
		  (i.e. change to "winmode=0" in parameter list) This allows Python 3.8+ to access DLLs to run darknet inference
	
	![image](https://user-images.githubusercontent.com/55846088/154984599-4ad396f9-4cfb-4c3b-8c32-4ab146fad937.png)
	
	d. Use: darknet_detections.py with any needed modifications to perform inference on a directory of .jpg images and create the detections list (same format as ground truth detections in INPUTS section)


