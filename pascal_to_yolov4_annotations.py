import glob
import os
import xml.etree.ElementTree as ET

'''
Convert the PASCAL VOC format .xml file into a yolov4 annotation .txt file
'''

#directories to search for files

#dirs = ['C:\\Users\\Ganesh Balu\\Documents\\SOCAL_frames\\frames']
dirs = ['C:\\Users\\reach\\Documents\\SOCAL\\frames']
classes = ["drill","suction","muscle","grasper","cottonoid","string","scalpel","tool"]

#create a list of all image (.jpeg and .jpg) files in the directory
def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpeg'):
        image_list.append(filename)

    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list

#converts the bounding box (min and max for x and y) into yolov4 format (center coords and width/height)
def convert(size, box):
    dw = 1./(size[0]) #width
    dh = 1./(size[1]) #height
    
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    
    w = box[1] - box[0]
    h = box[3] - box[2]

    #coords scaled to the size of the image so always between [0, 1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    #open the xml annotation file
    try:
        in_file = open(xml_path + "\\" + basename_no_ext + '.xml')
    except:
        print("no info for trial with: ", xml_path + "\\" + basename_no_ext + '.xml')
        #for trials where the frame size was not found in file -> probably need to correct those later
        return False

    out_file = open(output_path + "\\" + basename_no_ext + '.txt', 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    #trial = root.find("filename")[0:6]

    tools = [i.find('name').text for i in root.iter('object')]

    # if(len(tools) == 0):
    #     return False

    #find the class in the list of classes and use that index for the annotation
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue #skip the object if it is not in the class list (i.e. dont add to the xml file)
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    out_file.close()

    return True  #means that the xml file was created

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
        'S507T1' , 'S507T2'
    ]

def main():

    #for each directory of images, create the xml files appropriately
    for dir_path in dirs:

        full_img_path = dir_path + "\\JPEGImages"
        #full_dir_path = "C:\\Users\\reach\\Documents\\darknet-master\\darknet-master\\data\\obj"
        full_xml_path = dir_path + "\\Pascal Format XML Annotations"

        output_path = dir_path + "\\yolo"

        print(full_img_path)

        image_paths = getImagesInDir(full_img_path)

        #file path with the list of all images (needed for darknet -> rename to train.txt)
        list_file = open(dir_path + '\\train_sess_split.txt', 'w')

        test_counter = 0

        for image_path in image_paths:

            basename = os.path.basename(image_path)
            basename_no_ext = os.path.splitext(basename)[0]

            trial_id = os.path.splitext(os.path.basename(image_path))[0][:-15]

            done = True
            # converts the PASCAL to yolov4 format and saves it
            #done = convert_annotation(full_xml_path, output_path, image_path)

            #this add the image to the darknet train.txt file. Correct path based on AlexeyAB dir structure.
            #only do it if .xml file was created and saved
            if done:
                #change this line if you need using with a directory of .jpg files
                if (trial_id not in get_trial_test_set()):

                    list_file.write("data/obj/" + basename_no_ext + ".jpg" + "\n")

            test_counter += 1

        list_file.close()

        print("Finished processing: " + dir_path)


main()
