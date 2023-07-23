import pandas as pd
import os
import re

classes = ["drill","suction","muscle","grasper","cottonoid","string","scalpel","tool"]

def convert(size, box):
    dw = 1. / (size[0])  # width
    dh = 1. / (size[1])  # height

    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1

    w = box[1] - box[0]
    h = box[3] - box[2]

    # coords scaled to the size of the image so always between [0, 1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_pvoc(size, box):
    dw = 1. / (size[0])  # width
    dh = 1. / (size[1])  # height

    w = box[1] - box[0]
    h = box[3] - box[2]

    # coords scaled to the size of the image so always between [0, 1]
    x = box[0] * dw
    w = w * dw
    y = box[2] * dh
    h = h * dh

    return (x, y, w, h)


def convert_frame_object_to_yolo(frame_obj, destination, prefix=''):

    frame_file_name = os.path.split(frame_obj["name"])[1]

    yolo_file_name = re.sub(r".jpeg|.jpg", '.txt', frame_file_name)

    if ("txt" in yolo_file_name):
        myfile = open(os.path.join(destination, yolo_file_name), "w")

        for tool in frame_obj["tools"]:

            cls_id = classes.index(tool["type"])
            b = float(tool["coordinates"][0][0]), float(tool["coordinates"][1][0]), float(tool["coordinates"][0][1]), float(tool["coordinates"][1][1])

            bb = convert_pvoc((int(frame_obj["width"]), int(frame_obj["height"])), b)

            # print(str(cls_id) + " " + str(frame_obj["conf"]) + " " + " ".join([str(a) for a in bb]) + '\n')
            #myfile.write(str(cls_id) + " " + str(tool["conf"]) + " " + " ".join([str(a) for a in bb]) + '\n')
            
            myfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        myfile.close()

def main():

    print("test")

    detections = pd.read_csv("socal.csv", names=["trial_frame", "x1", "y1", "x2", "y2", "label"])

    output_dir = "C:/Users/Alan/Documents/detections_yolov4_format"

    FRAME_YOLO_PATH = output_dir

    for unique_frame in detections.groupby(["trial_frame"]):

        is_nan = False

        # use dict to get the frame dimensions
        # unique_frames[0][0:-20] => take frame name and extract the trial name (alwasy 20 chars from end)
        img_width = trial_dict[unique_frames[0][0:-20]][0]  #for socal
        img_height = trial_dict[unique_frames[0][0:-20]][1]  #for socal

        tools = []

        for index, row in unique_frame[1].iterrows():

            is_nan = False  # to keep track of frame having no tools

            if (pd.isna(row["label"])):
                is_nan = True  # exit for loop it frame has no tools (no tools/bounding boxes needed)
                break

            tools.append({
                'type': row["label"],
                'conf': row["score"],
                'coordinates': [
                    (row["x1"], row["y1"]),
                    (row["x2"], row["y2"])
                ]})

            # create the frame object dict to feed into the xml creator function
        if (is_nan == False):

            frame_data = {
                'name': row["trial_frame"],
                'width': img_width,
                'height': img_height,
                'tools': tools
            }

        else:
            frame_data = {
                'name': row["trial_frame"],
                'conf': row["score"],
                'width': img_width,
                'height': img_height,
                'tools': []
            }

        convert_frame_object_to_yolo(frame_data, FRAME_YOLO_PATH)
        #exit()

if __name__ == "__main__":
    main()