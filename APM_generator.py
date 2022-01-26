import os

import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter
import statistics
import math
import matplotlib.pyplot as plt

def normalize_coords(data, frame_width, frame_height):

    data["x1"] = data["x1"] / frame_width
    data["x2"] = data["x2"] / frame_width
    data["y1"] = data["y1"] / frame_height
    data["y2"] = data["y2"] / frame_height

    return data

def un_normalize_coords(data, frame_width, frame_height):

    data["x1"] = data["x1"] * frame_width
    data["x2"] = data["x2"] * frame_width
    data["y1"] = data["y1"] * frame_height
    data["y2"] = data["y2"] * frame_height

    return data

def add_video_info(data, file_id, width, height, frames):

    data["file_id"] = []
    data["width"] = []
    data["height"] = []
    data["total_frames"] = []

    new_data = {"file_id": file_id, "width": width, "height": height, "total_frames": frames}
    data = data.append(new_data, ignore_index=True)

    return data

def count_frames_w_tool(data, search_thresh, tool):

    frames = len(data.loc[(data["label"] == tool) & (data["score"] > search_thresh)]["frame"].unique())
    return frames

def count_frames_w_x_tools(data, num_tools):

    frames_w_tools = [len(list(j)) for i,j in groupby(list(data["frame"]))]

    return frames_w_tools.count(num_tools)

def find_first_frame_w_tool(data, search_thresh, tool):

    try:
        return list(data.loc[(data["label"] == tool) & (data["score"] > search_thresh)]["frame"])[0]
    except:
        return "na"

def calc_in_n_outs(data, search_thresh, tool):

    tool_filtered = data.loc[(data["label"] == tool) & (data["score"] > search_thresh)]

    unique_instances = list(tool_filtered["frame"].unique())

    ranges =[]

    for key, group in groupby(enumerate(unique_instances), lambda i: i[0] - i[1]):         
        group = list(map(itemgetter(1), group))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))

    return len(ranges)

def calc_bounding_box_area(x1, y1, x2, y2):

    h = y2 - y1 + 1
    w = x2 - x1 + 1 

    return float(h*w)

def get_bounding_box_list_df(tool_df):

    return [ tool_df["x1"].iloc[0], tool_df["y1"].iloc[0], tool_df["x2"].iloc[0], tool_df["y2"].iloc[0] ]

def get_bounding_box_list_row(tool_row):

    return [ tool_row["x1"], tool_row["y1"], tool_row["x2"], tool_row["y2"] ]


def calc_tools_overlap_area(a, b):


    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])

    if (dx >= 0) and (dy >= 0):
        return (dx * dy)
    else:
        return 0.0

def calc_total_tool_area(data, search_thresh, tool):

    tool_filtered = data.loc[(data["label"] == tool) & (data["score"] > search_thresh)] #gets all rows with the tool

    #for each row, calculate area

    x1s = tool_filtered["x1"].tolist()
    x2s = tool_filtered["x2"].tolist()
    y1s = tool_filtered["y1"].tolist()
    y2s = tool_filtered["y2"].tolist()

    total_area = 0

    for index in range(0, len(list(tool_filtered["frame"]))):
        
        total_area += calc_bounding_box_area(x1s[index], y1s[index], x2s[index], y2s[index])

    return total_area

def calc_tool_coord_sd(data, search_thresh, tool, coord):

    try:
        return statistics.stdev(data.loc[(data["label"] == tool) & (data["score"] > search_thresh)][coord].tolist())
    except:
        return 0 

def calc_tool_coord_difference_sd(data, search_thresh, tool, coord1, coord2):

    try:

        tool_filtered = data.loc[(data["label"] == tool) & (data["score"] > search_thresh)]

        center_data = [ (coord2_i + coord1_i)/2 for coord1_i, coord2_i in zip(tool_filtered[coord1].tolist(), tool_filtered[coord2].tolist())]

        return statistics.stdev(center_data)

    except:
        return 0 

def calc_tool_distance_covered(data, search_thresh, tool):

    tool_filtered = data.loc[(data["label"] == tool) & (data["score"] > search_thresh)]

    unique_instances = list(tool_filtered["frame"].unique())

    ranges = []

    for key, group in groupby(enumerate(unique_instances), lambda i: i[0] - i[1]):         
        group = list(map(itemgetter(1), group))
        group = list(map(int,group))
        ranges.append(group)

    consec_coords = []

    total_tool_dist = 0
    total_tool_frames = 0

    #print(ranges)

    for consec_frames in ranges:    #gives a list of frames with the tools at search

        for frame in consec_frames:

            right_frame_df = tool_filtered.loc[tool_filtered["frame"] == frame].sort_values(by=['x1'])

            if( (tool == "cottonoid") or (tool == "muscle") ):
                consec_coords.append([ (right_frame_df["x2"].iloc[0] + right_frame_df["x1"].iloc[0])/2, (right_frame_df["y2"].iloc[0] + right_frame_df["y1"].iloc[0])/2])

            else:
                consec_coords.append([ (right_frame_df["x2"].iloc[0] + right_frame_df["x1"].iloc[0])/2, right_frame_df["y1"].iloc[0] ])

            #if ( len(right_frame_df["x1"].tolist()) > 1 ):
            #    print("****", tool, right_frame_df["x1"].tolist())

        #print(consec_coords)

        for index in range(1, len(consec_coords)):

            dist = math.sqrt( (consec_coords[index][1] - consec_coords[index-1][1])**2 + (consec_coords[index][0] - consec_coords[index-1][0])**2 )

            total_tool_frames += 1

            total_tool_dist += dist

        consec_coords = []

    try:
        total_tool_speed = (total_tool_dist/total_tool_frames)
    except:
        total_tool_speed = 0.0

    return total_tool_dist, total_tool_speed


#***problem here, because some objects with high overlap are labeled with the different labels
# for each frame, compare all bounding boxes (like before). If any 2 boxes have too high of IOU, 0.9??, then remove the detection
#    with the the lower score


def get_high_score_tools(data, tools, best_tool_thresholds):

    high_score_data = pd.DataFrame()

    for tool in tools:

        high_score_data = high_score_data.append( data.loc[ (data["label"] == tool) & (data["score"] >=  float(best_tool_thresholds[tool])) ],  ignore_index=True )

    high_score_data = high_score_data.sort_values(by=["frame"])

    rows_to_drop = []

    for frame in high_score_data["frame"].unique():

        tools_in_frame_indices = high_score_data.loc[high_score_data["frame"] == frame].index.tolist()
        tools_in_frame = list(high_score_data.loc[high_score_data["frame"] == frame]["label"])

        #print(tools_in_frame, frame)

        if(len(tools_in_frame) != 1) :
            
            for tool1_index in range(0, len(tools_in_frame)):

                for tool2_index in range(tool1_index+1, len(tools_in_frame)):

                    #print(tools_in_frame[tool_index], tools_in_frame[tool2_index])

                    tool1_row = high_score_data.loc[(high_score_data["frame"] == frame) & (high_score_data["label"] == tools_in_frame[tool1_index]) & (high_score_data.index == tools_in_frame_indices[tool1_index])]
                    tool2_row = high_score_data.loc[(high_score_data["frame"] == frame) & (high_score_data["label"] == tools_in_frame[tool2_index]) & (high_score_data.index == tools_in_frame_indices[tool2_index])]

                    if(tools_in_frame[tool1_index] != tools_in_frame[tool2_index]):

                        iou = calc_iou(get_bounding_box_list_df(tool1_row), get_bounding_box_list_df(tool2_row))

                        if(iou > 0.85):

                            #print("two boxes are the same, but different")

                            #print(frame, iou, tool1_index, tools_in_frame[tool1_index], tool2_index, tools_in_frame[tool2_index])

                            if(float(tool1_row["score"].iloc[0]) > float(tool2_row["score"].iloc[0])):

                                rows_to_drop.append(tool2_row.index[0])
                                #print(frame, tool2_row.index[0], tools_in_frame[tool2_index], " vs ", tools_in_frame[tool1_index])
                            
                            else:

                                rows_to_drop.append(tool1_row.index[0])
                                #print(frame, tool1_row.index[0], tools_in_frame[tool1_index], " vs ", tools_in_frame[tool2_index])
                            
    high_score_data.drop(rows_to_drop, axis=0, inplace=True)

    return high_score_data


#***This probably does not work properly....change to way Guillaume did it...(merge dfs with the 2 tools. Then, sort by frame. For a frame, get IOU for the two tools. If high enough, count frame)
def calc_total_tool_overlap(high_score_tools, search_thresh, tool1, tool2, overlap_thresh = 0.1):

    total_overlap = 0

    overlap_frames = 0

    #high_score_tools = data.loc[data["score"] > search_thresh]

    for frame in high_score_tools["frame"].unique():

        tools_in_frame_indices = high_score_tools.loc[high_score_tools["frame"] == frame].index.tolist()
        tools_in_frame = list(high_score_tools.loc[high_score_tools["frame"] == frame]["label"])

        #print(tools_in_frame, frame)

        if(len(tools_in_frame) != 1) :
            
            for tool1_index in range(0, len(tools_in_frame)):

                for tool2_index in range(tool1_index+1, len(tools_in_frame)):

                    #print(tools_in_frame[tool_index], tools_in_frame[tool2_index])

                    tool1_row = high_score_tools.loc[(high_score_tools["frame"] == frame) & (high_score_tools["label"] == tools_in_frame[tool1_index]) & (high_score_tools.index == tools_in_frame_indices[tool1_index])]
                    tool2_row = high_score_tools.loc[(high_score_tools["frame"] == frame) & (high_score_tools["label"] == tools_in_frame[tool2_index]) & (high_score_tools.index == tools_in_frame_indices[tool2_index])]

                    if(tools_in_frame[tool1_index] != tools_in_frame[tool2_index]):

                        if( ((tool1_row["label"].iloc[0] == tool1) and (tool2_row["label"].iloc[0] == tool2)) or ((tool1_row["label"].iloc[0] == tool2) and (tool2_row["label"].iloc[0] == tool1)) ):

                            overlap_area = calc_tools_overlap_area(get_bounding_box_list_df(tool1_row), get_bounding_box_list_df(tool2_row))

                            #print(frame, calc_iou(get_bounding_box_list_df(tool1_row), get_bounding_box_list_df(tool2_row)), tool1_index, tools_in_frame[tool1_index], tool2_index, tools_in_frame[tool2_index])
                            #print(tool1_row)
                            #print(tool2_row)

                            if( calc_iou(get_bounding_box_list_df(tool1_row), get_bounding_box_list_df(tool2_row)) > overlap_thresh):

                                overlap_frames += 1

                            total_overlap += overlap_area

    return total_overlap, overlap_frames

def get_video_dimensions(trial_ID):

    input_df = pd.read_csv("socal_trial_outcomes.csv")
    height = input_df.loc[input_df["trial_id"] == trial_ID]["trial_video_height"].iloc[0]
    width = input_df.loc[input_df["trial_id"] == trial_ID]["trial_video_width"].iloc[0]

    return width, height

def get_num_frames(trial_ID):

    input_df = pd.read_csv("frame_to_trial_mapping.csv")

    max_frame_num = max(input_df.loc[input_df["trial_id"] == trial_ID]["frame_number"].tolist())

    return max_frame_num


# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def calc_iou(boxA, boxB):
    # if boxes dont intersect
    if do_boxes_intersect(boxA, boxB) is False:
        return 0
    interArea = get_Intersection_Area(boxA, boxB)
    union = get_Union_Area(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    return iou

# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def do_boxes_intersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def get_Intersection_Area(b1, b2):

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    if ((x2 < x1) or (y2 < y1)):
        return 0

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    # A overlap
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area

def get_Union_Area(boxA, boxB, interArea=None):
    area_A = calc_bounding_box_area(boxA[0], boxA[1], boxA[2], boxA[3])
    area_B = calc_bounding_box_area(boxB[0], boxB[1], boxB[2], boxB[3])
    if interArea is None:
        interArea = get_Intersection_Area(boxA, boxB)
    return float(area_A + area_B - interArea)

def find_best_thresholds(detections, truth, trial_ID, tools_list):

    truth["trial_id"] = [x[0:6] for x in truth["trial_frame"]]
    truth["frame"] = [int(x[-13:-5]) for x in truth["trial_frame"]]

    truth = truth.loc[(truth["trial_id"] == trial_ID)]
    truth.dropna(inplace=True)

    truth.drop(["trial_frame"], axis = 1, inplace = True)

    print(tools_list)

    #result = calculate_metrics(detections, truth, trial_ID, tools_list, 0.5)

    results, best_tool_thresholds = PlotPrecisionRecallCurve(detections, truth, trial_ID, tools_list, IOUThreshold=0.5, showGraphic=False)

    return best_tool_thresholds

def calc_avg_precision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1+i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

#need to make sure truth and detections only contain detections from the same trial
def calculate_metrics(net_detections, truth, trial_ID, tools_list, IOUThreshold=0.5):

    """Get the metrics used by the VOC Pascal 2012 challenge.
    Get
    Args:
        boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
        bounding boxes;
        IOUThreshold: IOU threshold indicating which detections will be considered TP or FP
        (default value = 0.5);
        method (default = EveryPointInterpolation): It can be calculated as the implementation
        in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
        interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
        or EveryPointInterpolation"  (ElevenPointInterpolation);
    Returns:
        A list of dictionaries. Each dictionary contains information and metrics of each class.
        The keys of each dictionary are:
        dict['class']: class representing the current dictionary;
        dict['precision']: array with the precision values;
        dict['recall']: array with the recall values;
        dict['AP']: average precision;
        dict['interpolated precision']: interpolated precision values;
        dict['interpolated recall']: interpolated recall values;
        dict['total positives']: total number of ground truth positives;
        dict['total TP']: total number of True Positive detections;
        dict['total FP']: total number of False Positive detections;
    """

    ret = []  # list containing metrics (precision, recall, average precision) of each class
    # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
    groundTruths = []
    # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
    detections = []
    # Get all classes
    classes = []

    for index, row in truth.iterrows():

        groundTruths.append([
                row["frame"],
                row["label"], 1,
                get_bounding_box_list_row(row)
            ])

    for index, row in net_detections.iterrows():
        detections.append([
                row["frame"],
                row["label"],
                row["score"],
                get_bounding_box_list_row(row)
            ])

    for c in tools_list:
        # Get only detection of class c
        dects = []
        [dects.append(d) for d in detections if d[1] == c]  #get only the detections for a specific tool
        # Get only ground truths of class c, use filename as key
        gts = {}
        npos = 0
        for g in groundTruths:
            if g[1] == c:
                npos += 1
                gts[g[0]] = gts.get(g[0], []) + [g]  #for each frame, creates gts dict with key=frame# and val=ground truths in that frame for the tool

        # sort detections by decreasing confidence
        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        thresholds = np.zeros(len(dects))

        # create dictionary with amount of gts for each image
        det = {key: np.zeros(len(gts[key])) for key in gts}

        print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
        # Loop through detections
        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][3],))

            # Find ground truth image/frame number
            gt = gts[dects[d][0]] if dects[d][0] in gts else []

            iouMax = 0
            for j in range(len(gt)):  #for each ground truth annotation in a specific frame
                # print('Ground truth gt => %s' % (gt[j][3],))

                iou = calc_iou(dects[d][3], gt[j][3]) #calculate IOU between each detection and each ground truth

                #basically find the detection bounding box with the greatest overlap with the ground truth annotations being compared
                if (iou > iouMax):
                    iouMax = iou
                    jmax = j

            thresholds[d] = dects[d][2]

            # Assign detection as true positive/don't care/false positive
            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1  # count as false positive
                # print("FP")
        # compute precision, recall and average precision

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos #tru pos / (tru pos + false neg)

        false_neg = npos - acc_TP
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        f1_score = 2 * (prec * rec) / (prec + rec)

        # Depending on the method, call the right implementation
        [ap, mpre, mrec, ii] = calc_avg_precision(rec, prec)

        # add class result in the dictionary to be returned
        r = {
            'class': c,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'thresholds': thresholds,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'false positives': acc_FP,
            'true positives': acc_TP,
            'false negatives': false_neg,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP),
            'f1 score': f1_score
        }

        ret.append(r)

    return ret

def PlotPrecisionRecallCurve(net_detections, truth, trial_ID, tools_list, IOUThreshold=0.5, showAP=False, showInterpolatedPrecision=False, savePath=None, showGraphic=True):

    """PlotPrecisionRecallCurve
    Plot the Precision x Recall curve for a given class.
    Args:
        boundingBoxes: Object of the class BoundingBoxes representing ground truth and detected
        bounding boxes;
        IOUThreshold (optional): IOU threshold indicating which detections will be considered
        TP or FP (default value = 0.5);
        method (default = EveryPointInterpolation): It can be calculated as the implementation
        in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
        interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
        or EveryPointInterpolation"  (ElevenPointInterpolation).
        showAP (optional): if True, the average precision value will be shown in the title of
        the graph (default = False);
        showInterpolatedPrecision (optional): if True, it will show in the plot the interpolated
         precision (default = False);
        savePath (optional): if informed, the plot will be saved as an image in this path
        (ex: /home/mywork/ap.png) (default = None);
        showGraphic (optional): if True, the plot will be shown (default = True)
    Returns:
        A list of dictionaries. Each dictionary contains information and metrics of each class.
        The keys of each dictionary are:
        dict['class']: class representing the current dictionary;
        dict['precision']: array with the precision values;
        dict['recall']: array with the recall values;
        dict['AP']: average precision;
        dict['interpolated precision']: interpolated precision values;
        dict['interpolated recall']: interpolated recall values;
        dict['total positives']: total number of ground truth positives;
        dict['total TP']: total number of True Positive detections;
        dict['total FP']: total number of False Negative detections;
    """

    results = calculate_metrics(net_detections, truth, trial_ID, tools_list, IOUThreshold)

    best_tool_thresholds = {}

    result = None
    # Each result represents a class
    for result in results:
        if result is None:
            raise IOError('Error: Class %d could not be found.' % classId)

        classId = result['class']
        precision = result['precision']
        recall = result['recall']
        thresholds = result['thresholds']
        average_precision = result['AP']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']
        f1_score = result['f1 score']

        try:
            max_f1_score = np.nanmax(f1_score)
            max_f1_index = list(f1_score).index(max_f1_score)
            best_threshold = thresholds[max_f1_index]    

            best_tool_thresholds[classId] = best_threshold

            print("max f1 score:", max_f1_score, best_threshold, classId)

            if(showGraphic == True or savePath is not None):
                plt.close()

                plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')

                plt.plot(recall, precision, label='Precision')

                plt.plot(recall[max_f1_index], precision[max_f1_index], 'ro', label= ('optimal thresh: '+ str(best_threshold)[0:6]) )

                plt.xlabel('recall')
                plt.ylabel('precision')

                if showAP:
                    ap_str = "{0:.2f}%".format(average_precision * 100)
                    # ap_str = "{0:.4f}%".format(average_precision * 100)
                    plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(classId), ap_str))
                else:
                    plt.title('Precision x Recall curve \nClass: %s' % str(classId))
                plt.legend(shadow=True)
                plt.grid()

                if savePath is not None:
                    plt.savefig(os.path.join(savePath, str(classId) + '.png'))
                if showGraphic is True:
                    plt.show()
                    # plt.waitforbuttonpress()
                    plt.pause(0.05)

        except:
            print("no score for tool", classId)

            #**Default threshold for a tool if a best threshold cannot be determined (not enough instances or not present)
            best_tool_thresholds[classId] = 0.5

    return results, best_tool_thresholds

def generate_APMs_from_detections_file(fileName):

    np.seterr(divide='ignore', invalid='ignore')

    data = pd.read_csv(fileName) #read in the input data file

    trial_ID = data["vid"][0]

    video_width, video_height = get_video_dimensions(trial_ID)

    #data = normalize_coords(data, video_width, video_height)  # normalize the coords relative to frame size

    #total_frames = int(max(data["frame"]))
    total_frames = get_num_frames(trial_ID)
    total_frames_w_tools = len(list(data["frame"].unique()))

    all_tools = list(data["label"].unique())
    all_tools.sort()

    #main_tools = ['suction', 'grasper', 'cottonoid', 'string', 'muscle']
    tools = ['suction', 'grasper', 'cottonoid', 'string', 'muscle']
    tools.sort()

    #-----------------------read ground truth to calculate confidence score threshold for tools
    truth = pd.read_csv("socal.csv", names=["trial_frame", "x1", "y1", "x2", "y2", "label"])

    #truth = normalize_coords(truth, video_width, video_height)

    best_tool_thresholds = find_best_thresholds(data, truth, trial_ID, all_tools)

    high_score_tools_data = get_high_score_tools(data, all_tools, best_tool_thresholds)

    high_score_data = normalize_coords(high_score_tools_data, video_width, video_height)

    #-----------------------------------------------------------------------

    APM_data = pd.DataFrame() #adding in the columns as they are computed

    APM_data = add_video_info(APM_data, data["vid"][0], video_width, video_height, total_frames)

    search_thresh = 0.5

    #***best_tool_thresholds = dict with tool and best threshold for that tool

    for tool in tools:
        APM_data["frames_w_"+str(tool)] = [count_frames_w_tool(data, best_tool_thresholds[tool], tool) / total_frames]

    for i in range(1, 6):
        APM_data["frames_w_" + str(i) + "_tools"] = [count_frames_w_x_tools(high_score_tools_data, i) / total_frames]

    APM_data["frames_w_at_least_1_tool"] = [ len(high_score_data["frame"].unique()) / total_frames ]

    APM_data["frames_w_0_tools"] = [ (total_frames - len(high_score_data["frame"].unique())) / total_frames ]

    for tool in tools:

        APM_data["first_frame_w_"+str(tool)] = [find_first_frame_w_tool(data, best_tool_thresholds[tool], tool)]

    total_in_n_outs = 0

    for tool in tools:

        APM_data[str(tool)+"_in_n_outs"] = [calc_in_n_outs(data, best_tool_thresholds[tool], tool)]
        total_in_n_outs += APM_data[str(tool)+"_in_n_outs"][0]

    APM_data["total_in_n_outs"] = total_in_n_outs

    for tool in tools:
        APM_data["area_covered_"+tool] = [calc_total_tool_area(data, best_tool_thresholds[tool], tool) / (total_frames * video_width * video_height)]

    data = normalize_coords(data, video_width, video_height)  # normalize the coords relative to frame size

    for tool in tools:
        for coord in ["x1", "y1", "x2", "y2"]:
            
            APM_data[tool+"_"+coord+"_"+"sd"] = calc_tool_coord_sd(data, best_tool_thresholds[tool], tool, coord)

    for tool in tools:
        APM_data[tool+"_x_center_sd"] = calc_tool_coord_difference_sd(data, best_tool_thresholds[tool], tool, "x1", "x2")
        APM_data[tool+"_y_center_sd"] = calc_tool_coord_difference_sd(data, best_tool_thresholds[tool], tool, "y1", "y2")

    for tool in tools:    
        
        tool_distance, tool_speed = calc_tool_distance_covered(data, best_tool_thresholds[tool], tool)
        APM_data["distance_covered_"+tool] = tool_distance
        APM_data["speed_"+tool] = tool_speed

    data = un_normalize_coords(data, video_width, video_height)  # un_normalize the coords relative to frame size

    sc_overlap, sc_frames = calc_total_tool_overlap(high_score_tools_data, search_thresh, "suction", "cottonoid", 0.15)
    gm_overlap, gm_frames = calc_total_tool_overlap(high_score_tools_data, search_thresh, "grasper", "muscle", 0.05)

    APM_data["sc_overlap_area"] = [sc_overlap]
    APM_data["gm_overlap_area"] = [gm_overlap]
    APM_data["sc_overlap_frames"] = [sc_frames/total_frames]
    APM_data["gm_overlap_frames"] = [gm_frames/total_frames]

    #print(data.head())
    print(APM_data.iloc[:,-4:])

    #APM_data.to_csv(fileName[:-4]+"_APM.csv", index=False)
    print("[Done] - " + fileName)

    return APM_data, APM_data.columns.values.tolist()

def main() :

    '''
    Questions:

        1. which tools to compute for
        2. length of video - i.e. number of frames in entire video, not just from detections
        3. dividing by total number of frames vs frames with tools vs highest frame number in file
        4. detection threshold / score that is accurate from detections file
            -> affects detection of cottonoids a lot
    '''

    fileName = "S102T2_retinanet.csv"

    file_APMs_df, APM_columns = generate_APMs_from_detections_file(fileName)

    file_APMs_df.to_csv(fileName[:-4]+"_APM.csv", index=False)


main()

