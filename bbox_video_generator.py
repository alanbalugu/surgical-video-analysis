import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_video_dimensions(trial_ID):
    input_df = pd.read_csv("socal_trial_outcomes.csv")
    height = input_df.loc[input_df["trial_id"] == trial_ID]["trial_video_height"].iloc[0]
    width = input_df.loc[input_df["trial_id"] == trial_ID]["trial_video_width"].iloc[0]

    return width, height

def get_bounding_box_list_row(tool_row):
    return [ tool_row["x1"], tool_row["y1"], tool_row["x2"], tool_row["y2"] ]


def get_bounding_box_list_df(tool_df):
    return [ tool_df["x1"].iloc[0], tool_df["y1"].iloc[0], tool_df["x2"].iloc[0], tool_df["y2"].iloc[0] ]

def calc_bounding_box_area(x1, y1, x2, y2):
    h = y2 - y1 + 1
    w = x2 - x1 + 1 

    return float(h*w)

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

    return best_tool_thresholds, truth

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
        [dects.append(d) for d in detections if d[1] == c]
        # Get only ground truths of class c, use filename as key
        gts = {}
        npos = 0
        for g in groundTruths:
            if g[1] == c:
                npos += 1
                gts[g[0]] = gts.get(g[0], []) + [g]

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
            # Find ground truth image
            gt = gts[dects[d][0]] if dects[d][0] in gts else []
            iouMax = 0
            for j in range(len(gt)):
                # print('Ground truth gt => %s' % (gt[j][3],))
                iou = calc_iou(dects[d][3], gt[j][3])

                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            # Assign detection as true positive/don't care/false positive
            thresholds[d] = dects[d][2]

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


def get_high_score_tools(data, tools, best_tool_thresholds):

    high_score_data = pd.DataFrame()

    for tool in tools:

        high_score_data = high_score_data.append( data.loc[ (data["label"] == tool) & (data["score"] >=  float(best_tool_thresholds[tool])) ],  ignore_index=True )

    high_score_data = high_score_data.sort_values(by=["frame"])

    rows_to_drop = []

    for frame in high_score_data["frame"].unique():

        right_frame_df = high_score_data.loc[high_score_data["frame"] == frame]

        tools_in_frame_indices = right_frame_df.index.tolist()
        tools_in_frame = list(right_frame_df["label"])

        #print(tools_in_frame, frame)

        if(len(tools_in_frame) != 1) :
            
            for tool1_index in range(0, len(tools_in_frame)):

                for tool2_index in range(tool1_index+1, len(tools_in_frame)):

                    #print(tools_in_frame[tool_index], tools_in_frame[tool2_index])

                    tool1_row = right_frame_df.loc[(right_frame_df["label"] == tools_in_frame[tool1_index]) & (right_frame_df.index == tools_in_frame_indices[tool1_index])]
                    tool2_row = right_frame_df.loc[(right_frame_df["label"] == tools_in_frame[tool2_index]) & (right_frame_df.index == tools_in_frame_indices[tool2_index])]

                    if(tools_in_frame[tool1_index] != tools_in_frame[tool2_index]):

                        iou = calc_iou(get_bounding_box_list_df(tool1_row), get_bounding_box_list_df(tool2_row))

                        if(iou > 0.9):

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
#----------------------------------------------------------------


def main():

    data = pd.read_csv("S102T2_retinanet.csv") #read in the input data file

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

    best_tool_thresholds, truths_only_for_trial = find_best_thresholds(data, truth, trial_ID, all_tools)

    high_score_tools_data = get_high_score_tools(data, all_tools, best_tool_thresholds)

    #-----------------------------------------------------------------------------------------

    image_folder = "C:\\Users\\Ganesh Balu\\Documents\\SOCAL_frames\\frames\\JPEGImages\\"
    video_name = 'bounding_box_video_S102T2.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
    images = [img for img in images if img[0:6] == "S102T2"]

    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 3, (width,height))

    for image in images:
        right_frame_df = high_score_tools_data[high_score_tools_data["frame"] == int(image[-13:-5])]

        frame_image = cv2.imread(os.path.join(image_folder, image))
        copy_frame_image = frame_image.copy()

        for index, row in right_frame_df.iterrows():

            top_left = (int(row["x1"]), int(row["y1"]))
            bottom_right = (int(row["x2"]), int(row["y2"]))

            cv2.rectangle(copy_frame_image, top_left, bottom_right, (255,0,0), thickness= 3, lineType=cv2.LINE_8)
            cv2.putText(copy_frame_image, row["label"], (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        right_frame_df_truth = truths_only_for_trial[truths_only_for_trial["frame"] == int(image[-13:-5])]

        for index, row in right_frame_df_truth.iterrows():

            top_left = (int(row["x1"]), int(row["y1"]))
            bottom_right = (int(row["x2"]), int(row["y2"]))

            cv2.rectangle(copy_frame_image, top_left, bottom_right, (0,255,0), thickness= 3, lineType=cv2.LINE_8)
            cv2.putText(copy_frame_image, row["label"], (bottom_right[0]-120, top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.putText(copy_frame_image, image[-13:-5], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(copy_frame_image, "blue=model, green=truth", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        video.write(copy_frame_image)

    video.release()

main()