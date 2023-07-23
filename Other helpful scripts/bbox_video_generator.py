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

# calculates the best threshold for a tool using SOCAL ground truth detections
def find_best_thresholds(detections, truth, trial_IDs, tools_list, showGraphs=False):
    truth["trial_id"] = [x[0:6] for x in truth["trial_frame"]]  # just the trial id
    truth["frame"] = [int(x[-13:-5]) for x in truth["trial_frame"]]  # just the frame number

    truth = truth[truth.trial_id.isin(trial_IDs)]
    truth.dropna(inplace=True)
    print(tools_list)

    results, best_tool_thresholds, tool_precisions = PlotPrecisionRecallCurve(detections, truth, trial_IDs, tools_list,
                                                                              IOUThreshold=0.50, showGraphic=showGraphs)

    return best_tool_thresholds, tool_precisions


# calculates average precision
def calc_avg_precision(rec, prec):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1 + i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + ((mrec[i] - mrec[i - 1]) * mpre[i])

    # ap = sum([mpre[i] for i in ii])/len(ii)  #????

    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

# Calculates metrics given the detections and ground truth data for a trial
def calculate_metrics(net_detections, truth, trial_ID, tools_list, IOUThreshold=0.50):
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

    # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates X,Y,X2,Y2)])
    groundTruths = []

    # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
    detections = []

    # Get all classes
    classes = []

    for index, row in truth.iterrows():
        groundTruths.append([
            row["trial_frame"].replace(".jpeg", ".jpg"),
            row["label"], 1.0,
            get_bounding_box_list_row(row)
        ])

    for index, row in net_detections.iterrows():
        detections.append([
            row["trial_frame"],
            row["label"],
            row["score"],
            get_bounding_box_list_row(row),
        ])

    detections = sorted(detections, key=lambda conf: conf[2], reverse=True)

    for c in tools_list:
        # Get only detection of class c
        dects = []
        [dects.append(d) for d in detections if (d[1] == c)]  # get only the detections for a specific tool

        # Get only ground truths of class c, use filename as key
        gts = {}
        npos = 0
        for g in groundTruths:
            if g[1] == c:
                npos += 1
                gts[g[0]] = gts.get(g[0], []) + [
                    g]  # for each frame, creates gts dict with key=frame# and val=ground truths in that frame for the tool

        # sort detections by decreasing confidence
        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)

        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))

        thresholds = np.zeros(len(dects))

        # create dictionary with amount of gts for each image
        det = {key: np.zeros(len(gts[key])) for key in gts}

        #print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
        # Loop through detections

        vals = []
        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][3],))

            # Find ground truth image/frame number
            gt = gts[dects[d][0]] if dects[d][0] in gts else []

            iouMax = 0
            jmax = 0
            for j in range(len(gt)):  # for each ground truth annotation in a specific frame
                # print('Ground truth gt => %s' % (gt[j][3],))

                # print(dects[d], gt[j])
                iou = calc_iou(dects[d][3], gt[j][3])  # calculate IOU between each detection and each ground truth

                # Find the detection bbox with the greatest overlap with the ground truth annotations being compared
                if (iou > iouMax):
                    iouMax = iou
                    jmax = j

            # print(dects[d][0], dects[d][1], iouMax, jmax)

            thresholds[d] = dects[d][2]

            # Assign detection as true positive/don't care/false positive
            if (iouMax > IOUThreshold):

                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
                    # print("TP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1

        # compute precision, recall and average precision

        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)

        try:
            rec = np.divide(acc_TP, npos)  # tru pos / (tru pos + false neg)
            rec = np.append(rec, rec[len(rec) - 1])

            prec = np.divide(acc_TP, np.add(acc_FP, acc_TP))
            prec = np.append(prec, 0.0)
        except:

            rec = np.divide(acc_TP, npos)  # tru pos / (tru pos + false neg)
            prec = np.divide(acc_TP, np.add(acc_FP, acc_TP))

        # rec = np.append(rec, 1.0)

        false_neg = (npos - acc_TP)

        f1_score = 2 * np.divide(np.multiply(prec, rec), np.add(prec, rec))

        # Depending on the method, call the right implementation

        [ap, mpre, mrec, ii] = calc_avg_precision(rec, prec)
        # [ap, mpre, mrec, ii] = ElevenPointInterpolatedAP(rec, prec)

        # add class result in the dictionary to be returned. There are the calculates metrics for that tool
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

def get_trial_test_set():
    return [
        'S201T1', 'S201T2','S202T1', 'S202T2','S203T1', 'S203T2','S204T1', 'S204T2','S205T1', 'S205T2', 'S206T1', 'S206T2','S207T1', 'S207T2',
        'S502T1', 'S502T2','S502T2', 'S504T1', 'S504T2', 'S505T1', 'S505T2', 'S506T1', 'S507T1', 'S507T2'
    ]  #FOR SOCAL

# plots the Prec x Recall curve and returns the best confidence threshold for all tools
def PlotPrecisionRecallCurve(net_detections, truth, trial_IDs, tools_list, IOUThreshold=0.5, showAP=True,
                             showInterpolatedPrecision=False, savePath=None, showGraphic=True):
    # showGraphic = False
    # net_detections2 = net_detections.loc[net_detections["label"].isin(tools_list)]
    results = calculate_metrics(net_detections, truth, trial_IDs, tools_list, IOUThreshold)

    best_tool_thresholds = {}
    tool_precisions = {}

    # Each result represents a class
    for result in results:
        if result is None:
            raise IOError('Error: Class %d could not be found.')

        classId = result['class']
        precision = result['precision']  # average precision
        recall = result['recall']  # average recall
        thresholds = result['thresholds']
        average_precision = result['AP']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']

        npos = result['total positives']  # total real ground truth pos for that tool
        true_positives = result["true positives"]  # cumulative TPs for each threshold
        false_positives = result["false positives"]  # cumulative FPs for each threshold

        total_tp = result['total TP']
        total_fp = result['total FP']
        f1_score = result['f1 score']

        try:
            max_f1_score = np.nanmax(f1_score)
            max_f1_index = list(f1_score).index(max_f1_score)
            best_threshold = thresholds[max_f1_index]

            best_tool_thresholds[classId] = best_threshold

            # best_precision = true_positives[max_f1_index] / (true_positives[max_f1_index] + false_positives[max_f1_index])

            tool_precisions[classId] = average_precision

            # print(classId, "max f1 score:", max_f1_score, "best threshold: ", best_threshold, " best precision: ", best_precision)
            print(classId, "Average precision: ", average_precision)
            print(classId, total_tp, total_fp)

            if (showGraphic is True or savePath is not None):
                plt.close()

                #plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')

                plt.plot(recall, precision, label='Precision')
                fig = plt.gcf()
                fig.set_size_inches(10, 7)

                plt.xlim(0, 1.0)
                plt.xticks(np.arange(0, 1.1, 0.1))
                plt.ylim(0, 1.0)
                plt.yticks(np.arange(0, 1.1, 0.1))

                #plt.plot(recall[max_f1_index], precision[max_f1_index], 'ro', label=('optimal thresh: ' + str(best_threshold)[0:6]))

                print(classId, "best thresh: ", str(best_threshold)[0:6])

                ax = plt.gca()
                ax.set_xlabel('Recall', fontsize=16)
                ax.set_ylabel('Precision', fontsize=16)
                ax.tick_params(labelsize=14.0, length=5.0, width=2.0)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2.0)  # change width

                if showAP:
                    # bp_str = "{0:.2f}%".format(best_precision * 100)
                    bp_str = "{0:.2f}%".format(average_precision * 100)

                    plt.title('Class: %s, AP: %s' % (str(classId), bp_str), fontsize=20)
                else:
                    plt.title('Precision x Recall curve \nClass: %s' % str(classId))

                #plt.legend(shadow=False)

                if savePath is not None:
                    plt.savefig(os.path.join(savePath, str(classId) + '.png'))
                if showGraphic is True:
                    plt.rcParams['savefig.dpi'] = 400
                    plt.savefig("/Users/alanbalu/Library/CloudStorage/OneDrive-Personal/Alan/GUSOM 2025/Donoho Research/SOCAL Shannon Entropy Paper/Figures/"+ str(classId) + " pr curve.png")
                    #plt.show()
                    # plt.waitforbuttonpress()
                    plt.pause(0.05)

        except Exception as e:
            print("no score for tool: ", classId)
            print(e)

            # **Default threshold for a tool if a best threshold cannot be determined (not enough instances or not present)
            best_tool_thresholds[classId] = 0.5
            tool_precisions[classId] = np.nan

    return results, best_tool_thresholds, tool_precisions

# returns the data with only high confidence detections and removes duplicate bboxes (IOU > 0.9)
def get_high_score_tools(data, tools, best_tool_thresholds):
    high_score_data = pd.DataFrame()

    for tool in tools:
        high_score_data = pd.concat(
            [high_score_data, data.loc[(data["label"] == tool) & (data["score"] >= float(best_tool_thresholds[tool]))]],
            ignore_index=True)

    high_score_data = high_score_data.sort_values(by=["trial_frame"])

    rows_to_drop = []

    for frame in high_score_data["trial_frame"].unique():

        right_frame_df = high_score_data.loc[high_score_data["trial_frame"] == frame]

        tools_in_frame_indices = right_frame_df.index.tolist()
        tools_in_frame = list(right_frame_df["label"])

        # print(tools_in_frame, frame)

        if (len(tools_in_frame) != 1):

            for tool1_index in range(0, len(tools_in_frame)):

                for tool2_index in range(tool1_index + 1, len(tools_in_frame)):

                    # print(tools_in_frame[tool_index], tools_in_frame[tool2_index])

                    tool1_row = right_frame_df.loc[(right_frame_df["label"] == tools_in_frame[tool1_index]) & (
                                right_frame_df.index == tools_in_frame_indices[tool1_index])]

                    tool2_row = right_frame_df.loc[(right_frame_df["label"] == tools_in_frame[tool2_index]) & (
                                right_frame_df.index == tools_in_frame_indices[tool2_index])]

                    #if (tools_in_frame[tool1_index] != tools_in_frame[tool2_index]):

                    iou = calc_iou(get_bounding_box_list_df(tool1_row), get_bounding_box_list_df(tool2_row))
                    # print(tools_in_frame[tool1_index], tools_in_frame[tool2_index])
                    # print(get_bounding_box_list_df(tool1_row), get_bounding_box_list_df(tool2_row), iou)

                    if (iou > 0.5):

                        if (float(tool1_row["score"].iloc[0]) > float(tool2_row["score"].iloc[0])):

                            rows_to_drop.append(tool2_row.index[0])
                        else:

                            rows_to_drop.append(tool1_row.index[0])


    high_score_data.drop(rows_to_drop, axis=0, inplace=True)

    return high_score_data

#----------------------------------------------------------------

def main():

    data = pd.read_csv('yolov4_socal_detections_4.7_fixed.csv', header=0)

    data["trial"] = [frame[0:6] for frame in data["trial_frame"]]

    # all_tools = list(data["label"].unique())
    # all_tools.sort()
    #
    trial_IDs = list(data["trial_frame"].unique())
    trial_IDs = [img[0:6] for img in trial_IDs]
    trial_IDs = list(set(trial_IDs))
    trial_IDs.sort()
    #
    #main_tools = ['suction', 'grasper', 'cottonoid', 'string', 'muscle']
    tools = ['suction', 'grasper', 'cottonoid', 'string', 'muscle']
    tools.sort()
    #
    # #-----------------------read ground truth to calculate confidence score threshold for tools
    truth = pd.read_csv("socal.csv", names=["trial_frame", "x1", "y1", "x2", "y2", "label"])

    truth["trial"] = [frame[0:6] for frame in truth["trial_frame"]]
    truth = truth.loc[truth["trial"].isin(trial_IDs)]

    best_tool_thresholds, tool_precisions = find_best_thresholds(data, truth, trial_IDs, tools)
    # best_tool_thresholds = [0.25 for tool in tools]

    trial_id = "S505T1" #frames 50-65

    data = data.loc[data["trial"] == trial_id]
    truth = truth.loc[truth["trial"] == trial_id]
    #truth.dropna(inplace=True)

    frames = [int(img[-12:-4]) for img in data["trial_frame"]] #extracts frame number
    data["frame"] = frames

    high_score_tools_data = get_high_score_tools(data, tools, best_tool_thresholds)
    #high_score_tools_data = data #.loc[data["score"] > 0.25]

    #-----------------------------------------------------------------------------------------

    image_folder = "C:/Users/Alan/Documents/SOCAL_frames/frames/JPEGImages/" 
    video_name = "C:/Users/Alan/Documents/" + str(trial_id) +"_yolov4.avi" #+"_yolov3.avi"

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = [img for img in images if img[0:6] == trial_id]

    images.sort()

    print(trial_id)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 2, (width,height))

    # sort to only include a small subset of frames ------------------
    data = data.loc[(data["frame"] >= 50) & (data["frame"] <= 65)]
    truth = truth.loc[(truth["frame"] >= 50) & (truth["frame"] <=65)]
    images = list(set(list(truth["trial_frame"])))
    images = [img[0:-4]+"jpg" for img in images]
    images.sort()

    print(images)

    for image in images:

        frame_image = cv2.imread(os.path.join(image_folder, image))
        copy_frame_image = frame_image.copy()

        right_frame_df_truth = truth.loc[truth["trial_frame"] == image.replace(".jpg", ".jpeg")]

        for index, row in right_frame_df_truth.iterrows():

            top_left = (int(row["x1"]), int(row["y1"]))
            bottom_right = (int(row["x2"]), int(row["y2"]))

            cv2.rectangle(copy_frame_image, top_left, bottom_right, (0,255,0), thickness= 3, lineType=cv2.LINE_8)
            cv2.putText(copy_frame_image, row["label"], (bottom_right[0]-120, top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        right_frame_df = high_score_tools_data[high_score_tools_data["frame"] == int(image[-12:-4])]

        for index, row in right_frame_df.iterrows():
            top_left = (int(row["x1"]), int(row["y1"]))
            bottom_right = (int(row["x2"]), int(row["y2"]))

            cv2.rectangle(copy_frame_image, top_left, bottom_right, (255, 0, 0), thickness=3, lineType=cv2.LINE_8)
            cv2.putText(copy_frame_image, row["label"], (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 0, 0), 2)

        cv2.putText(copy_frame_image, image, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(copy_frame_image, "blue=model, green=truth", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        video.write(copy_frame_image)

    video.release()

main()