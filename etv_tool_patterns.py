import os
import numpy as np
import pandas as pd
import itertools
from operator import itemgetter
import math
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from scipy import stats
from scipy.interpolate import make_interp_spline
from scipy import mean
# from sklearn.cluster import AgglomerativeClustering
# import scipy.cluster.hierarchy as shc
# from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind
from matplotlib.path import Path

verts = [
   (0., -8),  # left, bottom
   (0., 8),  # left, top
   (0.00001, 8),  # right, top
   (0.00001, -8),  # right, bottom
   (0., 0.),  # back to left, bottom
]

codes = [
    Path.MOVETO, #begin drawing
    Path.LINETO, #straight line
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY, #close shape. This is not required for this shape but is "good form"
]

path = Path(verts, codes)

# normalize the coordinates in the dataframe to [0,1] based on frame dimensions
def normalize_coords(data, frame_width, frame_height):
    data["x1"] = data["x1"] / frame_width
    data["x2"] = data["x2"] / frame_width
    data["y1"] = data["y1"] / frame_height
    data["y2"] = data["y2"] / frame_height

    return data

# reverse the normalization of the dataframe coordinates
def un_normalize_coords(data, frame_width, frame_height):
    data["x1"] = data["x1"] * frame_width
    data["x2"] = data["x2"] * frame_width
    data["y1"] = data["y1"] * frame_height
    data["y2"] = data["y2"] * frame_height

    return data


# Add empty columns to dataframe for frame info
def add_video_info(data, file_id, width, height, frames):
    data["trial_ID"] = []
    data["width"] = []
    data["height"] = []
    data["total_frames"] = []

    new_data = {"trial_ID": file_id, "width": width, "height": height, "total_frames": frames}
    data = data.append(new_data, ignore_index=True)

    return data

# calculatess the area of the bounding box
def calc_bounding_box_area(x1, y1, x2, y2):
    h = y2 - y1 + 1
    w = x2 - x1 + 1

    return float(h * w)

# creates a list with the bounding box coordinates using a dataframe object
def get_bounding_box_list_df(tool_df):
    return [tool_df["x1"].iloc[0], tool_df["y1"].iloc[0], tool_df["x2"].iloc[0], tool_df["y2"].iloc[0]]


# creates a list with the bounding box coordinates using a row object
def get_bounding_box_list_row(tool_row):
    return [tool_row["x1"], tool_row["y1"], tool_row["x2"], tool_row["y2"]]

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


# uses the socal_trial_outcomes.csv file to get the dimensions based on the trial
# default size is 1920 x 1280
def get_video_dimensions(trial_ID):
    input_df = pd.read_csv("socal_trial_outcomes.csv")

    height = input_df.loc[input_df["trial_id"] == trial_ID]["trial_video_height"].iloc[0]
    width = input_df.loc[input_df["trial_id"] == trial_ID]["trial_video_width"].iloc[0]

    if (math.isnan(height)):
        height = 1280
        width = 1920

    return width, height


# calculates the number of frames for a given trial using frame_to_trial_mapping.csv
def get_num_frames(trial_IDs, truth_df):

    frame_numbers = {}

    for ID in trial_IDs:
        try:
            frame_numbers[ID] = max(truth_df.loc[input_df["trial_id"] == ID]["frame"].tolist())
        except:
            frame_numbers[ID] = 0

    return frame_numbers


# calculates the IOU value for 2 bounding boxes
def calc_iou(boxA, boxB):
    # if boxes dont intersect
    if do_boxes_intersect(boxA, boxB) is False:
        return 0
    interArea = get_Intersection_Area(boxA, boxB)
    union = get_Union_Area(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    return iou


# Checks if bounding boxes intersect
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


# calculates the intersection area between 2 bounding boxes
def get_Intersection_Area(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    if (do_boxes_intersect(b1, b2) == False):
        return 0.0

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    # A overlap
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area


# calculates the union area for 2 bounding boxes
def get_Union_Area(boxA, boxB, interArea=None):
    area_A = calc_bounding_box_area(boxA[0], boxA[1], boxA[2], boxA[3])
    area_B = calc_bounding_box_area(boxB[0], boxB[1], boxB[2], boxB[3])
    if interArea is None:
        interArea = get_Intersection_Area(boxA, boxB)
    return float(area_A + area_B - interArea)


# calculates the best threshold for a tool using SOCAL ground truth detections
def find_best_thresholds(detections, truth, trial_IDs, tools_list, showGraphs=False):
    #truth["trial_id"] = [x[0:5] for x in truth["trial_frame"]]  # just the trial id
    #truth["frame"] = [int(x[-10:-4]) for x in truth["trial_frame"]]  # just the frame number

    truth = truth[truth.trial_id.isin(trial_IDs)]
    truth.dropna(inplace=True)
    print(tools_list)

    results, best_tool_thresholds, tool_precisions = PlotPrecisionRecallCurve(detections, truth, trial_IDs, tools_list,
                                                                              IOUThreshold=0.5, showGraphic=showGraphs)

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
            row["trial_frame"],
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
                gts[g[0]] = gts.get(g[0], []) + [g]  # for each frame, creates gts dict with key=frame# and val=ground truths in that frame for the tool

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
                ax.set_xlabel('Recall', fontsize=20)
                ax.set_ylabel('Precision', fontsize=20)
                ax.tick_params(labelsize=18.0, length=5.0, width=2.0)
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2.0)  # change width

                if showAP:
                    # bp_str = "{0:.2f}%".format(best_precision * 100)
                    bp_str = "{0:.2f}%".format(average_precision * 100)

                    plt.title('Class: %s, AP: %s' % (str(classId), bp_str), fontsize=26)
                else:
                    plt.title('Precision x Recall curve \nClass: %s' % str(classId))

                #plt.legend(shadow=False)

                if savePath is not None:
                    plt.savefig(os.path.join(savePath, str(classId) + '.png'))
                if showGraphic is True:
                    plt.rcParams['savefig.dpi'] = 400
                    #plt.savefig("/Users/alanbalu/Library/CloudStorage/OneDrive-Personal/Alan/GUSOM 2025/Donoho Research/SOCAL Shannon Entropy Paper/Figures/"+ str(classId) + " pr curve.png")
                    plt.show()
                    # plt.waitforbuttonpress()
                    #plt.pause(0.05)

        except Exception as e:
            print("no score for tool: ", classId)
            print(e)

            # **Default threshold for a tool if a best threshold cannot be determined (not enough instances or not present)
            best_tool_thresholds[classId] = 0.5
            tool_precisions[classId] = np.nan

    return results, best_tool_thresholds, tool_precisions

def smooth_labels(data, tool, threshold=10):

    data = data.loc[data["label"] == tool]

    trials = list(data["trial_id"].unique())

    dict = {}

    for trial in trials:

        trial_tool_filtered = data.loc[data["trial_id"] == trial]

        unique_instances = list(trial_tool_filtered["frame"].unique())

        groups = get_ranges(unique_instances)

        run = True
        index = 0

        if(len(groups) > 1):
            while(run):

                end = groups[index][len(groups[index])-1]
                beg = groups[index + 1][0]

                if((beg-end) < threshold):  #if the gap between ranges is smaller than threshold, fill that gap in

                    groups[index] = list( groups[index] + [i for i in range(end + 1, beg)] + groups[index + 1] )
                    groups.remove(groups[index + 1])

                else:
                    index += 1

                if(len(groups) == (index+1)): run = False

        dict[trial] = groups

    return dict

def fill_tool_gaps(smooth_trials, data, tool, min_frames=2):

    cut_data = data[["trial_frame", "frame", "label", "trial_id"]]

    new_tool_data = pd.DataFrame(columns=["trial_frame", "frame", "label", "trial_id"])
    removed_tool_data = pd.DataFrame(columns=["trial_frame", "frame", "label", "trial_id"])

    for trial in smooth_trials.keys():

        new_data = cut_data.loc[cut_data["trial_id"] == trial]

        missing_frames = []
        removal_frames = []

        for range in smooth_trials[trial]:  #for each range of frames where tools are present

            for i in range:  #for each frame in that range

                new_new_data = new_data.loc[new_data["frame"] == i]  #get the tools in that frame

                if (len(range) > min_frames): #tool must be present for min_frames to be legit

                    if(tool not in list(new_new_data["label"])):
                        missing_frames.append(i)

                else:
                    removal_frames.append(i)  #remove frames if the tool is only present for < min_frames

        for frame in removal_frames:

            test = new_data.loc[ (new_data["frame"] == frame) & (new_data["label"] == tool) ]
            removed_tool_data = pd.concat([removed_tool_data, test], ignore_index=True).sort_index()

        extension = ".jpg"

        for frame in missing_frames:

            try:
                test = new_data.loc[new_data["frame"] == frame].iloc[0].copy()
                test["label"] = tool

                new_tool_data = pd.concat([new_tool_data,pd.DataFrame([test],columns=new_tool_data.columns)], ignore_index=True).sort_index()

                if(".jpeg" in test["trial_frame"]): extension = ".jpeg"

            except:

                num_zeros = 8 - sum(c.isdigit() for c in str(frame))
                num = "0"*num_zeros + str(frame)
                name = trial+"_frame_"+ num + extension
                test = [name, frame, tool, trial]
                new_tool_data = pd.concat([new_tool_data, pd.DataFrame([test],columns=new_tool_data.columns)], ignore_index=True).sort_index()

    return new_tool_data, removed_tool_data

def calculate_tool_patterns(high_score_data, tools, trial_IDs, trial_frames_dict, bin_count=10):

    scaled_data = high_score_data.copy()
    scaled_data = scaled_data.sort_values(by=["trial_frame"])

    scaled_frame = []

    for index, row in scaled_data.iterrows():
        scaled_frame.append(float(row["frame"]) / float(trial_frames_dict[row["trial_id"]]))

    scaled_data["scaled_frame"] = scaled_frame

    bins = list(np.linspace(0, 1, bin_count+1))

    trial_tool_totals_dict = {}
    trial_totals = {}

    trial_probs = {}
    tool_probs = {}
    for tool in tools:
        tool_probs[tool] = []
        trial_probs[tool] = {}

    for trial in trial_IDs:

        trial_tool_totals_dict[trial] = {}

        for index, tool in enumerate(tools):
            trial_tool_totals_dict[trial][tool] = [0 for i in range(0, len(bins) - 1)]
            trial_totals[tool] = 0

        new_scaled_data = scaled_data.loc[scaled_data["trial_id"] == trial]

        new_scaled_data["groups"] = pd.cut(new_scaled_data.scaled_frame, bins)

        groups_list = list(new_scaled_data["groups"].unique())
        groups_list.sort(reverse=False)

        for group_index, group in enumerate(groups_list):

            filtered = new_scaled_data.loc[new_scaled_data["groups"] == group]

            tool_counts = filtered["label"].value_counts()

            for index, value in enumerate(tool_counts):

                #print(index, tool_counts.index[index], value, group)

                if(tool_counts.index[index] in tools):

                    trial_tool_totals_dict[trial][tool_counts.index[index]][group_index] = value
                    trial_totals[tool_counts.index[index]] += value

        for tool in trial_totals.keys():
            # trial_length = len(new_scaled_data["trial_frame"].unique())
            trial_probs[tool][trial] = [ np.divide(i, trial_totals[tool]) for i in trial_tool_totals_dict[trial][tool]]
            # trial_probs[tool][trial] = [ np.divide(i, trial_length) for i in trial_tool_totals_dict[trial][tool]]

    values = {}

    for tool in tools:
        values[tool] = []
        for trial in trial_IDs:
            values[tool].append(trial_probs[tool][trial])

    tool_props_means = {}
    tool_props_std = {}

    for tool in tools:
        data = np.array(values[tool])
        tool_props_means[tool] = list(np.nanmean(data, axis=0))
        tool_props_std[tool] = list(np.nanstd(data, axis=0))

    bin_centers2 = [round(i + (1.0 / (2.0 * bin_count)), 2) for i in bins]
    bin_centers2 = bin_centers2[:-1]

    plt.title("tool proportions in bins averaged across trials")
    plt.xticks(np.arange(0.0, 1.1, (1.0 / bin_count)))
    plt.ylim(0, 0.3)

    for tool in tools:
        plt.plot(bin_centers2, tool_props_means[tool], label=tool)
        #plt.errorbar(bin_centers2, tool_props_means[tool], yerr=tool_props_std[tool])

    plt.legend(shadow=False, frameon=False, loc=(1.04, 0), fontsize=16)
    #plt.tight_layout(rect=[0, 0, 0.9, 0.9])

    ax = plt.gca()
    ax.set_xlabel('Trial Progress', fontsize=20)
    ax.set_ylabel('Percent of Tool Use', fontsize=20)
    ax.tick_params(labelsize=18.0, length=5.0, width=2.0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width
    fig = plt.gcf()
    fig.set_size_inches(10,7, forward=True)
    plt.tight_layout()
    plt.show()
    # plt.rcParams['savefig.dpi'] = 400
    # plt.savefig("/Users/alanbalu/Downloads/Figure.png")
    plt.clf()

def find_tool_patterns(data, tools, trial_IDs, trial_frames_dict, showGraphs=False):

    data = data.loc[data["label"].isin(tools)]
    data = data.sort_values(by=["trial_id"])

    outcomes = pd.read_csv("socal_trial_outcomes.csv")
    outcomes = outcomes.loc[outcomes["trial_id"].isin(trial_IDs)].sort_values(by=["trial_id"])

    trial_IDs = list(outcomes["trial_id"].unique())

    blood_loss = list(outcomes["blood_loss"])
    success = list(outcomes["success"])

    tth_scaled = []

    for index, row in outcomes.iterrows():
        tth_scaled.append(row["tth"]) #/trial_frames_dict[row["trial_id"]])

    data = data.loc[data["trial_id"].isin(list(outcomes["trial_id"].unique()))].sort_values(by=["trial_id"])

    combs = []
    for i in range(1, len(tools) + 1):
        combs.append(list(itertools.combinations(tools, i)))

    combs.sort()

    combs_list = []
    for i in combs:
        for j in i:
            combs_list.append(list(j))

    combs_list.insert(0, ['empty'])

    new_dict = {}

    # 0 for no tools, 1-31 for combinations of tools
    for i in range(0, len(combs_list)):
        new_dict[i] = []

    entropy_list = []
    cumu_divers = {}
    tool_combs_dict = {}

    for trial in trial_IDs:

        trial_data = data.loc[data["trial_id"] == trial]

        frame_entropy = []

        num_frames = trial_frames_dict[trial]

        for frame in range(1, num_frames + 1):  # frame in unique_frames:

            unique_labels = list(trial_data.loc[trial_data["frame"] == frame]["label"].unique())

            if (len(unique_labels) == 0):
                # print(trial, " empty ", frame)
                frame_entropy.append(0)

            val_combs = list(itertools.permutations(unique_labels, len(unique_labels)))

            # frame_lens.append(len(unique_labels))

            for index, i in enumerate(val_combs):
                if (list(i) in combs_list):
                    # print(combs_list.index(list(i))
                    frame_entropy.append(combs_list.index(list(i)))
                    break

        # trial_series = pd.Series(frame_entropy).value_counts().sort_index()
        # plt.bar(range(len(trial_series)), trial_series.values, align='center')
        # plt.xticks(range(len(trial_series)), trial_series.index.values, size='small')
        # plt.title(trial+" tool combs histogram")
        # plt.show()
        # plt.clf()

        # print(trial_frames_dict[trial], len(frame_entropy), frame_entropy)

        tool_combs_dict[trial] = frame_entropy
        entropy_series = pd.Series(frame_entropy)
        counts = entropy_series.value_counts()  # this can be used to find patterns
        counts_index_list = list(counts.index.values)

        for i in range(0, len(combs_list)):

            if (i in counts_index_list):

                # normalize this to the length of trial due to conflicting correlations
                new_dict[i].append(counts.iloc[counts_index_list.index(i)] / trial_frames_dict[trial])
            else:
                new_dict[i].append(0)

        probs = [i / len(entropy_series) for i in counts] # or entropy series or counts?

        entropy = stats.entropy(probs)
        entropy = entropy / math.log(len(entropy_series)) #normalize here by the log of the length of values
        #entropy = entropy / len(entropy_series)
        entropy_list.append(entropy)

        cum_trial_diversity = []
        for i in range(1, len(entropy_series)):
            entropy_segment = entropy_series[0:i]
            segment_probs = [i / len(entropy_segment) for i in entropy_segment.value_counts()]
            segment_entropy = stats.entropy(segment_probs)
            segment_entropy = segment_entropy / math.log(len(entropy_segment))
            #segment_entropy = segment_entropy / len(entropy_segment)

            #uniq_combs = len(entropy_segment.unique()) # Cumulative cumulative
            #cum_trial_diversity.append(uniq_combs)
            if(np.isnan(segment_entropy)): segment_entropy = 0.0
            cum_trial_diversity.append(segment_entropy)

        cumu_divers[trial] = cum_trial_diversity

    if(showGraphs == True):
        for trial in trial_IDs:
            plt.plot(range(0, len(cumu_divers[trial])), cumu_divers[trial])
            plt.title(trial)
            plt.ylim(0,30)
            plt.show()
            plt.clf()

    combs_count_df = pd.DataFrame()

    for comb in new_dict.keys():
        combs_count_df[comb] = new_dict[comb]

    combs_count_df["entropy"] = entropy_list

    return combs_count_df, combs_list, trial_IDs, blood_loss, tth_scaled, success, cumu_divers, tool_combs_dict


def get_ranges(unique_instances):

    groups = []

    for key, group in itertools.groupby(enumerate(unique_instances), lambda i: i[0] - i[1]):
        group = list(map(itemgetter(1), group))
        group = list(map(int, group))

        groups.append(group)

    return groups

def label_distributions(data, tools, trials, showFig=False, saveFig=False):

    group_dict = {}

    for trial in trials:

        trial_data = data.loc[data["trial_id"] == trial]
        trial_data.dropna(inplace=True)

        group_dict[trial] = {}

        for tool in tools:

            trial_tool_frames = list(trial_data.loc[trial_data["label"] == tool]["frame"])

            unique_instances = set(trial_tool_frames)
            unique_instances = list(unique_instances)
            unique_instances.sort()

            groups = get_ranges(unique_instances)
            group_dict[trial][tool] = groups

            #print(tool, len(groups)) #prints the number of ranges. correlations??

            if (len(trial_tool_frames) > 0):
                trial_tool_pres = [tool for i in range(0, len(trial_tool_frames))]
            else:
                trial_tool_pres = [tool]
                trial_tool_frames = [-10]

            if(showFig or saveFig):
                plt.scatter(trial_tool_frames, trial_tool_pres, label=tool, s=1)
                plt.title(trial + " - Tool Presence Distributions")
                # plt.xlim([-20, max(trial_tool_frames)+100])
                ax = plt.gca()
                ax.set_xlim(left=-10)
                plt.ylabel("tools")
                plt.xlabel("frames in trial")
                plt.tight_layout()

        if(saveFig):
            plt.savefig(trial + "_label_distribution.jpg")
        # 'S104T2', 'S304T2', 'S201T1' -> high
        # 'S305T1', 'S301T1', 'S301T2' -> low
        if(showFig and trial in ['S305T1', 'S301T1', 'S301T2']):
            plt.show()

        plt.clf()

    return group_dict

def gen_tool_combs_graphs(tool_combs_dict, combs_list, trial_IDs, showGraphs=True):

    dict = {"empty": "e", "muscle": "m", "cottonoid": "c", "suction": "sc", "grasper": "g", "string": "st" }

    new_combs_list = [[dict[y] for y in x] for x in combs_list]

    for trial in ["S201T1", "S305T1"]:

        trial_combs_df = pd.DataFrame()
        trial_combs_df["combs"] = tool_combs_dict[trial]
        trial_combs_df["frame"] = [x for x in range(1, len(tool_combs_dict[trial])+1)]

        if(showGraphs):
            plt.scatter(trial_combs_df["frame"], trial_combs_df["combs"], s=30, marker=path, c="black")
            plt.title(trial+" tool combs pattern across trial", fontsize=26)
            ax = plt.gca()
            ax.set_xlabel('Trial Frames', fontsize=20)
            ax.set_ylabel('Possible Tool Combinations', fontsize=20)
            ax.tick_params(labelsize=12.0, length=5.0, width=2.0)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.2)  # change width

            plt.yticks(range(0, len(new_combs_list)), new_combs_list)
            #plt.xlabel("Trial Frames")
            #plt.ylabel("Possible Tool Combinations")
            plt.tight_layout()
            plt.show()
            plt.clf()

        # trial_combs_df["groups"] = pd.cut(trial_combs_df.frame, bins)
        #
        # groups_list = list(trial_combs_df["groups"].unique())
        # groups_list.sort(reverse=False)
        #
        # for group_index, group in enumerate(groups_list):
        #
        #     filtered = trial_combs_df.loc[trial_combs_df["groups"] == group]
        #     comb_counts = filtered["comb"].value_counts()

def dataset_overview(truth, trial_IDs_test, trial_IDs_training, tools):

    print("overview of dataset")

    truth2 = truth.loc[truth["label"].isin(tools)]

    truth_training = truth2.loc[truth2["trial_id"].isin(trial_IDs_training)]
    truth_test = truth2.loc[truth2["trial_id"].isin(trial_IDs_test)]

    print("training \n", truth_training["label"].value_counts())
    print("testing \n", truth_test["label"].value_counts())

    test_props = [val for val in truth_training["label"].value_counts().to_list()]
    train_props = [val for val in truth_test["label"].value_counts().to_list()]

    plt.bar(truth_training["label"].value_counts().index, truth_training["label"].value_counts().to_list())
    #plt.bar(truth_training["label"].value_counts().index, train_props)
    plt.title("training set overview")
    #plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.show()
    plt.clf()

    plt.bar(truth_test["label"].value_counts().index, truth_test["label"].value_counts().to_list())
    #plt.bar(truth_test["label"].value_counts().index, test_props)
    plt.title("testing set overview")
    #plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.show()
    plt.clf()

# generates all the APM for a give detections files for a single trial_id
def generate_APMs_from_detections_file(fileName, truthName, showGraphs=False):
    '''
    Change trial_ID stuff so that it uses everything
    Change video dimensions -> maybe check trials and see if all have same dimensions. else throw error
    change get num frames to add up for all trials present in test set -> create list of trial_IDs in test set
    '''

    #***need to make file for frame_to_trial_mapping (and etv_outcomes (video dimensions)

    np.seterr(divide='ignore', invalid='ignore')

    data = pd.read_csv(fileName, names=["trial_frame", "frame", "x1", "y1", "x2", "y2", "score", "label", "trial_id"],
                       header=0)  # read in the input data file

    trial_IDs = list(data["trial_id"].unique())  # test set trial ids

    # total_frames = int(max(data["frame"]))

    all_tools = list(data["label"].unique())
    all_tools.sort()

    # list of tools to calculate metrics for
    tools = ['tool']  # drill # tool (hopefully these did not affect results)
    tools.sort()

    # -----------------------read ground truth to calculate confidence score threshold for tools
    truth = pd.read_csv(truthName, names=["trial_frame", "x1", "y1", "x2", "y2", "label"], header=0)
    truth = truth.sort_values(by="trial_frame")
    truth.dropna(inplace=True)
    truth["trial_id"] = [i[0:8] for i in truth["trial_frame"]]
    truth["frame"] = [int(i[-11:-4]) for i in truth["trial_frame"]]
    #truth = truth.loc[truth["trial_id"].isin(trial_IDs)]
    trial_IDs_truth = list(truth["trial_id"].unique())
    trial_IDs_test = list(data["trial_id"].unique())
    # trial_IDs_test = [x for x in trial_IDs if x in get_trial_test_set()]
    # trial_IDs_train = [x for x in trial_IDs if x not in get_trial_test_set()]
    #trial_IDs = [x for x in trial_IDs if x not in get_trial_test_set()]

    trial_frames_dict = get_num_frames(list(truth["trial_id"].unique()), truth)  # returns a dict of number of frames (val) for each trial (key)

    #dataset_overview(truth, trial_IDs_test, trial_IDs_train, all_tools)

    # get the best tool thresholds based on f1 score compared to ground truths
    # truth_test = truth.loc[truth["trial_id"].isin(trial_IDs_test)]

    #print(data.head())
    #print(truth.head())
    print(trial_IDs)

    best_tool_thresholds, tool_precisions = find_best_thresholds(data, truth, trial_IDs_test, all_tools, showGraphs=True)
    high_score_data = get_high_score_tools(data, all_tools, best_tool_thresholds)
    print(best_tool_thresholds)
    exit()

    #tool proportions in each bin **
    #calculate_tool_patterns(truth, tools, trial_IDs, trial_frames_dict, bin_count=10)

    #tool presence charts
    #trial_groups_dict = label_distributions(truth, tools, trial_IDs, showFig=False, saveFig=False)

    #new_tool_data = high_score_data[["trial_frame", "frame", "label", "trial_id"]].copy() #****
    # new_tool_data = truth[["trial_frame", "frame", "label", "trial_id"]].copy()
    # for tool in tools:
    #     smooth_trials = smooth_labels(truth, tool, threshold=5)  # threshold=15 => means gaps < 15 frames will filled in with tool
    #     miss_tool_data, removed_data = fill_tool_gaps(smooth_trials, truth, tool, min_frames=3)
    #     new_tool_data = pd.concat([new_tool_data, miss_tool_data], ignore_index=True)
    #     new_tool_data = pd.merge(new_tool_data, removed_data, how='outer', indicator=True).query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)

    #combinations of tool use
    tool_combs_df, combs_list, good_trial_IDs, blood_loss, tth_scaled, success, diversity_dict, tool_combs_dict = find_tool_patterns(truth, tools, trial_IDs, trial_frames_dict, showGraphs=False)
    tool_combs_df2, combs_list2, good_trial_IDs2, blood_loss2, tth_scaled2, success2, diversity_dict2, tool_combs_dict2 = find_tool_patterns(high_score_data, tools, trial_IDs_test, trial_frames_dict, showGraphs=False)
    #^^ the second one is only for the testing set and only from yolo detections

    # gen_tool_combs_graphs(tool_combs_dict, combs_list, good_trial_IDs, showGraphs=True) #outputs graph of high and lot entrop tool combs

    '''
    plt.figure(figsize=(10, 7))
    plt.title("Dendograms")
    dend = shc.dendrogram(shc.linkage(tool_combs_df, method='ward'))
    plt.show()
    plt.clf()

    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(tool_combs_df)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(tool_combs_df)
    print(pca.explained_variance_ratio_)'''

    cols = tool_combs_df.columns
    tool_combs_df["trial_id"] = good_trial_IDs
    tool_combs_df2["trial_id"] = good_trial_IDs2
    tool_combs_df["blood_loss"] = blood_loss
    tool_combs_df["tth"] = tth_scaled
    tool_combs_df["success"] = success
    tool_combs_df2["success"] = success2
    tool_combs_df["trial"] = [int(x[-1:]) for x in tool_combs_df["trial_id"]]

    #tool_combs_df["cluster"] = cluster.labels_

    #principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    #great graph showing PC1 and PC2 and clustering. Clear separation based on the two PCs.
    #Now that does this mean?? Seems like PC1 explains the clustering mostly, PC2 not as well but helps

    #plt.scatter(principalDf["PC1"], principalDf["PC2"], c=tool_combs_df["success"], cmap='rainbow')
    #plt.scatter(principalDf["PC1"], principalDf["PC2"], c=list(tool_combs_df["trial"]), cmap='rainbow')
    # plt.scatter(blood_loss, tool_combs_df["entropy"], c=cluster.labels_, cmap='rainbow')
    # plt.title("bloodloss vs entropy, colors=clusters")
    # #plt.ylim(0, 1.0)
    # plt.show()
    # plt.clf()
    # exit()

    '''
    principalDf = pd.concat([principalDf, tool_combs_df["cluster"], tool_combs_df["entropy"]], axis=1)

    trial1v2 = [(i-1) for i in list(tool_combs_df["trial"])]

    demograph = pd.read_csv("socal_participant_demographics.csv")

    demograph2 = demograph.copy()
    demograph["participant_id"] = ["S"+x+"T1" for x in demograph["participant_id"]]
    demograph2["participant_id"] = ["S"+x+"T2" for x in demograph2["participant_id"]]

    demograph = pd.concat([demograph, demograph2], axis=0)
    demograph = demograph.loc[demograph["participant_id"].isin(good_trial_IDs)]
    demograph = demograph.sort_values(by=["participant_id"]).reset_index(drop=True)

    demograph["training_status"] = demograph["training_status"].map({"Attending": 1, "Trainee": 0})

    attend_trials_list = list(demograph.loc[demograph["training_status"] == 1]["participant_id"].unique())
    attend_trials_list = ["S"+x for x in attend_trials_list]
    trainee_trials_list = list(demograph.loc[demograph["training_status"] == 0]["participant_id"].unique()) #?? 93 vs 97
    trainee_trials_list = ["S" + x for x in trainee_trials_list]

    attend_truth = truth.loc[truth["trial_id"].isin(attend_trials_list)]
    trainee_truth = truth.loc[truth["trial_id"].isin(trainee_trials_list)]
    '''
    success_trials_list = list(tool_combs_df[tool_combs_df["success"] == 1]["trial_id"].unique())
    fail_trials_list = list(tool_combs_df[tool_combs_df["success"] == 0]["trial_id"].unique())

    print("t-test results")
    print(ttest_ind(list(tool_combs_df.loc[tool_combs_df["trial_id"].isin(success_trials_list)]["entropy"]), list(tool_combs_df.loc[tool_combs_df["trial_id"].isin(fail_trials_list)]["entropy"]), usevar='unequal', value=0))
    print(tool_combs_df[["success", "entropy"]].groupby("success").describe())  # means and stdv etc

    success_truth = truth.loc[truth["trial_id"].isin(success_trials_list)]
    fail_truth = truth.loc[truth["trial_id"].isin(fail_trials_list)]

    #tool_combs_df["trial1v2"] = trial1v2  #0 or 1 based on trial
    #tool_combs_df["training_status"] = demograph["training_status"]

    tool_combs_df["trial_abbrev"] = [x[1:4] for x in good_trial_IDs]
    # print(len(tool_combs_df["trial_abbrev"]))

    '''
    tool_combs_df = tool_combs_df.sort_values(by=["entropy"])

    # Plot of shannon entropy change for each surgeon & success/fail counts
    for abbrev in tool_combs_df["trial_abbrev"].unique():

        success_val = True
        surgeon_trials = tool_combs_df.loc[tool_combs_df["trial_abbrev"] == abbrev] #should be two
        for index, row in surgeon_trials.iterrows():

            if (row["trial_id"] in fail_trials_list): success_val = False
            else: success_val = True

            if(row["trial_id"][-2:] == 'T1'):
                if(success_val == False): plt.plot(row["trial_abbrev"], row["entropy"], 'bx')
                else: plt.plot(row["trial_abbrev"], row["entropy"], 'bo')

            elif(row["trial_id"][-2:] == 'T2'):
                if (success_val == False):
                    plt.plot(row["trial_abbrev"], row["entropy"], 'rx')
                else:
                    plt.plot(row["trial_abbrev"], row["entropy"], 'ro')

    #plt.legend(loc="upper left")
    plt.xticks(rotation=90)
    plt.xlabel("Surgeon Identifier")
    plt.ylabel("Normalized Shannon Entropy")
    plt.yticks(np.arange(0.0, 0.7, 0.1))
    plt.title("blue = trial 1; red = trial 2")
    plt.show()'''

    # tool proportions in each bin **
    # calculate_tool_patterns(success_truth, tools, success_trials_list, trial_frames_dict, bin_count=10)
    # calculate_tool_patterns(fail_truth, tools, fail_trials_list, trial_frames_dict, bin_count=10)

    # plt.scatter(principalDf["PC1"], principalDf["PC2"], c=tool_combs_df["success"], cmap='rainbow')
    # plt.title("PC1 vs PC2, colors=success")
    # plt.show()
    # plt.clf()
    # exit()

    training_entropy = tool_combs_df.loc[tool_combs_df["trial_id"].isin(trial_IDs_train)]["entropy"]
    training_success = tool_combs_df.loc[tool_combs_df["trial_id"].isin(trial_IDs_train)]["success"]

    test_entropy = tool_combs_df2.loc[tool_combs_df2["trial_id"].isin(trial_IDs_test)]["entropy"]  #change df to 2 for detections
    test_success = tool_combs_df2.loc[tool_combs_df2["trial_id"].isin(trial_IDs_test)]["success"] #change df to 2 for detections

    # print(list(tool_combs_df.loc[tool_combs_df["trial_id"].isin(trial_IDs_test)]["success"]))
    # print(list(tool_combs_df2.loc[tool_combs_df2["trial_id"].isin(trial_IDs_test)]["success"]))

    print("---------")
    X = sm.add_constant(training_entropy)
    model = sm.Logit(training_success, X).fit()
    print_model = model.summary()
    print(print_model)
    Y_hat = sm.add_constant(test_entropy)
    test_predict = model.predict(Y_hat)
    #print(test_predict)
    test_predict2 = list(map(round, test_predict))
    # confusion matrix
    cm = metrics.confusion_matrix(test_success, test_predict2)
    print("Confusion Matrix : \n", cm)

    X_ = np.linspace(0.2,0.7, 500)
    X_ = sm.add_constant(X_)
    Y_ = model.predict(X_)

    # #plt.plot(test_entropy, test_predict, "bo", label="Regression")
    # plt.plot(X_, Y_, "bo", label="Regression", markersize=2)
    # plt.plot(test_entropy, test_success, "ro", label="Ground truth", markersize=5)
    # plt.title("Logistic Regression - Predicting Trial Success with Entropy")
    # plt.xlim(0.2, 0.7)
    # ax = plt.gca()
    # ax.set_xlabel('Normalized Shannon Entropy', fontsize=20)
    # ax.set_ylabel('Trial Success Likelihood', fontsize=20)
    # ax.tick_params(labelsize=18.0, length=5.0, width=2.0)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     ax.spines[axis].set_linewidth(2.0)  # change width
    # plt.xticks(np.arange(0.2, 0.7, 0.05))
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # #plt.ylim(-0.1, 1.1)
    # #plt.legend(frameon=False)
    # plt.show()
    # plt.clf()

    # accuracy score of the model
    print('% correctly predicted = ', metrics.accuracy_score(test_success, test_predict2))
    fpr, tpr, _ = metrics.roc_curve(test_success, test_predict2)
    auc = metrics.roc_auc_score(test_success, test_predict2)
    # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    # plt.show()
    # plt.clf()
    precision, recall, thresholds = metrics.precision_recall_curve(test_success, test_predict)
    precision = np.insert(precision, 0, 0.0)
    recall = np.insert(recall, 0, 1.0)
    auc = metrics.average_precision_score(test_success, test_predict)
    auc = round(auc, 3)

    print(list(test_success), list(test_predict))
    print(thresholds)

    # plt.plot(recall, precision, label="AUC = " + str(auc))
    # plt.title("Precision-Recall Curve from Logistic Regression of Shannon Entropy to Predict Trial Success \n AUC: %s" % (str(auc)), fontsize=20)
    # plt.xticks(np.arange(0, 1.1, 0.1))
    # ax = plt.gca
    # ax.set_xlabel('Recall (Sensitivity)', fontsize=20)
    # ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=20)
    # ax.tick_params(labelsize=18.0, length=5.0, width=2.0)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     ax.spines[axis].set_linewidth(2.0)  # change width
    #
    # print(precision)
    # #plt.legend(loc="lower left", frameon=False)
    # plt.show()
    # plt.clf()
    # print("---------")

    #show graph of clusters for each tool combination -> trials on y, col on x, colors
    # for col in cols:
    #     plt.scatter(tool_combs_df[col], tool_combs_df["trial_id"], c=demograph["training_status"], cmap='rainbow')
    #
    #     if(str(col).isnumeric()): plt.title(combs_list[col])
    #     else: plt.title(col)
    #     plt.show()
    #     plt.clf()
    #
    # print(list(cols))

    '''
    rsq_list = []
    best_col_list = []

    for col in cols:
        try:
            X = sm.add_constant(tool_combs_df[col])
            model = sm.Logit(tool_combs_df["trial1v2"], X).fit()
            #model = sm.OLS(tool_combs_df["blood_loss"], X).fit()
            print_model = model.summary()
            # print(print_model)
            rsq_list.append(model.prsquared) #or prsquared for logit
            best_col_list.append(col)
        except:
           print(list(tool_combs_df[col]))

    best_col_df = pd.DataFrame()
    best_col_df["rsq"] = rsq_list
    best_col_df["col"] = best_col_list
    best_col_df.sort_values(by=['rsq'], ascending=False, inplace=True)
    #print(best_col_df[0:5])
    vals = list(best_col_df['col'][0:5])  # +[len(combs_list)]

    for index, i in enumerate(list(best_col_df['col'][0:5])):
        if(i != "entropy"): print(i, combs_list[i], list(best_col_df['rsq'])[index])
        else: print(i, list(best_col_df['rsq'])[index])

    print("final: tool combs vs training status ------------")
    print(vals)
    X = sm.add_constant(tool_combs_df[vals])
    model = sm.Logit(tool_combs_df["trial1v2"], X).fit()
    #model = sm.OLS(tool_combs_df["blood_loss"], X).fit()
    print_model = model.summary()
    print(print_model)

    for col in vals:

        trainee = tool_combs_df.query('trial1v2 == 0')[col]
        attending = tool_combs_df.query('trial1v2 == 1')[col]

        if (col != "entropy"):
            print(combs_list[col], ttest_ind(trainee, attending))
        else:
            print(col, ttest_ind(trainee, attending))

        print(tool_combs_df[[col, "trial1v2"]].groupby("trial1v2").describe())  #means and stdv etc

    # trial_series = pd.Series(frame_entropy).value_counts().sort_index()
    # plt.bar(range(len(trial_series)), trial_series.values, align='center')
    # plt.xticks(range(len(trial_series)), trial_series.index.values, size='small')
    # plt.title(trial+" tool combs histogram")
    # plt.show()
    # plt.clf()
    '''

    trial_diversity = pd.DataFrame()
    bin_count = 10
    diversity_means_fail = [0.0 for x in range(0, 10)]
    diversity_means_succ = [0.0 for x in range(0, 10)]

    for trial in good_trial_IDs: ##change here and put into method
        bins = list(np.linspace(-1, len(diversity_dict[trial])+1, bin_count + 1))
        trial_df = pd.DataFrame()
        trial_df["cumu_entr"] = pd.Series(diversity_dict[trial])  #cumulative entropy for a trial, not binned yet
        trial_df["groups"] = pd.cut(trial_df.index, bins)
        groups_list = list(trial_df["groups"].unique())
        trial_means = []

        for group_index, group in enumerate(groups_list):
            filtered = trial_df.loc[trial_df["groups"] == group]
            trial_means.append( filtered["cumu_entr"].mean() )

        if(trial in fail_trials_list): #***change here
            diversity_means_fail = np.add(diversity_means_fail, trial_means)
        else:
            diversity_means_succ = np.add(diversity_means_succ, trial_means)

        #plt.plot(range(1, len(diversity_dict[trial])+1), [x for x in diversity_dict[trial]])

    diversity_means_both = np.add(diversity_means_fail, diversity_means_succ)
    diversity_means_both = diversity_means_both * (1/ len(good_trial_IDs))
    diversity_means_fail = diversity_means_fail * (1 / len(fail_trials_list)) #***change here
    diversity_means_succ = diversity_means_succ * (1 / len(success_trials_list))
    # print(diversity_means_both)

    #plt.show()
    #plt.clf()

    bin_centers2 = np.arange(0.05, 1.05, 0.1)
    plt.title("Average Cumulative Shannon Entropy of SOCAL Dataset Trials")
    #plt.xticks(np.arange(0.0, 1.0, (1.0 / bin_count)))
    plt.plot(bin_centers2, diversity_means_succ, label="Success")
    plt.plot(bin_centers2, diversity_means_fail, label="Fail")
    #plt.plot(bin_centers2, diversity_means_both, label="All")
    plt.legend(frameon=False, loc="lower left", shadow=False, fontsize=20)
    ax = plt.gca()
    ax.set_xlabel('Trial Progress', fontsize=20)
    ax.set_ylabel('Normalized Shannon Entropy', fontsize=20)
    ax.tick_params(labelsize=18.0, length=5.0, width=2.0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.0)  # change width

    plt.yticks(np.arange(0.0, 0.6, 0.1))
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.show()
    plt.clf()

    #print("MAP: ", np.nansum(np.array(list(tool_precisions.values()))) / len(tools))

def main():

    #fileName = "..\\yolov4_socal_detections_4.7_fixed.csv"

    #fileName = "/Users/alanbalu/Library/CloudStorage/OneDrive-Personal/Alan/GUSOM 2025/Donoho Research/ETV 1.7.23/yolov4_etv_detections_1.4.23_fixed.csv"
    fileName = "/Users/alanbalu/Library/CloudStorage/OneDrive-Personal/Alan/GUSOM 2025/Donoho Research/ETV 1.7.23/yolov4_etv_YT_detections_1.4.23_fixed.csv"

    truth = "/Users/alanbalu/Library/CloudStorage/OneDrive-Personal/Alan/GUSOM 2025/Donoho Research/ETV 1.7.23/etv_truth.csv"
    generate_APMs_from_detections_file(fileName, truth, showGraphs=False)

if __name__ == "__main__":
    main()