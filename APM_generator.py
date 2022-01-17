import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter
import statistics
import math

def normalize_coords(data, frame_width, frame_height):

    data["x1"] = data["x1"] / frame_width
    data["x2"] = data["x2"] / frame_width
    data["y1"] = data["y1"] / frame_height
    data["y2"] = data["y2"] / frame_height

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

def count_frames_w_x_tools(data, search_thresh, num_tools):

    frames_w_tools = [len(list(j)) for i,j in groupby(list(data.loc[data["score"] > search_thresh]["frame"]))]
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

    h = y2 - y1
    w = x2 - x1

    return (h*w)

def get_bounding_box_list(tool_row):

    return [ tool_row["x1"].iloc[0], tool_row["y1"].iloc[0], tool_row["x2"].iloc[0], tool_row["y2"].iloc[0] ]

def calc_tools_ovelap_area(a, b):


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
        return statistics.pstdev(data.loc[(data["label"] == tool) & (data["score"] > search_thresh)][coord].tolist())
    except:
        return 0 

def calc_tool_coord_difference_sd(data, search_thresh, tool, coord1, coord2):

    try:

        tool_filtered = data.loc[(data["label"] == tool) & (data["score"] > search_thresh)]

        center_data = [ (coord2_i + coord1_i)/2 for coord1_i, coord2_i in zip(tool_filtered[coord1].tolist(), tool_filtered[coord2].tolist())]

        return statistics.pstdev(center_data)

    except:
        return 0 

def calc_tool_distance_covered(data, search_thresh, tool):

    tool_filtered = data.loc[(data["label"] == tool) & (data["score"] > search_thresh)]

    unique_instances = list(tool_filtered["frame"].unique())

    ranges =[]

    for key, group in groupby(enumerate(unique_instances), lambda i: i[0] - i[1]):         
        group = list(map(itemgetter(1), group))
        group = list(map(int,group))
        ranges.append(group)

    consec_coords = []

    total_tool_dist = 0
    total_tool_frames = 0

    for consec_frames in ranges:    #gives a list of frames with the tools at search_thresh

        for frame in consec_frames:


            if(tool == "cottonoid" or tool == "muscle"):
                consec_coords.append([ (tool_filtered.loc[tool_filtered["frame"] == frame]["x2"].iloc[0] + tool_filtered.loc[tool_filtered["frame"] == frame]["x1"].iloc[0])/2, (tool_filtered.loc[tool_filtered["frame"] == frame]["y2"].iloc[0] + tool_filtered.loc[tool_filtered["frame"] == frame]["y1"].iloc[0])/2])
            else:
                consec_coords.append([ tool_filtered.loc[tool_filtered["frame"] == frame]["x1"].iloc[0], tool_filtered.loc[tool_filtered["frame"] == frame]["y1"].iloc[0] ])

            total_tool_frames += 1

        for index in range(1, len(consec_coords)):

            total_tool_dist += math.sqrt( (consec_coords[index][1]-consec_coords[index-1][1])**2 + (consec_coords[index][0]-consec_coords[index-1][0])**2 )

        consec_coords = []

    try:
        total_tool_speed = (total_tool_dist/total_tool_frames)
    except:
        total_tool_speed = 0.0

    return total_tool_dist, total_tool_speed


def calc_total_tool_overlap(data, search_thresh, tool1, tool2):

    sc_overlap = 0
    gm_overlap = 0

    high_score_tools = data.loc[data["score"] > search_thresh]

    tool1_filtered = data.loc[(data["label"] == tool1) & (data["score"] > search_thresh)]
    tool2_filtered = data.loc[(data["label"] == tool2) & (data["score"] > search_thresh)]

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

                        if( ((tool1_row["label"].iloc[0] == "cottonoid") and (tool2_row["label"].iloc[0] == "suction")) or ((tool1_row["label"].iloc[0] == "suction") and (tool2_row["label"].iloc[0] == "cottonoid")) ):

                            overlap_area = calc_tools_ovelap_area(get_bounding_box_list(tool1_row), get_bounding_box_list(tool2_row))

                            sc_overlap += overlap_area

                            #print(tool1_row["label"].iloc[0], tool2_row["label"].iloc[0], overlap_area)

                        elif( ((tool1_row["label"].iloc[0] == "grasper") and (tool2_row["label"].iloc[0] == "muscle")) or ((tool1_row["label"].iloc[0] == "muscle") and (tool2_row["label"].iloc[0] == "grasper")) ):

                            overlap_area = calc_tools_ovelap_area(get_bounding_box_list(tool1_row), get_bounding_box_list(tool2_row))

                            gm_overlap += overlap_area

                            #print(tool1_row["label"].iloc[0], tool2_row["label"].iloc[0], overlap_area)

    return sc_overlap, gm_overlap

def generate_APMs_from_detections_file(fileName):

    data = pd.read_csv(fileName) #read in the input data file

    data = normalize_coords(data, 1920, 1080)  # normalize the coords relative to frame size

    total_frames = int(max(data["frame"]))
    total_frames_w_tools = len(list(data["frame"].unique()))

    #tools = list(data["label"].unique())
    #main_tools = ['suction', 'grasper', 'cottonoid', 'string', 'muscle']
    tools = ['suction', 'grasper', 'cottonoid', 'string', 'muscle']

    APM_data = pd.DataFrame() #adding in the columns as they are computed

    APM_data = add_video_info(APM_data, data["vid"][0], 1920, 1080, total_frames)

    search_thresh = 0.5

    for tool in tools:
        APM_data["frames_w_"+str(tool)] = [count_frames_w_tool(data, search_thresh, tool) / total_frames]

    for i in range(1, 6):
        APM_data["frames_w_" + str(i) + "_tools"] = count_frames_w_x_tools(data, search_thresh, i)

    APM_data["frames_w_at_least_1_tool"] = len(data.loc[data["score"] > search_thresh]["frame"].unique())

    for tool in tools:

        APM_data["first_frame_w_"+str(tool)] = [find_first_frame_w_tool(data, search_thresh, tool)]

    total_in_n_outs = 0

    for tool in tools:

        APM_data[str(tool)+"_in_n_outs"] = [calc_in_n_outs(data, search_thresh, tool)]
        total_in_n_outs += APM_data[str(tool)+"_in_n_outs"][0]

    APM_data["total_in_n_outs"] = total_in_n_outs

    for tool in tools:
        APM_data["area_covered_"+tool] = [calc_total_tool_area(data, search_thresh, "grasper") / total_frames]

    for tool in tools:
        for coord in ["x1", "y1", "x2", "y2"]:
            
            APM_data[tool+"_"+coord+"_"+"sd"] = calc_tool_coord_sd(data, search_thresh, tool, coord)

    for tool in tools:
        APM_data[tool+"_x_center_sd"] = calc_tool_coord_difference_sd(data, search_thresh, tool, "x1", "x2")
        APM_data[tool+"_y_center_sd"] = calc_tool_coord_difference_sd(data, search_thresh, tool, "y1", "y2")

    for tool in tools:    
        
        tool_distance, tool_speed = calc_tool_distance_covered(data, search_thresh, tool)
        APM_data["distance_covered_"+tool] = tool_distance
        APM_data["speed_"+tool] = tool_speed

    sc_overlap, gm_overlap = calc_total_tool_overlap(data, search_thresh, "cottonoid", "grasper")

    APM_data["sc_overlap"] = [sc_overlap]
    APM_data["gm_overlap"] = [gm_overlap]

    #print(data.head())
    #print(APM_data.iloc[:,-4:])

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