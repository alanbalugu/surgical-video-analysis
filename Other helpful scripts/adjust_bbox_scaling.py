import numpy as np
import pandas as pd
import math

def get_socal_test_set():
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

def get_video_dimensions(trial_ID):

    input_df = pd.read_csv("SOCAL Trial outcomes condensed.csv")

    height = input_df.loc[input_df["trial_id"] == trial_ID]["trial_video_height"].iloc[0]
    width = input_df.loc[input_df["trial_id"] == trial_ID]["trial_video_width"].iloc[0]

    if(math.isnan(height)):
        height = 1280
        width = 1920

    return width, height

def main():

    #script scales the bounding boxes to 1000 x 1000 to standardize the dimentions of the annotations

    #socal_gt = pd.read_csv("socal.csv", names=["trial_frame", "x1", "y1", "x2", "y2", "label"], header=0)

    detections = pd.read_csv("yolov4_socal_detections_4.7_fixed.csv")

    trials = [img[0:6] for img in detections["trial_frame"]]
    detections["trial"] = trials

    uniq_trials = list(detections["trial"].unique())

    print(uniq_trials)

    trial_dict = {}

    for trial in uniq_trials:

        trial_dict[trial] = get_video_dimensions(trial)

    scaled_data = detections.copy()

    print(scaled_data.head())
    x1 = []
    x2 = []
    y1 = []
    y2 = []

    for index, row in scaled_data.iterrows():
        x1.append(row["x1"] * trial_dict[row["trial"]][0] / 1000 )
        x2.append( row["x2"] * trial_dict[row["trial"]][0] / 1000 )
        y1.append(row["y1"] * trial_dict[row["trial"]][1] / 1000)
        y2.append(row["y2"] * trial_dict[row["trial"]][1] / 1000)

    scaled_data["x1"] = x1
    scaled_data["x2"] = x2
    scaled_data["y2"] = y1
    scaled_data["y2"] = y2

    scaled_data.to_csv("yolov4_socal_detections_4.7_fixed_scaled.csv", index=False)


if __name__ == "__main__":
    main()
