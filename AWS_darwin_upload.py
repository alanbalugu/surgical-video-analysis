import requests
import sys
import os
import argparse

#python3 AWS_darwin_upload_new.py --batch True --type video --storage_name donoholab1 --team_name surgical-data-science-collaborative --dataset_name site_102_donation_103 --api_key DmEt_0-.ijvbnnYicCjTLntK2R3UQwFGcdIJaSzO --fps 5 --testing True --files_list ".txt"

api_key = ""
team_slug = ""
storage_name = ""
dataset_slug = ""

def upload(data_type, AWS_file_name, fps="native", testing=False,api_key="api-key"):

    print("uploading...", AWS_file_name, end=" ")

    if(len(AWS_file_name) < 2): exit()

    display_name = os.path.split(AWS_file_name)[1]

    # path_list = os.path.normpath(AWS_file_name).split(os.sep)
    # folder_name = path_list[len(path_list)-3][0]
    # display_name = folder_name + "_" + display_name
    # print(display_name)

    payload = {
        "items": [
            {
                 "as_frames": False,
                 "key": AWS_file_name,
                 "filename": display_name,
                 "fps": fps,
                 "type": "video",
    }
        ],

        "storage_name": storage_name,
    }

    print("\n", payload)

    if(testing==True):

        print("Just Testing!")

    elif(testing==False):

        print("NOT TESTING!!")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"ApiKey {api_key}"
        }
        
        response = requests.put(
            f"https://darwin.v7labs.com/api/teams/{team_slug}/datasets/{dataset_slug}/data",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            print("request failed", response.status_code, response.text)
        else:
            print("success", response.text)
    else:
        print("bad testing parameter")

def main():

    n = len(sys.argv)
    print("Total arguments passed:", n)

#--------

    parser = argparse.ArgumentParser(description='Script to upload files from AWS to Darwin V7')

    parser.add_argument("--batch", help="Boolean, Use if you are uploading a batch of files specified in a text file.", type=bool, default=False)

    parser.add_argument("--type", help="String, Data type - 'image' or 'video'.", type=str, default="video")

    parser.add_argument("--storage_name", help="String, AWS storage name of bucket.", type=str, default="")

    parser.add_argument("--team_name", help="String, Name of Darwin team.", type=str, default="")

    parser.add_argument("--dataset_name", help="String, Name of Darwin dataset (all lowercase and replace space with dash (-).", type=str, default="")

    parser.add_argument("--api_key", help="String, API key for Darwin V7.", type=str, default="")

    parser.add_argument("--files_list", help="String, List of files to upload.", type=str, default="")

    parser.add_argument("--fps", help="String, FPS to upload video.", type=str, default="native")

    parser.add_argument("--files", help="String(s), Files to upload separated by spaces.", nargs='*')

    parser.add_argument("--testing", help="Boolean, Testing flag", type=bool, default=False)

    args = parser.parse_args()

    global api_key
    global team_slug
    global storage_name
    global dataset_slug

    api_key = args.api_key
    team_slug = args.team_name
    storage_name = args.storage_name
    dataset_slug = args.dataset_name

    testing = args.testing
    data_type = args.type
    fps = args.fps
    batch = args.batch
    files = args.files
    files_list = args.files_list

    print("key", api_key)
    print("team", team_slug)
    print("S3", storage_name)
    print("dataset",dataset_slug)

    print("test?", testing)
    print("type", data_type)
    print("fps", fps)
    print("batch", batch)
    print("files", files)
    print("txt", files_list)

    if((fps.isnumeric() == False) and (fps != "native")):
        print("bad fps...exiting")
        exit()

    if((dataset_slug.isnumeric() == True)):
        print("bad dataset name...exiting")
        exit()
# 
#--------

    if (n<3):
        print("no arguments passed...exiting")
        exit()


    if(data_type not in ["image", "video"]):
        print("bad data_type, try again...exiting")
        exit()

    print("data type: ", data_type)
    print("dataset: ", dataset_slug)

    if((batch == False) and (files is not None)):

        #python3 AWS_darwin_upload.py video video_path
        for file in files:

            print("uploading...", file, end=" ")

            upload(data_type=data_type, AWS_file_name=file, fps=fps, testing=testing)

    elif(batch == True):

        with open(files_list, "r", encoding="utf-8") as file:

            for line in file:

                file_name = line.strip()
                upload(data_type=data_type, AWS_file_name=file_name, fps=fps, testing=testing, api_key=api_key)


if __name__ == "__main__":
    main()
