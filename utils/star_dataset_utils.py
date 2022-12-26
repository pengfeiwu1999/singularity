import json
import pdb
import csv
annotation_train_source = "/home/wupf/STAR/Questions, Answers and Situation Graphs/STAR_train.json"
annotation_val_source = "/home/wupf/STAR/Questions, Answers and Situation Graphs/STAR_val.json"
annotation_train_target = "/home/wupf/singularity/to/data/anno_downstream/star_mc_train.json"
annotation_val_target = "/home/wupf/singularity/to/data/anno_downstream/star_mc_val.json"
video_to_key_frame = "/home/wupf/STAR/Situation Video Data/Video_Keyframe_IDs.csv"
train_file = open(annotation_train_source, "r", encoding="utf8")
val_file = open(annotation_val_source, "r", encoding="utf8")
video_to_key_frame_dict = {}
with open(video_to_key_frame, "r", encoding="utf8") as f:    # 用于记录每个Video的关键帧编号
    lines = f.readlines()
    for iter in lines[1:]:
        info = iter.split('"')[1].split(",")
        info[0] = info[0][2:]
        info[-1] = info[-1][:-2]
        info_num = [int(iter.strip().strip("'")) for iter in info]
        video_to_key_frame_dict[iter.split(",")[0]] = info_num

anno_train_list = json.load(train_file)
anno_val_list = json.load(val_file)
target_train_list = []
target_val_list = []
for iter in anno_train_list:
    if not len(iter["choices"]) == 4:
        print("答案数目有问题")
    info_dict = {}
    info_dict["video"] = iter["video_id"]+".mp4"
    info_dict["caption"] = []
    info_dict["key_frames"] = video_to_key_frame_dict[iter["question_id"]]
    info_dict["question_id"] = iter["question_id"]
    for ans in iter["choices"]:
        if ans["choice"] == iter["answer"]:
            if "answer" in info_dict:
                print("答案有重复")
            info_dict["answer"] = ans["choice_id"]
        info_dict["caption"].append("Question: "+iter["question"]+" Answer: "+ans["choice"])
    if "answer" not in info_dict:
        print("没找到答案")

    target_train_list.append(info_dict)

target_train_file = open(annotation_train_target, "w", encoding="utf8")
json.dump(target_train_list, target_train_file)

for iter in anno_val_list:
    if not len(iter["choices"]) == 4:
        print("答案数目有问题")
    info_dict = {}
    info_dict["video"] = iter["video_id"]+".mp4"
    info_dict["caption"] = []
    info_dict["key_frames"] = video_to_key_frame_dict[iter["question_id"]]
    info_dict["question_id"] = iter["question_id"]
    for ans in iter["choices"]:
        if ans["choice"] == iter["answer"]:
            if "answer" in info_dict:
                print("答案有重复")
            info_dict["answer"] = ans["choice_id"]
        info_dict["caption"].append("Question:"+iter["question"]+" Answer:"+ans["choice"])
    if "answer" not in info_dict:
        print("没找到答案")

    target_val_list.append(info_dict)

target_val_file = open(annotation_val_target, "w", encoding="utf8")
json.dump(target_val_list, target_val_file)



