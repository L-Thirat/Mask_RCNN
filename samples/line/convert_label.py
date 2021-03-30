import glob
import json
import os

dir_dataset = "../../datasets/"
project = "line"


def labelme2one(dir_path):
    one_summarize = {}
    lst_path = glob.glob(dir_path + "*.json")
    for p in lst_path:
        if "via_region" not in p:
            with open(p, "r") as f:
                data = json.load(f)
                filename = (data["imagePath"].split("\\")[-1])
                one_summarize[filename] = {}
                one_summarize[filename]["filename"] = filename

                sel_x, sel_y = [], []
                for item in data["shapes"]:
                    if "line" in item["label"]:
                        for point in item["points"]:
                            sel_x.append(int(point[0]))
                            sel_y.append(int(point[1]))

                one_summarize[filename]["regions"] = {
                        "0": {
                            "shape_attributes": {
                                "name": "polygon",
                                "all_points_x": sel_x,
                                "all_points_y": sel_y
                            },
                            "region_attributes": {}
                        }
                    }

    with open(dir_path + 'via_region_data.json', 'w') as outfile:
        json.dump(one_summarize, outfile)


labelme2one(os.path.join(dir_dataset, project+"/train/",))
labelme2one(os.path.join(dir_dataset, project+"/val/",))
