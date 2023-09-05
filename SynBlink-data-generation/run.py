import glob
import os
import subprocess
import time
import threading
import datetime
import random


fbx_model_folder = r"**/Fbx_model_path" # Replace this folder to Fbx model path
fbx_model_paths = glob.glob("{}/*/*.Fbx".format(fbx_model_folder))

output_folder = r"**/SynBlink-50K" # Replace this folder to output path

# createfolder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

os.system("chcp 65001") # Support Chinese

imgs_per_model = 500

cmds = []

for i, fbx_path in enumerate(fbx_model_paths):
    
    print(fbx_path)
    print("---"*10)
    print("{}/{}".format(i+1, len(fbx_model_paths)))

    id = fbx_path.split("\\")[-1].split(".")[0]
    blend_env = "env.blend"
    sex = "F" if "Female" in fbx_path else "M"

    for j in range(imgs_per_model):
        blink = random.choice(["BL", "NB"]) # blink or no blink
        background = random.choice(["IMB", "HDB", "TEB"]) # COCO image / HDR image / texture image
        light = random.choice(["LL", "ML", "HL"]) # low light / middle light / high light

        output_path = r"{}/{}_{}_{}_{}_{}_{}/".format(output_folder, sex, str(id).zfill(2), str(j).zfill(4), blink, background, light)
        cmd = "blender -b {} --python blender_scripts.py -- {} {}".format(blend_env, fbx_path, output_path)

        p = subprocess.call(cmd)
    
    
        