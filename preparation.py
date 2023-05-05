import functions as fun
#from functions import *
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from pathlib import Path
from PIL import Image, ImageOps
import os
from math import ceil, floor
from pandas import DataFrame
from sklearn.cluster import KMeans
import sys
import argparse


parser = fun.MyParser(description='Process some input values.')
parser.add_argument('--basedir', metavar='PATH_ORIGINAL', type=str, nargs=1,
                    help='path "PATH_ORIGINAL" to original pictures')
parser.add_argument('--basedir_raw', metavar='PATH_RAW', type=str, nargs=1,
                    help='path "PATH_RAW" for cropped .raw images')
parser.add_argument('--basedir_crop', metavar='PATH_CROP', type=str, nargs=1,
                    help='path "PATH_CROP" for cropped .tif images')
args = parser.parse_args()

basedir = args.basedir[0]
basedir_raw = args.basedir_raw[0]
basedir_crop = args.basedir_crop[0]

if os.path.exists(f'{basedir_raw}'):
    shutil.rmtree(f'{basedir_raw}', onerror=fun.readonly_handler)
    
if os.path.exists(f"{basedir_crop}"):
    shutil.rmtree(f'{basedir_crop}', onerror=fun.readonly_handler)


# calling the shutil.copytree() method and
# passing the src,dst,and ignore parameter
shutil.copytree(f'{basedir}',
                f'{basedir_raw}',
                ignore=fun.ignore_files)

shutil.copytree(f'{basedir}',
                f'{basedir_crop}',
                ignore=fun.ignore_files)

#create cropped raw images
paths = glob.glob(f"{basedir}/*/*.bmp")
for path in paths:
    I = fun.load_temp_bmp(path)
    raw_path = f"{basedir_raw}/"+Path(path).parent.name+"/"+Path(path).stem+ '.raw'
    I.tofile(raw_path)

#Crop Images in .tif format
paths = glob.glob(f"{basedir}/*/*.bmp")

for path in paths:
    I = fun.load_temp_bmp(path)
    img = Image.fromarray(I)
    crop_path = f"{basedir_crop}/"+Path(path).parent.name+"/"+Path(path).stem
    img.save(Path(crop_path).with_suffix('.tif'))

# create overwiew plots
camera_pattern_len = len('FLIR00733_')
extension_len = len('.bmp')
sides = ['Back', 'Front', 'Left', 'Right']
to_filter = {'Space9_AFBismissing', 'Final_Final', 'final', 'FinalFinal', 'Final', 'finalfinal', 'final_final', 'Space7', 'Space7b', 'Space8', 'Space10', 'Space16', 'Space14', 'Space14a', 'Space14b', 'Sapce14', 'Space1', 'Space11', 'Space12', 'Space13', 'Space15', 'Space2', 'Space3', 'Space4', 'Space5', 'Space6', 'Space9', "Space01", "Space02", "Space03", "Space04", "Space05", "Space06", "Space07", "Space08", "Space09", "FLIR01477", "FLIR01478"}
#scan_iteration_folders = [p for p in glob.glob(f'{basedir}/**') if os.path.isdir(p)]
#print(f'{len(scan_iteration_folders)=}')
scan_identifier = {Path(p).stem[camera_pattern_len:] for p in glob.glob(f'{basedir}/**/*.bmp') if len(Path(p).stem) >= camera_pattern_len}
scan_identifier = scan_identifier.difference(to_filter)

specifier = {'_'.join(i.split('_')[1:-1]) for i in scan_identifier}
print(specifier)
scans = {s:{si:list() for si in sides} for s in specifier}
#scans.pop("")
#print(scans)
for p in glob.glob(f'{basedir}/**/*.bmp'):
    stem = Path(p).stem[camera_pattern_len:]
    components = stem.split('_')
    spec = '_'.join(components[1:-1])
    side = components[-1]
    if spec in specifier and side in sides:
        scans[spec][side].append(p)
        
#num_scans = {s:{si:len(l) for si,l in scans[s].items()} for s in scans}

for s in sides:
    l = list(fun.getList(scans))
    fig, axs = plt.subplots(16, 5, constrained_layout=False, figsize = (12,32))
    gridspec = axs[0,0].get_subplotspec().get_gridspec()
    fig.suptitle(str(f"{s}"), y = 0.90)
    for j in range(0,len(l)):
        spec = l[j]
        axs[j,0].title.set_text(str(f"{spec}"))
        for i in range(0,5):
            n_back = len(scans[str(f"{spec}")][s])
            path = scans[str(f"{spec}")][s][i] if i<n_back else None
            axs[j,i].axis('off')
            if not path is None:
                colidx = int(Path(path).parts[-2][0])-1
                axs[j,colidx].matshow(fun.load_temp_bmp(scans[str(f"{spec}")][s][i]))
                p = Path(scans[str(f"{spec}")][s][i])
                axs[j,colidx].annotate(p.parent.name,(0, 0))
    path = Path(basedir_raw).parent.name+f"/{s}_Year2_UA.pdf"
    if os.path.exists(path):
        os.remove(path)
        fig.savefig(path)
    else:
        fig.savefig(path)
    
