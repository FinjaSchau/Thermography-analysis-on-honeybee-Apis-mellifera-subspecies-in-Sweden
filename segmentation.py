#!/usr/bin/env python3
#python CODE/python/segmentation.py --yr 2 --num 5 --basedir_raw IMAGES/Raw_Images_Year2 --basedir_cross IMAGES/NoCross_Images_Year2 --savefolder IMAGES/Segmentation_Images_Year2
#python CODE/python/segmentation.py --yr 1 --num 5 --basedir_raw IMAGES/Raw_Images_Year1 --basedir_cross IMAGES/NoCross_Images_Year1 --savefolder IMAGES/Segmentation_Images_Year1

import functions as fun
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from PIL import Image
import glob
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import os
from math import ceil, floor
from pandas import DataFrame
from sklearn.cluster import KMeans
import sys
import argparse
from statistics import mean
from collections import OrderedDict
import xlsxwriter

#mpl.rc("image", cmap = "gray")

#define inputs from the console
parser = fun.MyParser(description='Process some input values.')
parser.add_argument('--yr', metavar='Year', type=int, nargs=1,
                    help='an integer "Year" representing the season (0,1,2...)')
parser.add_argument('--num', metavar='NUM_SEGMENTATIONS', type=int, nargs=1,
                    help='an integer "NUM_SEGMENTATIONS" for the number of segmentations')
parser.add_argument('--basedir_raw', metavar='PATH_RAW', type=str, nargs=1,
                    help='path "PATH_RAW" to folder with raw images')                
parser.add_argument('--basedir_cross', metavar='PATH_NOCROSS', type=str, nargs=1,
                    help='path "PATH_NOCROSS" to folder for NoCross images')      
parser.add_argument('--savefolder', metavar='PATH_SEGMENTATION', type=str, nargs=1,
                    help='path "PATH_SEGMENTATION" to folder for the segmented images')                                    
args = parser.parse_args()

#get inputs from the console
year = args.yr[0] 
num = args.num[0] #number of segmentations
basedir_raw = args.basedir_raw[0]
basedir_cross = args.basedir_cross[0]
save_folder = args.savefolder[0]

#delete directories if they already exist --> not possible to overwrite existing directories...
if os.path.exists(f'{basedir_cross}'):
    shutil.rmtree(f'{basedir_cross}', onerror=fun.readonly_handler)
    
if os.path.exists(f"{save_folder}"):
    shutil.rmtree(f'{save_folder}', onerror=fun.readonly_handler)
    
#create new directories for the images to be saved in
shutil.copytree(f'{basedir_raw}',
                f'{basedir_cross}',
                ignore=fun.ignore_files)
                
shutil.copytree(f'{basedir_raw}',
                f'{save_folder}',
                ignore=fun.ignore_files)

#crop image settings
shift_left  = 60
shift_right = 65
shift_top   = 64
shift_end   = 17


#####################Preparation -> get all file names and ignore non-hive images
camera_pattern_len = len('FLIR00733_')
extension_len = len('.raw')
sides = ['Back', 'Front', 'Left', 'Right']
to_filter = {'Space9_AFBismissing', 'Final_Final', 'final', 'FinalFinal', 'Final', 'finalfinal', 'final_final', 'Space7', 'Space7b', 'Space8', 'Space10', 'Space16', 'Space14', 'Space14a', 'Space14b', 'Sapce14', 'Space1', 'Space11', 'Space12', 'Space13', 'Space15', 'Space2', 'Space3', 'Space4', 'Space5', 'Space6', 'Space9'}
#scan_iteration_folders = [p for p in glob.glob(f'{basedir_raw}/**') if os.path.isdir(p)]
#print(scan_iteration_folders)
scan_identifier = {Path(p).stem[camera_pattern_len:] for p in glob.glob(f'{basedir_raw}/**/*.raw') if len(Path(p).stem) >= camera_pattern_len}
scan_identifier = scan_identifier.difference(to_filter)
#print(scan_identifier)
specifier = {'_'.join(i.split('_')[1:-1]) for i in scan_identifier}
#print(specifier)
scans = {s:{si:list() for si in sides} for s in specifier}
print(specifier)
for p in glob.glob(f'{basedir_raw}/**/*.raw'):
    stem = Path(p).stem[camera_pattern_len:]
    components = stem.split('_')
    spec = '_'.join(components[1:-1])
    side = components[-1]
    if spec in specifier and side in sides:
        scans[spec][side].append(p)
scans = OrderedDict(sorted(scans.items()))
scans.pop("")
#num_scans = {s:{si:len(l) for si,l in scans[s].items()} for s in scans}

l = sorted(list(fun.getList(scans)))

#bee_centroids = [[[None for i in range(5)] for j in range(len(l))] for sideidx in range(len(sides))]

#header_data = ["Locality", "Side", "Subspecies", "Name", "Class", "Year", "Month", "x_value", "y_value"]
#print(bee_centroids)
########################find and fill the cross, segmentation######################################
for sideidx, s in enumerate(sides):
    #workbook = xlsxwriter.Workbook(f"Centroids_{s}_seg{num}_year{year}.xlsx")
    #worksheet = workbook.add_worksheet("Centroids")
    #for col_num, data in enumerate(header_data):
    #    worksheet.write(0, col_num, data)
    #row = 1
    #col = 0
    for j in range(0,len(l)):
        spec = l[j]
        for i in range(0,5):
            n_back = len(scans[str(f"{spec}")][s])
            if i<n_back:
                path = scans[str(f"{spec}")][s][i]
                I = np.fromfile(f"{path}", dtype=np.uint16).reshape((175,150))
                
                cross, I = fun.fill_cross(I)
                
                #remove handle
                handleTop = cross[0] - 55
                handleButtom = cross[0] - 25
                handleLeft = cross[1] - 25
                handleRight = cross[1] + 25
                
                I[handleTop:handleButtom, handleLeft:handleRight] = 0

                img = Image.fromarray(I)
                
                save_path = f"{basedir_cross}/"+ Path(path).parent.name+"/"+Path(path).stem
                img.save(Path(save_path).with_suffix('.tif'))

                #exclude top and bottom
                cutTop   = cross[0] - shift_top
                cutEnd   = cross[0] + shift_end

                #exclude left and right
                cutLeft  = cross[1] - shift_left
                cutRight = cross[1] + shift_right

                img = img.crop((cutLeft, cutTop, cutRight, cutEnd)) 

                I = np.array(img)
                
                wanted_values = np.linspace(0,1,num)
                uniques = np.unique(I)

                classifier = KMeans(n_clusters=num, random_state = 50).fit(uniques[...,None])
                #classes = classifier.predict(uniques[...,None])

                value_class = classifier.predict(np.linspace(0,2**16, 2**16)[:,None])
                
                boundaries = [0,]
                for k in range(num-1):
                    boundaries.append(np.nonzero(np.convolve(value_class, np.array([1,-1]), "valid"))[0][k]+1)
                boundaries.append(2**16)
                #print(boundaries)

                boundaries = [0, ]
                for i in range(2**16-2):
                    if value_class[i] != value_class[i+1]:
                        boundaries.append(i+1)
                boundaries.append(2**16)
                boundaries = np.array(boundaries)
                print(boundaries)             
                pairs = np.stack([uniques, classes], axis=-1)
                centroids  = classifier.cluster_centers_ 
                boundaries = np.array(sorted(np.round(centroids).flatten())).astype("int")
 
                segments = fun.get_segmentation(I, boundaries)
                
                fused = fun.fuse_segments(segments, wanted_values)

                #exclude top and bottom
                cutTop   = cross[0] - shift_top
                cutEnd   = cross[0] + shift_end
                fused[1:cutTop-1,:] = (wanted_values[0]+ wanted_values[1])/2
                fused[cutEnd+1:,:] = (wanted_values[0]+ wanted_values[1])/2

                #exclude left and right
                cutLeft  = cross[1] - shift_left
                cutRight = cross[1] + shift_right
                fused[:,1:cutLeft-1] = (wanted_values[0]+ wanted_values[1])/2
                fused[:,cutRight+1:] = (wanted_values[0]+ wanted_values[1])/2
                
                #handle
                handleTop = cross[0] - 55
                handleButtom = cross[0] - 25
                handleLeft = cross[1] - 25
                handleRight = cross[1] + 25
                
                fused[handleTop:handleButtom, handleLeft:handleRight] = (wanted_values[1]+ wanted_values[2])/2
                
                bees = segments[-1,...] #get last segment
                
                bees_pixels = np.stack(np.nonzero(bees), axis = -1)

                bees_centroid = np.mean(bees_pixels, axis = 0)

                bee_centroids[sideidx][j][i] = bees_centroid
                
                img = Image.fromarray(fused)
                #draw = ImageDraw.Draw(img)
                #draw.point((bees_centroid[1], bees_centroid[0]), fill = "red")
                
                save_path = f"{save_folder}/"+ Path(path).parent.name+"/"+Path(path).stem
                img.save(Path(save_path).with_suffix('.tif'))
                
                
    #             spec_name = spec.split("_")
    #             name = spec_name[0]
    #             species = spec_name[1]
                
    #             if year == 1:
    #                 if i == 0:
    #                     month = "November"
    #                     yr = "2020"
    #                 if i == 1:
    #                     month = "December"
    #                     yr = "2020"
    #                 if i == 2:
    #                     month = "January"
    #                     yr = "2021"
    #                 if i == 3:
    #                     month = "February"
    #                     yr = "2021"
    #                 if i == 4:
    #                     month = "March"
    #                     yr = "2021"

                        
    #             if year == 2:
    #                 if i == 0:
    #                     month = "November"
    #                     yr = "2021"
    #                 elif i == 1:
    #                     month = "December"
    #                     yr = "2021"
    #                 elif i == 2:
    #                     month = "January"
    #                     yr = "2022"
    #                 elif i == 3:
    #                     month = "February"
    #                     yr = "2022"
    #                 elif i == 4:
    #                     month = "March"
    #                     yr = "2022"

                        
    #             worksheet.write(row, col + 1, s)
    #             worksheet.write(row, col + 2, species)
    #             worksheet.write(row, col + 3, name)
    #             worksheet.write(row, col + 5, yr)
    #             worksheet.write(row, col + 6, month)
    #             worksheet.write(row, col + 7, bees_centroid[1])
    #             worksheet.write(row, col + 8, bees_centroid[0])
    #             row += 1
      
    # workbook.close()
#print(bee_centroids)   
################### create overview plots#########################################

scans = {s:{si:list() for si in sides} for s in specifier}

for p in glob.glob(f'{save_folder}/**/*.tif'):
    stem = Path(p).stem[camera_pattern_len:]
    components = stem.split('_')
    spec = '_'.join(components[1:-1])
    side = components[-1]
    if spec in specifier and side in sides:
        scans[spec][side].append(p)
scans = OrderedDict(sorted(scans.items()))  
#num_scans = {s:{si:len(l) for si,l in scans[s].items()} for s in scans}
scans.pop("")
l = sorted(list(fun.getList(scans)))

for sideidx, s in enumerate(sides):
    fig, axs = plt.subplots(16, 5, constrained_layout=False, figsize = (12,32))
    gridspec = axs[0,0].get_subplotspec().get_gridspec()
    fig.suptitle(str(f"{s}"), y = 0.90)
    for j in range(0,len(l)):
        spec = l[j]
        axs[j,0].set_title(str(f"{spec}"), color = "black", y=-12, fontsize="10")
        for i in range(0,5):
            n_back = len(scans[str(f"{spec}")][s])
            path = scans[str(f"{spec}")][s][i] if i<n_back else None
            axs[j,i].axis('off')
            if not path is None:
                colidx = int(Path(path).parts[-2][0])-1
                I = np.array(Image.open(path))
                axs[j,colidx].matshow(I, vmin = 0, vmax = 1)
                bee_centroid = bee_centroids[sideidx][j][i]
                if not bee_centroid is None:
                    axs[j, colidx].scatter([bee_centroid[1], ], [bee_centroid[0], ], marker = "x", color = "r")                
                p = Path(path)
                axs[j,colidx].annotate(p.parent.name,(0, 0))
    path = Path(save_folder).parent.name+f"/{s}_seg{num}_Year{year}_x_UA.pdf"
    if os.path.exists(path):
        os.remove(path)
        fig.savefig(path)
    else:
        fig.savefig(path)
    
