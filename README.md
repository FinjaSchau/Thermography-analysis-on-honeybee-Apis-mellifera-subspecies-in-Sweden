# Thermography-analysis-on-honeybee-Apis-mellifera-subspecies-in-Sweden



This file explains in which order the python files need to be executed and how.

## Prerequisites
The program language python needs to be installed.
Further, the file *functions.py* needs to be saved in the same directory as *preparation.py* and *segmentation.py*.
In addition, the folders *Orig_Images_Year1_UA* and *Orig_Images_Year2_UA* (get the folder from LINK) need to be present with 5 subfolders each (months).

Before being able to execute the files, all packages need to be installed that are mentioned in the Import sections.

## Execution 

1. *preparation.py* by using the following commands:

- Year 1: `python CODE/python/preparation.py --yr 1 --basedir IMAGES/Orig_Images_Year1_UA --basedir_raw IMAGES/Raw_Images_Year1 --basedir_crop IMAGES/Cropped_Images_UA_Year1`

- Year 2: `python CODE/python/preparation.py --yr 2 --basedir IMAGES/Orig_Images_Year2_UA --basedir_raw IMAGES/Raw_Images_Year2 --basedir_crop IMAGES/Cropped_Images_UA_Year2`

Description of the argparse arguments:
- `--yr`= Year (1 or 2)
- `--basedir`= path to the original images
- `--basedir_raw`= path into which the cropped raw images should be saved
- `--basedir_crop`= path into which the cropped .tif images should be saved

2. *segmentation.py* by using the following commands:

- Year 1: `python CODE/python/segmentation.py --yr 1 --num 5 --basedir_raw IMAGES/Raw_Images_Year1 --basedir_cross IMAGES/NoCross_Images_Year1 --savefolder IMAGES/Segmentation_Images_Year1`

- Year 2: `python CODE/python/segmentation.py --yr 2 --num 5 --basedir_raw IMAGES/Raw_Images_Year2 --basedir_cross IMAGES/NoCross_Images_Year2 --savefolder IMAGES/Segmentation_Images_Year2`

Description of the argparse arguments:
- `--yr`= Year (1 or 2)
- `--basedir_raw`= path to the raw images
- `--basedir_cross`= path into which the .tif images without the cross should be saved
- --savefolder`= path into which the segmented .tif images should be saved
