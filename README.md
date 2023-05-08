# Thermography-analysis-on-honeybee-Apis-mellifera-subspecies-in-Sweden

This file explains in which order the python files need to be executed and how.

# Prerequisites
The program language python needs to be installed.
Further, the file function.py needs to be saved in the same directory as preparation.py and segmentation.py.
In addition, the folders Orig_Images_Year1_UA and Orig_Images_Year2_UA (get the folder from LINK) need to be present with 5 subfolders each (months).

Before being able to execute the files, all packages need to be installed that are mentioned in the Import sections.

# Execution 

1. preparation.py by using the following command:

- Year 1: `python CODE/python/preparation.py --yr 1 --basedir IMAGES/Orig_Images_Year1_UA --basedir_raw IMAGES/Raw_Images_Year1 --basedir_cross IMAGES/NoCross_Images_Year1`

- Year 2: `python CODE/python/preparation.py --yr 2 --basedir IMAGES/Orig_Images_Year2_UA --basedir_raw IMAGES/Raw_Images_Year2 --basedir_cross IMAGES/NoCross_Images_Year2`
