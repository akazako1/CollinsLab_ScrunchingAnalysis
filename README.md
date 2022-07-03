# CollinsLab_ScrunchingAnalysis

### Files in this repository:
* `feature_extraction/main.py`: main script to perform feature extraction
* `datasets.py`: compiles images into numpy file to feed into main.py
* `image_cropping`: folder containing image_cropping script for cropping wells from plate images
* `read_input.py`: reads in images to feed into datasets.py for compilation or directly into main.py if desired
* `filtering.py`: image processing for feature extraction
* `createAVI.py`: script for creating .avi movies from the raw image sequence; simplifies the process of manual srunching scoring
* `get_well_data.py`: given the image sequence, generates the following files for the respective well: 1) MAL over time (.csv), 2) COM over time (.csv), 3) Aspect Ratio over time (.csv), 4) MAL vs time plot. 5) AVI movies showing the binarized image of the moving worm 
*  `main_peak_analysis.py`: given the files generated in generates .txt files with data that can be used to classify scrunching 
* `rm_background.py`: # This file provides functionality for binarizing images/removing the background 


### Running the scripts
1. Create folders with individual wells by running `crop_wells.py`. 
   -  Set `plateFolder` variable to the folder path with image sequence for a particular plate.
   -  If you only want to analyze specific wells, change the `wells` variable
   -  Refer to the script for more guidance on how to do that

2. To generate `.csv` files with frame-by-frame data for individual wells run  `get_well_data.py`
   - Refer to the script to adjust parameters/specify wells 
3. Run `main_peak_analysis` to perform classification. 
    -  Set `plateFolder` variable to the folder path with image sequence for a particular plate.
    -  Note that you have to have the following files (that you should've generated in step 2). Replace `X` in the path with the number of the well you want to analyze
      1)  `~/yourfoldername/results/well_data/MAL_wellX.csv`
      2)  `~/yourfoldername/results/well_data/COM_wellX.csv`
      3)  `~/yourfoldername/results/well_data/AspRatio_wellX.csv`



### Creating AVI movies 
- If you want to generate movies for individual wells, make sure to run the `crop_wells.py` script first. 
- Run `createAVI.py` script by typing `python3 createAVI.py` into the terminal. Then, follow the terminal prompts.