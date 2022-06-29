# CollinsLab_ScrunchingAnalysis

### Files in this repository:
* `feature_extraction/main.py`: main script to perform feature extraction
* `datasets.py`: compiles images into numpy file to feed into main.py
* `image_cropping`: folder containing image_cropping script for cropping wells from plate images
* `read_input.py`: reads in images to feed into datasets.py for compilation or directly into main.py if desired
* `filtering.py`: image processing for feature extraction
* `createAVI.py`: script for creating .avi movies from the raw image sequence; simplifies the process of manual srunching scoring
* `main_peak_analysis.py`: generates .txt files with data that can be used for further
* `get_well_data.py`


### Running the scripts
1. Create folders with individual wells by running `crop_wells.py`. 
   -  Set `plateFolder` variable to the folder path with image sequence for a particular plate.
   -  If you only want to analyze specific wells, change the `wells` variable
   -  Refer to the script for more guidance on how to do that

2. To generate `.csv` files with frame-by-frame data for individual wells run  `get_well_data.py`
   - Refer to the script to adjust parameters/specify wells 



### Creating AVI movies 
- If you want to generate movies for individual wells, make sure to run the `crop_wells.py` script first. 
- Run `createAVI.py` script by typing `python3 createAVI.py` into the terminal. Then, follow the terminal prompts.