# Time-lapse 3D Cell Analysis
This repository contains the source code for the paper "Deep Learning Enabled Time-Lapse 3D Cell Analysis", where we introduce an end-to-end analysis method for boundary tagged 3D cell images.

<a>
    <img src="workflow.png" width="100% height="70%"/>
</a>

This whole cell analysis mainly contains three steps: 3D cell segmentation, cellular/sub-cellular feature extraction, and cell tracking. These three components can be used together or separately.
                                                    
## Code 
### Installation 
You need to install SimpleITK, pytorch, scikit-image, opencv, DenseInferenceWrapper  

## 3D Cell Segmentation
### Testing
#### Histogram Match (with PNAS and Purdue Data)
- python hist_match.py
- celldataset.cell_testing('/source/data/celldata/') contains the background subtracted images
- celldataset.cell_training('/source/data/PNAS/') takes the PNAS dataset for training
- results in hist_match folder (z interpolation is done 5 times)

#### Predict
- python predict.py
- user model_last.tar file
- generates a probability map in prob_map folder
- uses pytorch, torchvision and scikit image

#### Post Processing 
- python postprocessing.py
- Enter Seeds as 15 and the threshold for black voxels as 0.05
- Hyperparams are in main function
- result folder has the output

### Training
- Data PNAS: VRL Data on [google drive](https://drive.google.com/drive/folders/1RvZYdojQGWGE1V6su1pZOOvJ0g0wKZLy)
- python main.py
                                                   
## 3D Cellular and Sub-cellular feature extraction
- cd sub_celluar
- python feature.py
- Input should be the individual segmented 3D TIFF image stack                                                    
- This will compute features include Cell volumn (in voxels), neighboring (Adjacent) Cells, 3D cell surface, three cell wall junction points, cell center, anticlinal wall segments
                                                    
## 3D Cellular and Sub-cellular feature extraction 
- cd Tracking
- python track.py
- Input should be a sequence of segmented TIFF 3D image stacks. Input names should be T+3 digit number to denote time instances.
- It will output tracked TIFF files and a txt files containing all cell/nuclei tracks. The details of the format of txt file, please refer to [celltrackingchallenge](http://celltrackingchallenge.net/)                                                    
                                                    
