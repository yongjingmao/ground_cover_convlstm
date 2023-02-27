# Ground cover forecasting

We implemented both [ConvLSTM](https://github.com/rudolfwilliam/satellite_image_forecasting) and [PredRNN](https://github.com/thuml/predrnn-pytorch) to predict future ground cover based on previous trends. Auxiliary data includes rainfall, temperature, 
soil moisture and runoff. 

We use [PyTorch](https://pytorch.org/) for the implementation.

## Data explanation
Datasets in this study were grouped into three categories as (1) target feature, (2) mask and (3) auxiliary features. (1) Target feature is the feature to predict. So far it only includes the [seasonal ground cover](https://portal.tern.org.au/metadata/22022). It is the first band of the input image data and the only band in the output image; (2) Mask is the second band, which is also from the ground cover data to distinguish the valid ground cover and data gaps, it is the second band of the input image but not in the output image; (3) all the climate and water balance data including rainfall, temperature, soil moisture and runoff data are auxiliary datasets, which are included in the input image but not in the output image. Bands 3 to 6 are the auxiliary datasets.

| Band | Data              | Category  | Data type |
| ---- | ----------------- | --------- | --------- |
| 1    | Ground cover      | Target    | Float     |
| 2    | Ground cover mask | Mask      | Binary    |
| 3    | Rainfall          | Auxiliary | Float     |
| 4    | Temperature       | Auxiliary | Float     |
| 5    | Soil moisture     | Auxiliary | Float     |
| 6    | Runoff            | Auxiliary | Float     |

For each site, all the input data were resampled to 128 × 128 pixels, and were finally concatenated to multiband images following the order in the table. While the gap in ground cover data can be indicated by the mask band, the data gap in auxiliary data were filled with the mean value at the corresponding time step. Finally, all the float bands were normalized respectively. After the above steps, for each site, an image with shape (height × width × band × seasons) of 128 × 128 × 6 × 138 can be obtained.
The resultant image was sliced along the temporal (season) domain with a window length of 16 seasons (4 years) and the step of 1 season. As a result, each slice has the shape of 128 × 128 × 6 × 16 and 123 slices can be obtained for training, validation and testing. In this study, the first 50% of slices were used for training, the next 25% for validation and the final 25% for testing.

Raw sample data were presented in the "Data/GC" folder with the "Train", "Test", "Val" subfolders containing the sliced training, testing and validation data. Each data record was saved in .npz format, each contains a "image" file (Ground cover and mask) and a "Auxiliary file". These raw data will then be procssed by the "data_preparation.py" in "Data" folder.

## Model structure

## Show Cases
## Get Started

