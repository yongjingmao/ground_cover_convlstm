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
Figure below shows the structure of the model in this study. For each input sequence with the shape of 128 × 128 × 6 × 16, along the channel dimension, it was separated to ground cover feature, mask feature and auxiliary feature. The output sequence only contains the ground cover layer. Along the temporal dimension, it was divided into context sequence (first 8 seasons) and target sequence (last 8 seasons). 

The input context sequence was directly fed into the memory cells (i.e. ConvLSTM or ST-LSTM) to produce the output with a 2D convolutional layer and pass information into the next memory cell. For the input target sequence, the input data were not directly fed into memory cells. Instead, there were used in combination with the predicted ground cover in the previous step according to the resampling method, which randomly sampled output and input ground cover (including mask) data, concatenated the auxiliary data and produced a new input to feed into the memory cells. In this step, a simulated mask with zero gap was generated for all the predicted ground cover to be consistent with the input data. There are two sampling methods used in this study. The first method is for standard sequence prediction of recurrent neural network, which purely used prediction result from the previous step to predict the next step and the true previous step is unknown, i.e. non-sampling. This non-sampling method is used for original ConvLSTM. Another sampling method is the Reverse Scheduled Sampling (RSS), which randomly hiding real observations with decreasing probabilities as the training progresses to force the model to learn more about long-term dynamics. For training, the RSS method was used for PredRNN. For forecasting and evaluation, non-sampling was used for both ConvLSTM and PredRNN.

Finally, the overlapping time steps (i.e. season 2~16) in output and input sequences were used to calculate the loss function. The Mean absolute error (MSE)  was used to quantify training loss while the ensemble of MSE, the Huber Loss and the Structural Similarity Index (SSIM) was used for the validation step. As MSE and Huber Loss indicates better performance with small values, SSIM is just the opposite, so 1-SSIM was used to calculate the ensemble loss. 

![Alt text](Figures/modelstructure.png?raw=true "Model structure")

## Show Cases
![Alt text](Figures/PredRNN_rs/images.jpg?raw=true "Model performance for a single data record" )
*Context, target and predicted images for a sampled testing record. The first row is the input context sequence, the second row is the target sequence, the third row is the predicted target images and the last row is the MAE between target and prediction.*

![Alt text](Figures/PredRNN_rs/timeseries.jpg?raw=true "Time series (all data records) of model performance (The next season prediction)")
*Time series (all data records) of model performance (The next season prediction). The top panel shows the time series of observed and predicted ground cover. Grey bars shows the cloud cover and the black vertical line indicates when the new ground cover mapping method was introduced. The bottom panel shows the calculated MAE and SSIM for each time step.*

## Get Started
1. Setup environment
```
conda create -n convlstm python=3.9
pip install env.txt
```
2. Parameter configeration. 
- An example of generating configeration files for PredRNN\
(wd: work directory;
pd: path of data;
mt: model type;
ts: training sampling method; 
iw: image width; 
ih: image height; 
nl: number of layer; 
hc: number of hidden channel; 
mc: number of mask channel; 
oc: number of output channel; 
bs: batch size; 
e: epoch; 
ft: number of future steps for training; 
ct: number of context steps for training; 
fv: number of future steps for validation)

```
python config/config.py -wd .. -pd ../Data/GC -mt PredRNN 
-ts rs -iw 128 -ih 128 -nl 3 -hc 64 
-mc 1 -oc 1 -ic 6 -bs 4 -e 1000     
-ft 8 -ct 8 -fv 8
```

- More parameters to config can be found with

```
python config/config.py -h
```

3. Train model with sample data\
(wd: work directory)
```
python scripts/model_train.py -wd ..
```
4. Use model for prediction\
The trained ConvLSTM and PredRNN can be downloaded from [CloudStor](https://cloudstor.aarnet.edu.au/plus/s/UvG1xAWvU4HOXM0)
(wd: work directory; md: model directory)
```
python scripts/model_pred.py -wd .. -md PATH_TO_TRAINED_MODEL
``` 

