# Ground cover forecasting

We implemented both [ConvLSTM](https://github.com/rudolfwilliam/satellite_image_forecasting) and [PredRNN](https://github.com/thuml/predrnn-pytorch) to predict future ground cover based on previous trends. Auxiliary data includes rainfall, temperature, 
soil moisture and runoff. 

We use [PyTorch](https://pytorch.org/) for the implementation.

## Data explanation
Datasets in this study were grouped into three categories as (1) target feature, (2) mask and (3) auxiliary features. (1) Target feature is the feature to predict. So far it only includes the [seasonal ground cover](https://portal.tern.org.au/metadata/22022). It is the first band of the input image data and the only band in the output image; (2) Mask is the second band, which is also from the ground cover data to distinguish the valid ground cover and data gaps, it is the second band of the input image but not in the output image; (3) all the climate and water balance data including rainfall and temperature from [SILO](https://www.longpaddock.qld.gov.au/silo/gridded-data/) as well as soil moisture and runoff from [AWO](https://awo.bom.gov.au/about/overview/dataAccess) are auxiliary datasets, which are included in the input image but not in the output image. Bands 3 to 6 are the auxiliary datasets.

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

![Alt text](Figures/image.png?raw=true "Model structure")

## Show Cases
![Alt text](Figures/PredRNN_rs/all_images.jpg?raw=true "Model performance for a single data record" )
*Context, target and predicted images for a sampled testing record. The first row is the input context sequence, the second row is the target sequence, the third row is the predicted target images and the fourth row is the MAE between target and prediction, and the last row is the comparison of histgrams.*

![Alt text](Figures/PredRNN_rs/timeseries.jpg?raw=true "Time series (all data records) of model performance (The next season prediction)")
*Time series (all data records) of model performance (The next season prediction). The top panel shows the time series of observed and predicted ground cover. Grey bars shows the cloud cover and the black vertical line indicates when the new ground cover mapping method was introduced. The next two panels show the calculated MAE and SSIM as well as their decay rate for each time step.*

![Alt text](predicted_maps/Pred.jpg?raw=true "GBRCA prediction" )
*The prediction of next-season ground cover for the entire GBRCA. The plot (a) is the observation for the last season in the context sequence; the plot (b) is the prediction corresponding to (a); (c) is the prediction for the next season; and (d) is the difference between (c) and (b) as (c)-(b)*

![Alt text](predicted_maps/Timeseries.jpg?raw=true "GBRCA time series" )
*The time series of observation (the last 16 seasons) and prediction (the last 15 seasons + uncoming season). The first plot shows the spatial average of prediction (dash line) and observation (solid line). The bar chart in the first plot compares the increment of averaged ground cover. The second plot calculates the scoring metrics (MAE and SSIM) comparing observation and prediction for the last 15 seasons. The solid and dash liens are the spatial average of MAE and SSIM respectively. The error bar shows the spatial std for MAE*

## Get Started
(Only steps in ***bold_italic*** can be implemented with sample data only)
### Setup environment
```
conda create -n convlstm python=3.9
pip install env.txt
```
### Data retrieve
1. Update AWO (including soilmoisture and runoff) data
(wd: work directory where retreived data will be saved)
```
python scripts/retrieve_AWO.py -wd {}
```
2. Update SILO (including rainfall and temperature) data
(wd: work directory where retreived data will be saved)
```
python scripts/retrieve_SILO.py -wd {}
```
3. Preprocess Auxiliary data (including normalization and projection)
```
python scripts/preprocess_AUX.py -wd {} -SILO {} -AWO {}
```
The above processes were included in a batch job
```
qsub PBS_jobs/data_retrieve.sh
```
### Data preparation for model training
1. Set training data, slice and preprocess data for each site
(wd: work directory where training data will be saved in a folder named "Sites";
Site: index for site;
l: windlow length;
iw: image width; 
ih: image height; 
tr: train split ratio;
vtr: validation and test splitting ratio)
```
python scrips/set_train_data.py -wd {} --Site {} -l {} -iw {} -ih {} -tr {} -vtr {}
```
- More parameters to config can be found with
```
python scrips/set_train_data.py -h
```
2. Site iteration can be achieved by submitting batch jobs
```
python PBS_jobs/qsub_data_prep.py
```
### Model train
1. ***Parameter configeration.***
- An example of generating configeration files for PredRNN\
(wd: work directory where all files during training will be saved;
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
python config/config.py -wd {} -pd ../Data/GC -mt PredRNN 
-ts rs -iw 128 -ih 128 -nl 3 -hc 64 
-mc 1 -oc 1 -ic 6 -bs 4 -e 1000     
-ft 8 -ct 8 -fv 8
```

- More parameters to config can be found with

```
python config/config.py -h
```

2. ***Train model with sample data***
(wd: work directory, same as the one used in config)
```
python scripts/model_train.py -wd {} 
```

3. Train model with entire dataset with batch jobs
```
python PBS_jobs/qsub_model_train.py
```
### Use model for prediction
The trained ConvLSTM and PredRNN can be downloaded from [CloudStor](https://cloudstor.aarnet.edu.au/plus/s/0lKj2HwjD0BVcK3)
(wd: work directory, same as the one used in config; md: model directory)
1. ***Model prediction for sampled data in a single site***
```
python scripts/model_pred.py -wd {} -md {}
``` 
2. Use model to predict large area in tiles and mosaic results\
(tw: tile width; th: tile height)
```
python scripts/model_mosaic.py -wd predicted_maps -tw 1280 -th 1280
```
or use batch job (recommended)
```
qsub PBS_jobs grid_pred.sh
```

