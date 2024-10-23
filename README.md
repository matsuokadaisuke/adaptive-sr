# adaptive-sr
Adaptive super-resolution for ocean bathymetric maps using a deep neural network and data augmentation
## Installation
```
pip install -r requirements.txt
```

## sr_dataset_builder

### Data preparation
Put original data in `sr_dataset_builder/input`  
Data format: `Pickle`

### main.py
Build a randomly-block-separated bathymetric chart dataset
```
python main.py
```

### bathymetric_map_feature.py
Calculate various features for bathymetric map data and save them as CSV
```
python bathymetric_map_feature.py [bathymetric_map_data(.pkl)]
```

### resize.py
Resize the image
```
python resize.py [config_file(.yml)]
```

### flip.py
Generate inverted images
```
python flip.py [config_file(.yml)]
```

### rotate.py
Generate rotated images
```
python rotate.py [config_file(.yml)]
```

### scale.py
Generate horizontal and vertical scaled images
```
python scale.py [config_file(.yml)]
```

### mixup.py
Generate images with mixup applied
```
python mixup.py [config_file(.yml)]
```

## sr_trainer
### Data preparation
Put separated datasets in `sr_trainer/data/{train, test, validation}`  

### main.py
Script for training and test execution
```
python main.py yml_fname {train,test}

positional arguments:
  yml_fname     filename including input parameters (.yml)
  {train,test}  execution mode: train or test
```

## Citation
Murakami, K., D. Matsuoka, N. Takatsuki, M. Hidaka, J. Kaneko, Y. Kido and E. Kikawa (20XX) Adaptive super-resolution for ocean bathymetric maps using deep neural network and data augmentation. *Earth and Space Science*, XXX
