## JPEG-Compression Artifact Removal

------

Image JPEG-Compression Artifact Removal using Deep Convolutional Neural Networks.

### Install Required Packages

------

First ensure that you have installed the following required packages:

- tensorflow
- keras
- opencv

Some tool packages you also need:

```shell
pip install json,importlib,argparse
```

See requirements.txt for details.

### Folder Structure

------

```
├── main.py             - that is responsible for the whole pipeline.
│
│
├── base                - this folder contains the abstract classes of the project components
│   ├── base_data_loader.py   - this file contains the abstract class of the data loader.
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains the models of your project.
│   └── sr_model.py
│
│
├── trainer             - this folder contains the trainers of your project.
│   └── sr_trainer.py
│
|
├── data_loader         - this folder contains the data loaders of your project.
│   └── sr_data_loader.py
│
│
├── configs             - this folder contains the experiment configs of your project.
│   └── sr_config.json
│
│
└── utils
     └── utils.py       - util functions for parsing arguments.
```

### Dataset

------

Download high resolution images for training. 

Here I used images from DIV2K - train - HR dataset as label.

No need for downloading the low resolution images as data, `sr_data_loader.py` will add JPEG compression to the label and generate the training data. 

### Getting Started

------

- Download your dataset and change the data path in `sr_config.json`

- You can change the training parameters and image process parameters (like JPEG-compression value) if you like.
- Start the training using:

```shell
python main.py -c ./config/sr_config.json
```

- Check the processed data and label in `./log/chache/`
- Start Tensorboard visualization using:

```shell
tensorboard --logdir=./log/ --port=1234
```

### Demo

------

