# Image Captioning Project

This project was created to test the perfomance of a image captioning pipeline using CLIP embedding and a fine-tuned DistilBART transformer model. To test the performance, the COCO and Flickr8k datasets from HuggingFace were used.

## Setup and Installation

This project uses a dedicated Python virtual environment (`venv`) named `venv_ic` to manage dependencies. To run the training and inference for this pipeline, please use the notebook files: `model_training.ipynb` and `inference.ipynb`.

### 1. Create and Activate Virtual Environment

You must first create and activate the environment to isolate project dependencies. Try to use Python 3.12.

| **Platform**             | **Command to Create**     | **Command to Activate**          |
| :----------------------- | :------------------------ | :------------------------------- |
| **Mac/Linux**            | `python3 -m venv venv_ic` | `source venv_ic/bin/activate`    |
| **Windows (CMD)**        | `python -m venv venv_ic`  | `venv_ic\Scripts\activate.bat`   |
| **Windows (PowerShell)** | `python -m venv venv_ic`  | `.\venv_ic\Scripts\Activate.ps1` |

To deactivate the environment when finished: `deactivate`

### 2. Install Dependencies

After activating the environment, install the required libraries (like `datasets` and `pandas`). The first cell in `model_training.ipynb` covers this.

`pip install -r requirements.txt`

To update the dependencies list after installing new libraries:

`pip freeze > requirements.txt`

It is highly reccomended to use a CUDA-compatible device for the training process. Alternatively, Google Colab can be used.

### 3. Optional - Download COCO Dataset

Download the COCO dataset if you would like to use it. The COCO dataset on HuggingFace doesn't attach the images, just the path to them. Because of this, the images have to be installed seperately.

```
!wget http://images.cocodataset.org/zips/train2014.zip
!wget http://images.cocodataset.org/zips/val2014.zip

!unzip train2014.zip -d coco
!unzip val2014.zip -d coco
```

## Training

To train the model, run the cells in `model_training.ipynb`. If you completed the previous steps, you can start from the 4th cell. Select the appropriate `DATASET_NAME`.

```
# DATASET_NAME = "jxie/flickr8k" # flickr8k
DATASET_NAME = "yerevann/coco-karpathy" # coco (make sure to download images beforehand)
```

1. The 4th cell will save the captions to `captions.json`.
2. The 5th cell embeds the images and save it to `patch_embed_float16.npy`.
3. The 6th cell splits the images for training.
4. The 7th cell loads the splits into a `Dataset` object.
5. The 8th cell creates the adapter layer from the embedded image vector to DistilBART.
6. The 9th cell trains the model and adapter layer weights. After training, the model is saved to `caption_model/`
7. The 10th cell tests the model by generating captions. The metrics used to measure them are BLEU and CIDEr.
8. The 11th cell shows the output using some test images.

## Inference

To use the model on images, use the `inference.ipynb` file. Make sure to have a trained model beforehand. Add the image paths to the second cell to generate the respective caption. This process involves loading the model, embedding the image, and then running it through the transformer to generate the caption. Depending on the complexity of the image, you can change the `max_length` to generate more tokens. However, we found 16 was appropriate.
