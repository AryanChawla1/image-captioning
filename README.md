# Image Captioning Project: Flickr8k Data

This project uses the Flickr8k dataset and the Hugging Face `datasets` library to prepare data for training an image captioning model.

The `prepare_data.py` script downloads the dataset, flattens the five captions per image into single image-caption pairs, and saves them into three optimized Parquet files.

## Setup and Installation

This project uses a dedicated Python virtual environment (`venv`) named `venv_ic` to manage dependencies.

### 1. Create and Activate Virtual Environment

You must first create and activate the environment to isolate project dependencies. Try to use Python 3.12. If you have errors with installing PyTorch with cua129, change the cua version or remove it entirely to use the CPU instead.

| **Platform**             | **Command to Create**     | **Command to Activate**          |
| :----------------------- | :------------------------ | :------------------------------- |
| **Mac/Linux**            | `python3 -m venv venv_ic` | `source venv_ic/bin/activate`    |
| **Windows (CMD)**        | `python -m venv venv_ic`  | `venv_ic\Scripts\activate.bat`   |
| **Windows (PowerShell)** | `python -m venv venv_ic`  | `.\venv_ic\Scripts\Activate.ps1` |

To deactivate the environment when finished: `deactivate`

### 2. Install Dependencies

After activating the environment, install the required libraries (like `datasets` and `pandas`).

`pip install -r requirements.txt`

To update the dependencies list after installing new libraries:

`pip freeze > requirements.txt`

### 3. Run Data Preparation

Once dependencies are installed, run the data preparation script.

`python prepare_data.py`

This command will:

1. Download the `jxie/flickr8k` dataset (if not already cached).

2. Process the `train`, `validation`, and `test` splits.

3. Create a directory named `data/`.

4. Save the three finalized, flattened dataset files as `flickr8k_train.parquet`, `flickr8k_validation.parquet`, and `flickr8k_test.parquet` inside the `processed_data/` directory.

### 4. Generate embeddings

After the Parquet files are created, run `python generate_embeddings.py` file to create embeddings.

It should result in three new files: `flickr8k_train_embedded.parquet`, `flickr8k_validation_embedded.parquet`, and `flickr8k_test_embedded.parquet`.

These files contain 4 columns: original image, original caption, image embeddings, caption embeddings.

These embeddings WILL be added in the .gitignore files.

## Tuning and stuff

Run the file `python train_captioner.py` to train the model.

We load the embedded dataset and create a projection layer between the embedded data and the model's input.

Then we train the model and save it at the end.

## Model Evaluation

Run the file `python generate_captions.py` to use the model to generate image captions from the test dataset. The captions are saved to generated_captions.csv.

Run the file `python evaluate_captions.py` to compare the generated captions with the test dataset captions using BLEU and CIDEr metrics.

## Extras

### check_scripts folder

This folder contains short python scripts to check if things are working.

`check_pytorch.py` checks if you can use GPU training over CPU

`check_embeddings.py` prints the first few rows of embeddings from the generated Parquet files
