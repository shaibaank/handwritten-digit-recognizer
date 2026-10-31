# Handwritten Digit Recognizer

A convolutional neural network that classifies 28x28 grayscale handwritten digits from the Kaggle Digit Recognizer dataset.

## Setup and run

Run these commands from the project directory. The commands work on Windows PowerShell, macOS, and Linux with Python installed:

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the [Kaggle Digit Recognizer dataset](https://www.kaggle.com/c/digit-recognizer), then place both files in this project directory. You can also place them in a `data/` subdirectory.

### Run the notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `mnist_digit_recognizer.ipynb` in the browser, then select **Run > Run All Cells**. The notebook preprocesses the data, trains the models, evaluates validation accuracy, creates plots in `outputs/`, and saves `submission.csv` and `mnist_cnn.keras`.

### Run the Python script

To run the same CNN pipeline without opening Jupyter:

```bash
python mnist_cnn.py
```

The script saves its plots in `outputs/`, creates `submission.csv`, and saves the trained model as `mnist_cnn.keras`.

## Dataset

The project uses the [Kaggle Digit Recognizer dataset](https://www.kaggle.com/c/digit-recognizer):

- `train.csv`: labelled images with a `label` column and 784 pixel columns
- `test.csv`: unlabelled images with 784 pixel columns

## Approach

1. Load and scale pixel values to the range 0-1.
2. Reshape each image to 28x28x1.
3. Split the labelled data into stratified training and validation sets.
4. Compare a Logistic Regression baseline with a CNN.
5. Evaluate accuracy, classification report, confusion matrix, and misclassified digits.
6. Generate a Kaggle submission file and save the trained CNN.

## Results

Run the notebook to generate the comparison table in section 9:

| Model | Validation accuracy |
|---|---|
| Logistic Regression (baseline) | _ |
| CNN | _ |

## Project files

- `mnist_digit_recognizer.ipynb`: notebook containing the complete analysis and training workflow
- `mnist_cnn.py`: standalone CNN training and evaluation script
- `requirements.txt`: Python dependencies
- `train.csv`: training dataset, downloaded from Kaggle
- `test.csv`: test dataset, downloaded from Kaggle
- `outputs/`: generated plots and evaluation figures
