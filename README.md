# Building an ML Model from Scratch — Workshop

A beginner-friendly, hands-on workshop repository for learning the end-to-end machine learning workflow with Python.

## What you'll learn
- Understand features, labels, classification, and regression
- Load and explore a dataset
- Split data into training and test sets
- Train a model with scikit-learn
- Evaluate predictions using accuracy, a confusion matrix, and a classification report
- Implement a tiny K-Nearest Neighbors classifier with NumPy to understand the algorithm

> In this workshop, “from scratch” means building the project step by step. The main end-to-end model uses scikit-learn; the bonus KNN exercise implements the core prediction logic manually with NumPy.

## Project
**Iris flower classification**: predict an Iris flower species from four measurements. The dataset is included in scikit-learn, so no separate download is needed.

## Requirements
- Python 3.10 or newer recommended
- Basic Python knowledge
- VS Code or Jupyter Notebook

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Varshitha3110/ml-model-from-scratch.git
cd ml-model-from-scratch
```

### 2. Create and activate a virtual environment
Windows:
```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the main project
```bash
python src/train_model.py
```

### 5. Run the from-scratch KNN exercise
```bash
python src/knn_from_scratch.py
```

## Repository map
```text
ml-model-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── workshop_walkthrough.ipynb
├── src/
│   ├── train_model.py
│   └── knn_from_scratch.py
└── exercises/
    └── participant_tasks.md
```

## Workshop flow (3 hours)
1. **0:00–0:20** — What ML is, examples, features vs. labels
2. **0:20–0:40** — Dataset exploration and train/test split
3. **0:40–1:15** — Train the first model
4. **1:15–1:25** — Break and questions
5. **1:25–1:55** — Evaluation and interpreting errors
6. **1:55–2:30** — Implement KNN prediction logic with NumPy
7. **2:30–2:50** — Participant exercises
8. **2:50–3:00** — Recap, questions, next steps

## Expected output
The training script prints the dataset shape, test accuracy, confusion matrix, classification report, and a sample prediction. Exact metric values can vary if you change the split or model settings.

## Common issues
- **`ModuleNotFoundError`**: activate the virtual environment and run `pip install -r requirements.txt`.
- **`python` command not found**: try `python3` on macOS/Linux.
- **Notebook kernel missing**: select the `.venv` interpreter in VS Code, or run `python -m ipykernel install --user --name ml-workshop`.
- **Low or different accuracy**: check that the random seed and train/test split match the example.

## Learning goals checklist
- [ ] Explain what features and labels are
- [ ] Explain why data is split into training and test sets
- [ ] Train and evaluate a classifier
- [ ] Interpret a confusion matrix
- [ ] Explain KNN distance and majority voting

## Author
Workshop materials by Varshitha.
