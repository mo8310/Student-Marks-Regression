# Student Marks Prediction - Linear Regression

A small regression project that predicts a student's marks based on how many courses they're taking and how many hours they study, using the Student Marks dataset. Includes both a from-scratch gradient descent implementation and a scikit-learn `LinearRegression` model.

## Overview

The idea is simple: does study time and course load predict how well a student scores? The dataset has just three columns (number of courses, study time in hours, and marks), so this is a straightforward regression problem used mainly to practice the full pipeline — EDA, splitting the data, training a model, and evaluating it — plus reimplementing linear regression by hand with gradient descent to see how it compares to `scikit-learn`'s version.

## Project Workflow

Data Loading → Data Info / Cleaning Check → Summary Statistics → EDA (correlation heatmap, boxplots, regression plot) → Train/Test Split → Manual Linear Regression (gradient descent) → scikit-learn Linear Regression → Evaluation (R², MSE, MAE) → Actual vs Predicted Plot → Prediction Function

## Dataset

The project uses the **Student Marks Dataset** (`Student_Marks.csv`, loaded from a Kaggle input path).

- **Samples:** 100 rows, 3 columns
- **Features:** `number_courses` (int, number of courses taken), `time_study` (float, hours studied)
- **Target:** `Marks` (float)
- **No missing values** — confirmed with `df.isnull().sum()`
- Summary stats: `number_courses` ranges 3–8 (mean ~5.29), `time_study` ranges 0.096–7.957 hours (mean ~4.08), `Marks` ranges 5.609–55.299 (mean ~24.42)

## Technologies & Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (`LinearRegression`, `train_test_split`, `mean_squared_error`, `r2_score`, `mean_absolute_error`)

## Implementation

### 1. Data Loading

The dataset is loaded directly from a CSV with `pd.read_csv`. Basic checks (`.info()`, `.head()`, `.tail()`, `.describe()`) are used to look at the data before doing anything else.

### 2. Data Preprocessing

There isn't much preprocessing needed here — the dataset has no missing values and is already numeric. A correlation heatmap, boxplots for each column (to check the spread of `number_courses`, `time_study`, and `Marks`), and a regression plot of `time_study` vs `Marks` are used to explore the relationships before modeling. For the scikit-learn model, features (`x`) are `time_study` and `number_courses`, and the target (`y`) is `Marks`, split 80/20 with `train_test_split(test_size=0.2, random_state=42)`.

### 3. Model

Two versions of linear regression are implemented:

**Manual gradient descent.** This part is set up as a separate, single-feature experiment — it predicts `time_study` from `number_courses` (not `Marks`), using `x` and `y` that are reassigned after the train/test split and reshaped into numpy arrays. Weight and bias are randomly initialized (`np.random.seed(42)`), then updated with gradient descent for up to 1000 epochs at a learning rate of 0.01, with an early stop if the cost drops below `1e-5`. This section is really a from-scratch demonstration of how gradient descent works, separate from the actual marks-prediction model below.

**scikit-learn Linear Regression.** This is the actual predictive model — `LinearRegression()` fit on `x_train` (`time_study`, `number_courses`) and `y_train` (`Marks`).

### 4. Training

The manual model is trained with a simple loop: compute predictions, compute the gradients for weight and bias, update both, track the cost each epoch, and stop early if the cost gets small enough. The scikit-learn model is trained with a single `.fit(x_train, y_train)` call — no hyperparameter tuning involved, since ordinary least squares doesn't need any.

### 5. Evaluation

The scikit-learn model is evaluated on the test set using R², MSE, and MAE. An "Actual vs Predicted" scatter plot (with a diagonal reference line) is used to visually check how close the predictions are to the real marks.

### 6. Prediction / Inference

A helper function, `predict_student_marks(number_courses, time_study)`, wraps the trained scikit-learn model and prints a predicted mark for a given number of courses and study hours.

## Results

**Manual gradient descent (predicting `time_study` from `number_courses`):**

| Metric | Value |
|---|---|
| Weight | 0.3745 |
| Bias | 2.0347 |
| Final cost (MSE/2) | 2.6896 |

**scikit-learn Linear Regression (predicting `Marks` from `time_study` and `number_courses`, on the test set):**

| Metric | Value |
|---|---|
| R² Score | 0.9460 (94.60%) |
| MSE | 14.2007 |
| MAE | 3.0793 |

The scikit-learn model explains about 94.6% of the variance in marks on the test set, which is a strong fit given how few features there are — study time in particular seems to be a strong predictor of marks, which lines up with what the correlation heatmap and regression plot show.

As a sanity check, the trained model predicts a mark of about **26.23** for a student taking 4 courses and studying 6.5 hours.

Plots generated in the notebook (not saved as separate image files — re-run the notebook or export them if you want to embed images here):

![Correlation Heatmap](results/correlation_heatmap.png)
![Feature Boxplots](results/boxplots.png)
![Study Time vs Marks Regression Plot](results/time_study_vs_marks.png)
![Actual vs Predicted Marks](results/actual_vs_predicted.png)

## Project Structure

```text
student-marks-linearregression/
│
├── student-marks-linearregression.ipynb
└── README.md
```

(The dataset CSV isn't included in the uploaded project — see the Dataset section for where it comes from.)

## How to Run

1. Make sure you have Python 3 installed.
2. Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. Download `Student_Marks.csv` (Kaggle: yasserh/student-marks-dataset) and place it in your working directory, updating the file path in the notebook (it currently points to a Kaggle input path).
4. Open and run the notebook top to bottom:

```bash
jupyter notebook student-marks-linearregression.ipynb
```

## Example Usage

```python
test_prediction = predict_student_marks(number_courses=4, time_study=6.5)
# Predicted Degree: 26.23151457279626
```

## Notes

- The manual gradient descent section and the scikit-learn model are solving two different problems (`time_study` from `number_courses`, vs. `Marks` from `time_study` + `number_courses`), so their results aren't directly comparable — this is worth keeping in mind since they're both under "Linear Regression" in the notebook but aren't predicting the same thing.
- The dataset is very small (100 rows), so the R² score, while high, should be read with that in mind — there's not a lot of data to validate generalization.
- No cross-validation was done; evaluation is based on a single 80/20 train/test split.
- No GPU is needed — everything here runs fine on CPU given the dataset size and model simplicity.
