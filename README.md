# Spotify Track Popularity Prediction 🎵📈

Predict Spotify track **popularity (0–100)** from audio features and lightweight metadata using a **two-stage ML pipeline**:

1) **Classification** — detect whether a track’s popularity is **0 vs > 0**  
2) **Regression** — predict the **exact popularity** for tracks predicted as non-zero (using the classifier’s probability as an extra feature)

> Notebook: `Popularity Prediction.ipynb`

---

## 🚀 Highlights

- **Dataset:** [maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (114,000 rows)
- **Target:** `popularity` (integer 0–100)
- **Best combo (chosen):** **RandomForest (classifier) + GradientBoosting (regressor)**
- **Test performance:** `R² = 0.463`, `MAE = 11.35`, `MSE = 269.55`  
  *(80/20 train/test split, scaling as below)*
- **Why two-stage?** The dataset has many **zero-popularity** tracks; modeling zeros separately improves fit and stability.

---




---

## 📚 Data & Features

**Source:** Hugging Face (linked above). After loading, the notebook performs light cleaning and EDA.

**Columns used for modeling** (raw columns dropped are listed below):
- Numeric/audio: `danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms`
- Musical meta: `key, mode, time_signature`
- Flag: `explicit` (0/1)
- **Target-encoded genre:** `track_genre_encoded` (mean popularity per genre)

**Dropped from X:** `track_id, artists, album_name, track_name, track_genre` (high cardinality text), and `duration_min` (derived for EDA only).

**Basic stats (after cleaning):**
- Unique tracks: **89,741**
- Unique artists: **31,437**
- Unique albums: **46,589**
- Avg popularity: **33.24**
- Explicit tracks: **8.55%**
- Avg duration: **3.80 minutes**

---

## 🧹 Preprocessing

- **Missing values:** very few → **rows dropped** (`df.dropna()`)
- **Genre handling:** **target encoding**  
  `track_genre_encoded = mean(popularity | track_genre)`
- **Scaling:**
  - `StandardScaler` for: `duration_ms, danceability, energy, track_genre_encoded, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo`
  - `MinMaxScaler` for: `key, time_signature`
- **Split:** `train_test_split(test_size=0.2, random_state=42)`

> ⚠️ **Outliers not removed** to retain true “viral/unusual” tracks; tree models are robust and benefit from full distribution.

---

## 🏗️ Modeling Strategy (Two-Stage)

1. **Stage A — Classification (0 vs >0)**  
   Models tried: `LogisticRegression (balanced)`, `DecisionTreeClassifier (balanced)`, `RandomForestClassifier (balanced)`, `GradientBoostingClassifier`, `XGBClassifier (scale_pos_weight)`, `SVC(probability=True, balanced)`, `KNeighborsClassifier`, `GaussianNB`

2. **Stage B — Regression (only on >0 cases)**  
   Models tried: `LinearRegression`, `Ridge`, `Lasso`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBRegressor`, `KNeighborsRegressor`, `SVR`

3. **Blending trick:** the **classifier’s non-zero probability** (`p(nonzero)`) is **appended as an extra feature** to the regressor.  
   Final predictions = `0` for negatives; regressor output for positives.

**Metrics reported:** `MAE`, `MSE`, `R²` on both train and test.

---

## 🏆 Results (Selected)

| Classifier + Regressor             | Test MAE | Test MSE | Test R² |
|------------------------------------|---------:|---------:|--------:|
| **RandomForest + GradientBoosting** (chosen) | **11.35** | **269.55** | **0.463** |
| RandomForest + RandomForest        | 10.03    | 239.81   | 0.522 |
| RandomForest + XGBoost             | 10.38    | 245.64   | 0.510 |
| RandomForest + SVR                 | 11.35    | 297.34   | 0.407 |
| DecisionTree + RandomForest        | 11.42    | 359.37   | 0.284 |

> Full table with 72 model combinations is generated in the notebook as `df_results`.

---

## ⚖️ Model Fit Analysis

To judge fit quality, we compare **train vs test R²**:

### ✅ Best Fit
- **RandomForest + GradientBoosting (chosen)**  
  - Balanced performance (`R² = 0.463`)  
  - Small gap between train and test → generalizes well.  
- **RandomForest + RandomForest / XGBoost**  
  - Higher accuracy, but slightly bigger variance than GradientBoosting.  

### 🔺 Overfit Models
- **DecisionTree + RandomForest**  
  - Train R² very high (close to 1.0), test R² = 0.28.  
  - **Reason:** Decision Tree memorizes training set → poor generalization.  
- **KNN regressors (any combo)**  
  - Train R² > 0.8, test R² often < 0.3.  
  - **Reason:** KNN is sensitive to noise and high-dimensional data.  

### 🔻 Underfit Models
- **Linear Regression / Ridge / Lasso**  
  - Train and test R² both ~0.25–0.35 (very low).  
  - **Reason:** Too simple; cannot capture non-linear music patterns.  
- **Naive Bayes classifier combos**  
  - Classifier fails to separate 0 vs >0 properly.  
  - **Reason:** Gaussian assumption doesn’t match audio features.  

---

## 🛠️ Environment & Setup

**Python:** 3.9+ recommended

**Install:**
```bash
pip install -U pip
pip install datasets scikit-learn xgboost pandas numpy matplotlib seaborn
