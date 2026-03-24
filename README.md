# 📱 Play Store App Review Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?style=for-the-badge&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> **End-to-end EDA Project** — Analysing Google Play Store applications and user reviews to understand app popularity, ratings, installs, category trends, and user sentiment.

---

## 📊 Project Overview

This project analyses two datasets — **Play Store Data** and **User Reviews** — to extract meaningful insights about what makes an app successful on the Google Play Store.

| Dataset | Description |
|---------|-------------|
| `Play Store Data.csv` | App metadata — name, category, rating, installs, size, price |
| `User Reviews.csv` | User reviews with sentiment polarity and subjectivity scores |

---

## 🔬 Analysis Sections

### 1️⃣ Data Loading & Overview
- Loaded both CSV datasets using `pd.read_csv()`
- Inspected shape, columns, and missing values

### 2️⃣ Data Cleaning & Preprocessing
- Converted `Rating`, `Reviews`, `Installs`, `Price` to numeric
- Cleaned `Installs` column — removed `+` and `,` symbols
- Converted `Size` to MB (handled K, M, G units + "Varies with device")
- Removed duplicate records
- Filled missing values with median imputation
- Exported cleaned datasets to CSV

### 3️⃣ Univariate Analysis
- **Ratings Distribution** — histogram showing most apps rated 4.0–4.5
- **Top 10 Categories** — Family, Games, Tools dominate
- **Content Rating Pie** — majority are "Everyone" rated

### 4️⃣ Bivariate Analysis
- **Free vs Paid Installs** — boxplot showing free apps get significantly more installs
- **Reviews vs Rating Scatter** — trendline showing stable ratings for popular apps
- **Top Categories by Total Installs** — Communication, Social, Games lead

### 5️⃣ Correlation Analysis
- Heatmap of Rating, Reviews, Installs, Price, Size
- Identifies relationships between numerical features

### 6️⃣ Sentiment Analysis
- Sentiment distribution — Positive / Negative / Neutral
- Average polarity by category
- Rating vs Polarity scatter plot per category

---

## 💡 Key Insights

1. 📈 Most apps are rated between **4.0 and 4.5** — Play Store skews positive
2. 🏆 **Family, Games, Tools** have the highest number of apps
3. 📥 **Free apps** get significantly more installs than paid apps
4. 💬 **Communication, Social, Games** categories have the highest total installs
5. 😊 Majority of user reviews are **Positive** in sentiment
6. 👨‍👩‍👧 Most apps are rated **"Everyone"** — suitable for all age groups

---

## 🗂️ Project Structure

```
PlayStore-EDA-Analysis/
│
├── playstore.py                   # Main EDA script
├── Play Store Data.csv            # Raw Play Store dataset
├── User Reviews.csv               # Raw User Reviews dataset
├── eda_outputs/
│   ├── play_store_cleaned.csv     # Cleaned Play Store data
│   ├── user_reviews_cleaned.csv   # Cleaned Reviews data
│   ├── ratings_hist.png           # Ratings distribution
│   ├── top_categories_barh.png    # Top categories by apps
│   ├── content_rating_pie.png     # Content rating pie chart
│   ├── installs_boxplot.png       # Free vs Paid installs
│   ├── reviews_rating_scatter.png # Reviews vs Rating
│   ├── correlation_heatmap.png    # Correlation heatmap
│   ├── sentiment_bar.png          # Sentiment distribution
│   └── kpis.txt                   # Key performance indicators
├── LICENSE                        # MIT License
└── README.md
```

---

## 🛠️ Tech Stack

```python
pandas        # Data loading, cleaning, aggregation
numpy         # Numerical operations, trendlines
matplotlib    # All visualizations
pathlib       # File path management
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Raj-Verma-1998/PlayStore-EDA-Analysis.git
cd PlayStore-EDA-Analysis

# 2. Install dependencies
pip install pandas numpy matplotlib

# 3. Update file paths in playstore.py (lines 13-14)
# play_path = Path("Play Store Data.csv")
# reviews_path = Path("User Reviews.csv")

# 4. Run
python playstore.py
```

---

## 📈 Visualizations

| Chart | Description |
|-------|-------------|
| `ratings_hist.png` | Distribution of app ratings |
| `top_categories_barh.png` | Top 10 categories by number of apps |
| `content_rating_pie.png` | Content rating distribution |
| `installs_boxplot.png` | Free vs Paid app installs comparison |
| `reviews_rating_scatter.png` | Reviews vs Rating with trendline |
| `correlation_heatmap.png` | Correlation between numeric features |
| `sentiment_bar.png` | Positive / Negative / Neutral review distribution |

---

## 👨‍💻 Author

**Raj Verma**
- GitHub: [@Raj-Verma-1998](https://github.com/Raj-Verma-1998)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
