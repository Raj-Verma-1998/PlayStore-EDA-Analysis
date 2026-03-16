# EDA for Play Store App Review Analysis
# -------------------------------------------------
# This notebook-style script loads the two datasets provided,
# cleans and preprocesses them, performs exploratory analysis,
# creates multiple matplotlib visualizations, and exports useful
# artifacts (cleaned data and figures).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# For displaying DataFrames nicely to the user
from ace_tools import display_dataframe_to_user

# Paths
play_path = Path("/mnt/data/Play Store Data.csv")
reviews_path = Path("/mnt/data/User Reviews.csv")
out_dir = Path("/mnt/data/eda_outputs")
out_dir.mkdir(exist_ok=True, parents=True)

# -----------------------
# 1) LOAD DATA
# -----------------------
play = pd.read_csv(play_path)
reviews = pd.read_csv(reviews_path)

# -----------------------
# 2) QUICK OVERVIEW
# -----------------------
summary_info = {
    "play_store_shape": play.shape,
    "user_reviews_shape": reviews.shape,
    "play_store_columns": list(play.columns),
    "user_reviews_columns": list(reviews.columns),
    "play_store_missing": play.isna().sum().to_dict(),
    "user_reviews_missing": reviews.isna().sum().to_dict(),
}
summary_df = pd.DataFrame({
    "Dataset": ["Play Store", "User Reviews"],
    "Rows": [play.shape[0], reviews.shape[0]],
    "Columns": [play.shape[1], reviews.shape[1]]
})
display_dataframe_to_user("Dataset Shapes", summary_df)

# -----------------------
# 3) DATA CLEANING
# -----------------------

# 3.1 Rating: remove invalid ratings (>5) and non-numeric issues
play = play[pd.to_numeric(play["Rating"], errors="coerce") <= 5].copy()
play["Rating"] = pd.to_numeric(play["Rating"], errors="coerce")

# 3.2 Reviews: ensure numeric
play["Reviews"] = pd.to_numeric(play["Reviews"], errors="coerce")

# 3.3 Installs: "1,000,000+" -> integer
def clean_installs(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace("+", "").replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")

play["Installs_num"] = play["Installs"].apply(clean_installs)

# 3.4 Price: "$2.99" or "0" -> float
def clean_price(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace("$", "").strip()
    return pd.to_numeric(x, errors="coerce")

play["Price_num"] = play["Price"].apply(clean_price)

# 3.5 Size: convert to MB; "Varies with device" -> NaN
def parse_size(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if "Varies" in s:
        return np.nan
    if s.endswith("k") or s.endswith("K"):
        # occasionally appears, treat as KB
        val = s[:-1]
        return pd.to_numeric(val, errors="coerce") / 1024.0
    if s.endswith("M") or s.endswith("m"):
        val = s[:-1]
        return pd.to_numeric(val, errors="coerce")
    if s.endswith("G") or s.endswith("g"):
        val = s[:-1]
        return pd.to_numeric(val, errors="coerce") * 1024.0
    # if like "8.7M" it's handled above; if numeric only
    return pd.to_numeric(s, errors="coerce")

play["Size_MB"] = play["Size"].apply(parse_size)

# 3.6 Type: ensure Free/Paid consistent
play["Type"] = play["Type"].fillna("Free")
play.loc[(play["Type"] != "Free") & (play["Price_num"] > 0), "Type"] = "Paid"

# 3.7 Content Rating: fill unknown as "Everyone"
play["Content Rating"] = play["Content Rating"].fillna("Everyone")

# 3.8 Drop exact duplicate app-version rows if any
play_before = play.shape[0]
play = play.drop_duplicates(subset=["App", "Category", "Current Ver"])
play_after = play.shape[0]

# 3.9 Reviews dataset: drop rows without Sentiment label or App
reviews = reviews.dropna(subset=["App", "Sentiment"]).copy()

# 3.10 Basic imputation: fill missing numeric with median where sensible
for col in ["Rating", "Reviews", "Installs_num", "Price_num", "Size_MB"]:
    median_val = play[col].median(skipna=True)
    play[col] = play[col].fillna(median_val)

# Export cleaned datasets
clean_play_path = out_dir / "play_store_cleaned.csv"
clean_reviews_path = out_dir / "user_reviews_cleaned.csv"
play.to_csv(clean_play_path, index=False)
reviews.to_csv(clean_reviews_path, index=False)

# -----------------------
# 4) UNIVARIATE ANALYSIS
# -----------------------
# 4.1 Ratings distribution
plt.figure()
play["Rating"].plot(kind="hist", bins=30, edgecolor="black")
plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(out_dir / "ratings_hist.png")
plt.show()

# 4.2 Top categories by number of apps
top_cat = play["Category"].value_counts().head(10).sort_values()
plt.figure()
top_cat.plot(kind="barh")
plt.title("Top 10 Categories by Number of Apps")
plt.xlabel("Number of Apps")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(out_dir / "top_categories_barh.png")
plt.show()

# 4.3 Content Rating distribution (pie)
content_counts = play["Content Rating"].value_counts()
plt.figure()
plt.pie(content_counts.values, labels=content_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Content Rating Distribution")
plt.axis("equal")
plt.tight_layout()
plt.savefig(out_dir / "content_rating_pie.png")
plt.show()

# -----------------------
# 5) BIVARIATE / MULTIVARIATE
# -----------------------

# 5.1 Installs vs Type (boxplot)
plt.figure()
data_to_plot = [play.loc[play["Type"]=="Free", "Installs_num"],
                play.loc[play["Type"]=="Paid", "Installs_num"]]
plt.boxplot(data_to_plot, labels=["Free", "Paid"], showfliers=False)
plt.title("Installs by App Type (Free vs Paid)")
plt.ylabel("Installs (count)")
plt.tight_layout()
plt.savefig(out_dir / "installs_boxplot.png")
plt.show()

# 5.2 Reviews vs Rating (scatter + trendline)
plt.figure()
plt.scatter(play["Reviews"], play["Rating"], alpha=0.3)
# trendline
coef = np.polyfit(play["Reviews"], play["Rating"], 1)
poly1d_fn = np.poly1d(coef)
xs = np.linspace(play["Reviews"].min(), play["Reviews"].max(), 100)
plt.plot(xs, poly1d_fn(xs))
plt.title("Reviews vs Rating with Trendline")
plt.xlabel("Reviews (count)")
plt.ylabel("Rating")
plt.tight_layout()
plt.savefig(out_dir / "reviews_rating_scatter.png")
plt.show()

# 5.3 Top 10 categories by installs (sum of installs)
cat_installs = play.groupby("Category")["Installs_num"].sum().sort_values(ascending=False).head(10).sort_values()
plt.figure()
cat_installs.plot(kind="barh")
plt.title("Top 10 Categories by Total Installs")
plt.xlabel("Total Installs")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(out_dir / "top_categories_installs_barh.png")
plt.show()

# 5.4 Numerical correlation heatmap (matplotlib imshow)
num_cols = ["Rating", "Reviews", "Installs_num", "Price_num", "Size_MB"]
corr = play[num_cols].corr()
plt.figure()
im = plt.imshow(corr.values, interpolation="nearest")
plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
plt.yticks(range(len(num_cols)), num_cols)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title("Correlation Heatmap (Numeric Features)")
# annotate
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
plt.tight_layout()
plt.savefig(out_dir / "correlation_heatmap.png")
plt.show()

# -----------------------
# 6) SENTIMENT ANALYSIS
# -----------------------

# 6.1 Sentiment distribution
sent_counts = reviews["Sentiment"].value_counts()
plt.figure()
sent_counts.plot(kind="bar")
plt.title("Sentiment Distribution in User Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(out_dir / "sentiment_bar.png")
plt.show()

# 6.2 Average sentiment polarity per app -> join with play to get category
avg_polarity = reviews.groupby("App")["Sentiment_Polarity"].mean().reset_index()
app_to_cat = play[["App", "Category"]].drop_duplicates()
avg_polarity = avg_polarity.merge(app_to_cat, on="App", how="left")

cat_polarity = avg_polarity.dropna(subset=["Category"]).groupby("Category")["Sentiment_Polarity"].mean().sort_values(ascending=False)
top_polarity = cat_polarity.head(10).sort_values()

plt.figure()
top_polarity.plot(kind="barh")
plt.title("Top 10 Categories by Avg. Review Polarity")
plt.xlabel("Average Polarity (-1 to 1)")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(out_dir / "top_categories_polarity_barh.png")
plt.show()

# 6.3 Compare average rating vs average polarity by category (scatter)
cat_rating = play.groupby("Category")["Rating"].mean()
common_idx = cat_rating.index.intersection(cat_polarity.index)
plt.figure()
plt.scatter(cat_rating.loc[common_idx], cat_polarity.loc[common_idx])
# add simple trendline
coef2 = np.polyfit(cat_rating.loc[common_idx].values, cat_polarity.loc[common_idx].values, 1)
line_fn = np.poly1d(coef2)
xline = np.linspace(cat_rating.loc[common_idx].min(), cat_rating.loc[common_idx].max(), 100)
plt.plot(xline, line_fn(xline))
plt.title("Avg. Rating vs Avg. Review Polarity (by Category)")
plt.xlabel("Average Rating")
plt.ylabel("Average Polarity")
plt.tight_layout()
plt.savefig(out_dir / "rating_vs_polarity_scatter.png")
plt.show()

# -----------------------
# 7) EXPORT KEY TABLES FOR REVIEW
# -----------------------
key_tables = {
    "Top Categories by Apps": play["Category"].value_counts().head(15).rename_axis("Category").reset_index(name="App_Count"),
    "Top Categories by Installs": play.groupby("Category")["Installs_num"].sum().sort_values(ascending=False).head(15).rename("Total_Installs").reset_index(),
    "Paid vs Free Summary": play.groupby("Type")[["Installs_num", "Rating", "Price_num"]].agg({"Installs_num":"median","Rating":"mean","Price_num":"mean"}).reset_index(),
}

for title, df in key_tables.items():
    display_dataframe_to_user(title, df)

# Save tables to CSV
for name, df in key_tables.items():
    safe_name = name.lower().replace(" ", "_") + ".csv"
    df.to_csv(out_dir / safe_name, index=False)

# -----------------------
# 8) BASIC INSIGHTS (computed for text summary outside)
# -----------------------
# Calculate some KPIs to print out
kpis = {}

# Free vs Paid median installs
free_med = play.loc[play["Type"]=="Free", "Installs_num"].median()
paid_med = play.loc[play["Type"]=="Paid", "Installs_num"].median()
kpis["free_vs_paid_median_installs"] = (free_med, paid_med)

# Highest-rated categories (min 50 apps to be robust)
cat_counts = play["Category"].value_counts()
eligible_cats = cat_counts[cat_counts >= 50].index
kpis["top_rated_categories"] = play[play["Category"].isin(eligible_cats)].groupby("Category")["Rating"].mean().sort_values(ascending=False).head(5).round(2)

# Correlation between reviews and rating
kpis["corr_reviews_rating"] = float(play["Reviews"].corr(play["Rating"]))

# Save KPIs for later reference
kpi_path = out_dir / "kpis.txt"
with open(kpi_path, "w") as f:
    f.write("Free vs Paid median installs: Free={}, Paid={}\n".format(int(free_med), int(paid_med)))
    f.write("Correlation (Reviews, Rating): {:.4f}\n".format(kpis["corr_reviews_rating"]))
    f.write("Top rated categories (>=50 apps):\n")
    f.write(kpis["top_rated_categories"].to_string())

# Provide paths for user to download
str({
    "cleaned_play_store_csv": str(clean_play_path),
    "cleaned_user_reviews_csv": str(clean_reviews_path),
    "outputs_dir": str(out_dir)
})
