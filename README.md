📱 Google Play Store Apps: Clustering and Install Prediction
Course: ISM4420 – Business Analytics
Institution: Florida International University
Professor: Dr. Hemang Subramanian
Team Members: Luciano Colla, Marko Rosic, Arturo Sierra
Date: April 2025

📌 Overview
This project explores the Google Play Store app ecosystem using machine learning techniques to:

Cluster apps based on their features

Analyze cluster characteristics

Predict install counts using regression models

Recommend marketing strategies tailored to app segments

🔍 Problem Statement
We aimed to:

Identify natural groupings of apps using clustering algorithms.

Characterize each group to understand what differentiates app segments.

Build regression models to predict app install counts.

Develop targeted marketing strategies for each cluster based on their traits.

🧼 Data Cleaning
Converted text fields (reviews, installs, price, size) into numeric formats.

Removed duplicates, invalid entries, and rows with missing data.

One-hot encoded categorical variables for modeling.

Standardized numeric features to ensure fair clustering.

📊 Dataset
Original: 10,841 apps with 13 features.

Cleaned: 7,420 apps and 9 key features.

Variables included: Category, Rating, Reviews, Size, Installs, Type, Price, Content Rating, and Genres.

📈 Modeling
Clustering
K-means (k=3) and Hierarchical Clustering (Ward’s method) were applied.

Hierarchical clustering yielded stronger cluster separation (Silhouette score: 0.82 vs. 0.19 for K-means).

Identified Clusters:
Top Global: Few apps with massive installs (e.g., WhatsApp, Instagram).

Popular: Mid-sized apps with millions of installs.

Niche: Most apps, low installs, some monetized upfront.

Regression Models
Built separate linear models for each cluster to predict install counts.

Key predictors: Reviews (strongest), Rating, Price, Size, Content Rating, and Genre.

R² values:

Niche: ~0.39

Popular: ~0.45

Top Global: ~1.00 (overfit due to small size and highly correlated predictors)

📣 Marketing Strategy
Top Global: Broad marketing (TV, global online ads, referrals), large budgets ($500K+).

Popular: Aggressive digital campaigns, influencer partnerships, moderate budgets ($50K–$200K).

Niche: Cost-effective ads, community engagement, focus on ratings/reviews, low budgets ($5K–$20K).

✅ Key Takeaways
Combining unsupervised (clustering) and supervised (regression) methods enables more actionable market segmentation.

App success varies significantly across clusters; understanding which segment an app belongs to informs realistic growth strategies.

Most predictive power comes from user engagement metrics like reviews.
