# ----------------------------------------------------------
# STEP 0: Load Required Libraries
# ----------------------------------------------------------
library(ggplot2)        # For plotting
library(dplyr)          # For data manipulation
library(cluster)        # For clustering and silhouette analysis
library(tidyr)          # For data reshaping

# ----------------------------------------------------------
# STEP 1: Load and Explore Dataset
# ----------------------------------------------------------
df <- read.csv("C:/Users/Luciano/Downloads/googleplaystore.csv (1)/googleplaystore.csv", stringsAsFactors = FALSE)
str(df)                 # Display structure of dataset
summary(df)             # Summary statistics of the dataset

# ----------------------------------------------------------
# STEP 2: Clean and Prepare Data
# ----------------------------------------------------------

## --- Clean 'Reviews' ---
df$Reviews <- as.numeric(df$Reviews)   # Convert reviews to numeric

## --- Clean 'Installs' ---
df$Installs <- gsub("[+,]", "", df$Installs)  # Remove '+' and ',' from installs
df$Installs <- as.numeric(df$Installs)         # Convert installs to numeric

## --- Clean 'Price' ---
df$Price <- gsub("\\$", "", df$Price)       # Remove dollar sign from price
df$Price <- as.numeric(df$Price)               # Convert price to numeric

## --- Clean 'Size' ---
df$Size[df$Size == "Varies with device"] <- NA   # Replace text with NA

df$Size[grepl("M", df$Size)] <- as.numeric(gsub("M", "", df$Size[grepl("M", df$Size)]))    # Convert MB values to numeric
df$Size[grepl("k", df$Size)] <- as.numeric(gsub("k", "", df$Size[grepl("k", df$Size)])) / 1024  # Convert kB to MB
df$Size <- as.numeric(df$Size)   # Ensure all size values are numeric

# ----------------------------------------------------------
# STEP 3: Drop Irrelevant Columns and Filter Invalid Ratings
# ----------------------------------------------------------
df_clean <- df[, !(names(df) %in% c("App", "Last.Updated", "Current.Ver", "Android.Ver"))]  # Drop unnecessary columns
df_clean <- df_clean[df_clean$Rating <= 5, ]  # Remove apps with invalid ratings

# ----------------------------------------------------------
# STEP 4: Handle Missing Values & Duplicates
# ----------------------------------------------------------
df_clean <- df_clean[!duplicated(df_clean), ]  # Remove duplicate rows if any
df_clean <- df_clean[complete.cases(df_clean), ]  # Drop rows with any NA

# Check summary of cleaned data
summary(df_clean)                    
print(colSums(is.na(df_clean)))
nrow(df) - nrow(df_clean)
ncol(df) - ncol(df_clean)

# ----------------------------------------------------------
# STEP 5: Encode Categorical Variables and Normalize Features
# ----------------------------------------------------------

## --- Convert to Factor ---
df_clean$Category <- as.factor(df_clean$Category)            # Convert category to factor
df_clean$Type <- as.factor(df_clean$Type)                    # Convert type to factor
df_clean$Content.Rating <- as.factor(df_clean$Content.Rating)  # Convert content rating to factor
df_clean$Genres <- as.factor(df_clean$Genres)                # Convert genres to factor

## --- Normalize Numeric Variables ---
numeric_columns <- sapply(df_clean, is.numeric)             # Identify numeric columns
numeric_features <- scale(df_clean[, numeric_columns])      # Standardize numeric features

## --- One-Hot Encode Factors ---
factor_features <- model.matrix(~ Category + Type + Content.Rating + Genres - 1, data = df_clean)  # Create dummy variables

## --- Combine All Cleaned Features ---
df_ready <- cbind(numeric_features, factor_features)        # Combine numeric and categorical features
str(df_ready)                                                # Check structure of final dataset
summary(df_ready)                                            # Summary statistics

# ----------------------------------------------------------
# STEP 6: Exploratory Visualizations
# ----------------------------------------------------------

## --- Rating Distribution ---
ggplot(df_clean, aes(x = Rating)) +
  geom_histogram(binwidth = 0.2, fill = "steelblue", color = "white") +
  labs(title = "Distribution of App Ratings", x = "Rating", y = "Count of Apps") +
  scale_x_continuous(limits = c(0, 5)) +
  theme_minimal()

## --- Installs Distribution ---
ggplot(df_clean, aes(x = Installs)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white") +
  labs(title = "Distribution of App Installs", x = "Installs", y = "Count of Apps") +
  theme_minimal()

## --- Paid App Price Distribution ---
ggplot(df_clean[df_clean$Type == "Paid", ], aes(y = Price)) +
  geom_boxplot(fill = "orange") +
  coord_cartesian(ylim = c(0, quantile(df_clean$Price[df_clean$Type == "Paid"], 0.95))) +
  labs(title = "Boxplot of App Prices (Paid Apps Only)", y = "Price (USD)") +
  theme_minimal()

## --- Top Categories ---
top_categories <- df_clean %>%
  count(Category, sort = TRUE) %>%
  slice_max(n, n = 10)

ggplot(top_categories, aes(x = reorder(Category, -n), y = n)) +
  geom_bar(stat = "identity", fill = "purple") +
  labs(title = "Top 10 App Categories", x = "Category", y = "Number of Apps") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## --- Reviews vs Ratings ---
ggplot(df_clean, aes(x = Reviews, y = Rating)) +
  geom_point(alpha = 0.3, color = "darkblue") +
  scale_x_log10() +
  scale_y_continuous(limits = c(0, 5)) +
  labs(title = "App Reviews vs Rating", x = "Reviews (log scale)", y = "Rating") +
  theme_minimal()

## --- Boxplots for Numeric Variables ---
numeric_data <- df_clean %>% select(where(is.numeric))
numeric_long <- pivot_longer(numeric_data, everything(), names_to = "Variable", values_to = "Value")

ggplot(numeric_long, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Boxplot of All Numeric Variables", x = "Variable", y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ----------------------------------------------------------
# STEP 7: K-means Clustering
# ----------------------------------------------------------

set.seed(123)  # Set seed for reproducibility

# --- Elbow Method to determine optimal number of clusters ---
wss <- sapply(1:10, function(k) {
  kmeans(df_ready, centers = k, nstart = 10)$tot.withinss  # Total within-cluster sum of squares
})

# Plot the elbow curve
plot(1:10, wss, type = "b", pch = 19,
     xlab = "Number of Clusters (k)",
     ylab = "Total Within-Cluster Sum of Squares",
     main = "Elbow Method for Optimal K")

# --- Apply K-means with k = 3 ---
kmeans_result <- kmeans(df_ready, centers = 3, nstart = 25)  # Perform K-means clustering

df_clean$Cluster <- as.factor(kmeans_result$cluster)  # Assign cluster labels to original data

# ----------------------------------------------------------
# STEP 8: Visualize K-means Clustering with PCA
# ----------------------------------------------------------

pca_result <- prcomp(df_ready)  # Perform PCA on scaled dataset

# Create a data frame with first 2 principal components and cluster labels
pca_df <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Cluster = df_clean$Cluster
)

# Scatter plot of PCA with clusters
ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(title = "K-Means Clustering Visualization (PCA)",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()

# Compute silhouette score for k-means clusters
dist_matrix <- dist(df_ready)  # Compute Euclidean distance matrix
silhouette_scores <- silhouette(as.numeric(df_clean$Cluster), dist_matrix)  # Silhouette object
mean_sil_width <- mean(silhouette_scores[, 3])  # Average silhouette width
cat("Average Silhouette Width:", mean_sil_width, "\n")

# ----------------------------------------------------------
# STEP 9: Hierarchical Clustering (Ward's Method)
# ----------------------------------------------------------

hc <- hclust(dist_matrix, method = "ward.D2")  # Perform hierarchical clustering

# Plot dendrogram
plot(hc, labels = FALSE, hang = -1, main = "Hierarchical Clustering Dendrogram")
rect.hclust(hc, k = 3, border = 2:4)  # Highlight clusters

df_clean$HC_Cluster <- as.factor(cutree(hc, k = 3))  # Assign hierarchical cluster labels

# Visualize hierarchical clustering using PCA
ggplot(pca_df, aes(x = PC1, y = PC2, color = df_clean$HC_Cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "Hierarchical Clustering (Ward's Method)", color = "Cluster") +
  theme_minimal()

# Compute silhouette score for hierarchical clustering
sil_hc <- silhouette(as.numeric(df_clean$HC_Cluster), dist_matrix)
mean_sil_hc <- mean(sil_hc[, 3])
cat("Mean Silhouette Width for Hierarchical Clustering:", mean_sil_hc, "\n")

# ----------------------------------------------------------
# STEP 10: Analyze Hierarchical Clusters
# ----------------------------------------------------------

df_clean$Cluster_Name <- recode(df_clean$HC_Cluster,  # Rename clusters with descriptive names
                                `1` = "Popular",
                                `2` = "Top Global",
                                `3` = "Niche")

# Summary statistics for each cluster
df_clean %>%
  group_by(Cluster_Name) %>%
  summarise(
    Count = n(),
    Price_mean = mean(Price), Price_sd = sd(Price),
    Rating_mean = mean(Rating), Rating_sd = sd(Rating),
    Reviews_mean = mean(Reviews), Reviews_sd = sd(Reviews),
    Size_mean = mean(Size), Size_sd = sd(Size),
    Installs_mean = mean(Installs), Installs_sd = sd(Installs)
  ) -> cluster_summary

print(cluster_summary)

# ----------------------------------------------------------
# STEP 11: Visualize Cluster Characteristics
# ----------------------------------------------------------

# Average Installs per Cluster
ggplot(df_clean, aes(x = Cluster_Name, y = Installs, fill = Cluster_Name)) +
  stat_summary(fun = mean, geom = "bar") +
  labs(title = "Average Installs per Cluster") +
  theme_minimal()

# Average Ratings per Cluster
ggplot(df_clean, aes(x = Cluster_Name, y = Rating, fill = Cluster_Name)) +
  stat_summary(fun = mean, geom = "bar") +
  labs(title = "Average Ratings per Cluster") +
  theme_minimal()

# Boxplot of Reviews by Cluster
ggplot(df_clean, aes(x = Cluster_Name, y = Reviews, fill = Cluster_Name)) +
  geom_boxplot() +
  labs(title = "Distribution of Reviews by Cluster") +
  theme_minimal()

# Scatter Plot: Reviews vs Installs by Cluster
ggplot(df_clean, aes(x = Reviews, y = Installs, color = Cluster_Name)) +
  geom_point(alpha = 0.5) +
  labs(title = "Reviews vs Installs by Cluster") +
  theme_minimal()

# ----------------------------------------------------------
# STEP 12: Predict Installs per Cluster (Regression)
# ----------------------------------------------------------

# Loop through clusters and fit linear models
r_squared_df <- data.frame(Cluster = character(), R_Squared = numeric(), stringsAsFactors = FALSE)

for (cluster in unique(df_clean$Cluster_Name)) {
  cluster_data <- df_clean[df_clean$Cluster_Name == cluster, ]  # Filter data for each cluster
  model <- lm(Installs ~ Rating + Reviews + Size + Price + Content.Rating + Genres, data = cluster_data)  # Fit linear model
  
  # Print regression summary
  cat("\n--- Regression for Cluster:", cluster, "---\n")
  print(summary(model))
  
  # Store R-squared
  r_squared_df <- rbind(r_squared_df,
                        data.frame(Cluster = cluster, R_Squared = summary(model)$r.squared))
}

# Bar plot of R-squared values per cluster
ggplot(r_squared_df, aes(x = Cluster, y = R_Squared, fill = Cluster)) +
  geom_bar(stat = "identity") +
  labs(title = "R-squared of Installs Regression by Cluster",
       x = "Cluster", y = "R-squared") +
  theme_minimal()

# ----------------------------------------------------------
# STEP 13: Actual vs Predicted Installs Visualization
# ----------------------------------------------------------

# Create data frame for predictions
prediction_plot_data <- data.frame()

for (cluster in unique(df_clean$Cluster_Name)) {
  cluster_data <- df_clean[df_clean$Cluster_Name == cluster, ]  # Get cluster data
  model <- lm(Installs ~ Rating + Reviews + Size + Price + Content.Rating + Genres, data = cluster_data)  # Fit model
  
  # Create prediction data frame
  predictions <- data.frame(
    Cluster = cluster,
    Actual = cluster_data$Installs,
    Predicted = predict(model, newdata = cluster_data)
  )
  
  prediction_plot_data <- rbind(prediction_plot_data, predictions)  # Append predictions
}

# Plot actual vs predicted installs
ggplot(prediction_plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, color = "darkblue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~ Cluster, scales = "free") +
  labs(title = "Actual vs Predicted Installs by Cluster",
       x = "Actual Installs",
       y = "Predicted Installs") +
  theme_minimal()



# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

