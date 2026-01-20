#1)load data and baisc check:
library(readr)
library(dplyr)

df <- read_csv("C:/Users/nidhi/OneDrive/Desktop/BCA-2326 IDS Project/student.csv")

head(df)

cat("Rows:", nrow(df), "Columns:", ncol(df))



#2)Feature Engineering+ EDA 
library(ggplot2)
library(corrplot)

# Convert studytime categories to actual hours
study_map <- c(`1`=2, `2`=5, `3`=10, `4`=15)
df$study_hours <- study_map[as.character(df$studytime)]

# Create attendance rate
max_abs <- max(df$absences, na.rm = TRUE)
df$attendance_rate <- 1 - (df$absences / max_abs)

# Select numeric columns for correlation
num_cols <- c("study_hours", "attendance_rate", "absences", "G1", "G2", "G3")
num_df <- df %>% select(any_of(num_cols)) %>% na.omit()


#CORELATION HEATMAP:
cor_matrix <- cor(num_df, use = "complete.obs")
corrplot(cor_matrix, method = "color", tl.col = "black")

#EDA: SCATTER PLOTS
# Study hours vs G3
ggplot(num_df, aes(x = study_hours, y = G3)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", color = "red") +
  ggtitle("Study Hours vs G3")



#3)Liner Regression Model
# Attendance rate vs G3
ggplot(num_df, aes(x = attendance_rate, y = G3)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", color = "red") +
  ggtitle("Attendance Rate vs G3")

# G2 vs G3
ggplot(num_df, aes(x = G2, y = G3)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", color = "red") +
  ggtitle("G2 vs G3")



# Select features
features <- c("G2", "study_hours", "attendance_rate")

df2 <- df %>% select(all_of(features), G3) %>% na.omit()

# Train-test split (80–20)
set.seed(42)
train_index <- sample(nrow(df2), 0.8 * nrow(df2))
train <- df2[train_index, ]
test  <- df2[-train_index, ]

# Linear Model
model <- lm(G3 ~ G2 + study_hours + attendance_rate, data=train)

# Predictions
pred1 <- predict(model, newdata=test)

# R² Score
r2 <- summary(model)$r.squared
print(paste("R²:", r2))

# MAE
mae <- mean(abs(test$G3 - pred1))
print(paste("MAE:", mae))


#4) Ridge Regression (r version):
library(glmnet)

# Prepare data
X <- model.matrix(G3 ~ ., data = df2)[, -1]  # remove intercept
y <- df2$G3

# Train-test split
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# Ridge regression with CV
ridge_model <- cv.glmnet(X_train, y_train, alpha = 0)

# Best lambda
best_lambda <- ridge_model$lambda.min
print(paste("Best Lambda:", best_lambda))

# Predictions
pred2 <- predict(ridge_model, s = best_lambda, newx = X_test)

# Metrics
r2_ridge <- 1 - sum((y_test - pred2)^2) / sum((y_test - mean(y_test))^2)
mae_ridge <- mean(abs(y_test - pred2))

print(paste("Ridge R²:", r2_ridge))
print(paste("Ridge MAE:", mae_ridge))


#5)Residuals and save predictions:


# Residuals
res1 <- y_test - pred1

# Plot residuals
ggplot(data.frame(res1), aes(x=res1)) +
  geom_histogram(bins=20, fill="skyblue", color="black") +
  ggtitle("Residuals: Linear Regression")

# Create prediction table
pred_df <- data.frame(
  test[, features],
  actual_G3 = test$G3,
  predicted_G3 = pred1
)

head(pred_df)

# Save CSV
write.csv(pred_df, "student_baseline_predictions_R.csv", row.names=FALSE)



