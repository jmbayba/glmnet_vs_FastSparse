# Load the successfully installed library
library(glmnet)
library(FastSparse)

df_train <- read.csv("C:/Users/bayba/jmb_analysis/train.csv")
# View(df_train)

# --- PART 1: Basic Visuals (Base R) ---
# Assuming your data is in a dataframe named 'df'
# Barplot for Customer Type
barplot(table(df_train$Customer.Type), main="Customer Type Distribution", col="lightblue")

# Histogram for Flight Distance
hist(df_train$Flight.Distance, main="Flight Distance Distribution", 
     xlab="Distance", col="lightgreen", breaks=20)

# Histogram for Age
hist(df_train$Age, main="Age Distribution", col="skyblue")

# Histogram for Arrival Delay in Minutes
hist(df_train$Arrival.Delay.in.Minutes, main="Arrival Delay", col="red")

# Boxplot of Age by Satisfaction
boxplot(Age ~ as.factor(satisfaction), data=df_train, 
        main="Age vs Satisfaction", col=c("salmon", "cyan"))

# Visualizing categorical varibles
barplot(table(df_train$Gender), main="Gender Distribution")
barplot(table(df_train$Class), main="Class Distribution")

# Visualizing in regards to satisfaction as the response variable
df_train$satisfaction <- as.factor(df_train$satisfaction)

library(ggplot2)
boxplot(Seat.comfort ~ satisfaction, data=df_train, col=c("red","green"))
boxplot(Inflight.entertainment ~ satisfaction, data=df_train, col=c("red","green"))
boxplot(Cleanliness ~ satisfaction, data=df_train, col=c("red","green"))
boxplot(Checkin.service ~ satisfaction, data=df_train, col=c("red","green"))
boxplot(Inflight.service ~ satisfaction, data=df_train, col=c("red","green"))
boxplot(Ease.of.Online.booking ~ satisfaction, data=df_train, col=c("red","green"))

ggplot(df_train, aes(x=Class, fill=satisfaction)) +
  geom_bar(position="fill") +
  ylab("Proportion")

ggplot(df_train, aes(x=Type.of.Travel, fill=satisfaction)) +
  geom_bar(position="fill")

ggplot(df_train, aes(x=Customer.Type, fill=satisfaction)) +
  geom_bar(position="fill")
ggplot(df_train, aes(x=satisfaction, y=Arrival.Delay.in.Minutes, fill=satisfaction)) +
  geom_boxplot()

ggplot(df_train, aes(x=satisfaction, y=Arrival.Delay.in.Minutes, fill=satisfaction)) +
  geom_boxplot()  +
  scale_y_log10()

ggplot(df_train, aes(x=satisfaction, y=Departure.Delay.in.Minutes, fill=satisfaction)) +
  geom_boxplot()

ggplot(df_train, aes(x=satisfaction, y=Departure.Delay.in.Minutes, fill=satisfaction)) +
  geom_boxplot()  +
  scale_y_log10()

ggplot(df_train, aes(x=satisfaction, y=
                       Flight.Distance, fill=satisfaction)) +
  geom_boxplot()

ggplot(df_train, aes(x=satisfaction, y=Age, fill=satisfaction)) +
  geom_boxplot()

# --- PART 2: Data Preparation ---
# Remove non-predictive ID columns
# Train
df_clean <- df_train[, !(names(df_train) %in% c("X", "id"))]
df_clean <- df_clean[complete.cases(df_clean),]

# Create numeric matrix X and label vector y
# model.matrix handles categorical factors like 'gender' and 'class'
x_vars <- model.matrix(as.factor(satisfaction) ~ . - 1, data = df_clean)
y_var  <- ifelse(df_clean$satisfaction == "satisfied", 1, 0)

# Test
df_test <- read.csv("C:/Users/bayba/jmb_analysis/test.csv")
#View(df_test)

# Remove non-predictive ID columns
df_cleanT <- df_test[, !(names(df_test) %in% c("X", "id"))]
df_cleanT <- df_cleanT[complete.cases(df_cleanT),]

# Create numeric matrix X and label vector y
# model.matrix handles categorical factors like 'gender' and 'class'
x_varsT <- model.matrix(as.factor(satisfaction) ~ . - 1, data = df_cleanT)
y_varT  <- ifelse(df_cleanT$satisfaction == "satisfied", 1, 0)

# Build individual models
# Lasso Regression

# Need to determine optimal penalization using cross validation to find minimal generalized error
# 1. Run cross-validation
cv_fit <- cv.glmnet(x = x_vars, y = y_var, alpha = 1) # alpha = 1 for Lasso, 0 for Ridge

# 2. Extract optimal lambda values
best_lambda <- cv_fit$lambda.min   # Lambda with minimum mean cross-validated error
print(best_lambda)

# 3. View results
plot(cv_fit)

# Lasso fit
airplaneLassoFit <- glmnet(x = x_vars, y = y_var, family = "binomial", lambda = best_lambda, alpha = 1)
summary(airplaneLassoFit)

# Extract coefficients for the optimal lambda
coeffs <- coef(airplaneLassoFit, s = "lambda.min")
print(coeffs)

# View only predictors with non-zero values
coeffs[coeffs[,1] != 0, , drop = FALSE]

# 1. Get the predicted class (0 or 1) using your cross-validated model
# Ensure your test data (x_test) is a matrix
pred_class <- predict(airplaneLassoFit, newx = x_varsT, s = "lambda.min", type = "class")

# 2. Calculate the Accuracy percentage
accuracy_lasso <- mean(pred_class == y_varT)
print(paste("Lasso Accuracy: ", round(accuracy_lasso * 100, 2), "%", sep=""))

# 3. Create a Confusion Matrix for a deeper look at errors
table(Predicted = pred_class, Actual = y_varT)

# Create the formula for the Lasso model that was discovered
coef_vec <- as.vector(coeffs)
names(coef_vec) <- rownames(coeffs)

# Remove zero coefficients (since Lasso shrinks many to 0)
coef_vec <- coef_vec[coef_vec != 0]

# Separate intercept
intercept <- coef_vec["(Intercept)"]
terms <- coef_vec[names(coef_vec) != "(Intercept)"]

# Build expression string
formula_str <- paste0(
  "logit(p) = ",
  round(intercept, 4),
  " + ",
  paste(
    paste0(round(terms, 4), " * ", names(terms)),
    collapse = " + "
  )
)

cat(formula_str)

# Print the number of variables
# Convert to vector
coef_vec_lasso <- as.vector(coeffs)
names(coef_vec_lasso) <- rownames(coeffs)

# Remove intercept
coef_vec_lasso <- coef_vec_lasso[names(coef_vec_lasso) != "(Intercept)"]

# Count non-zero coefficients
num_vars_lasso <- sum(coef_vec_lasso != 0)

print(num_vars_lasso)
# FastSparse model
cv_fit <- FastSparse.cvfit(x_vars, y_var, algorithm = "CDPSI", loss = "Logistic")

supp_size <- cv_fit$fit$suppSize[[1]]   # k values
cv_error  <- cv_fit$cvMeans[[1]][,1]    # CV loss
cv_sd     <- cv_fit$cvSDs[[1]][,1]
best_index <- which.min(cv_error)
best_k <- supp_size[best_index]
print(best_k)
k_index <- which(cv_fit$k == best_k)


beta_matrix <- cv_fit$fit$beta[[1]]   # dgCMatrix (24 x 100)
intercepts  <- cv_fit$fit$a0[[1]]

coef_vec <- beta_matrix[, best_index]
coef_vec <- as.vector(coef_vec)

names(coef_vec) <- cv_fit$fit$varnames

intercept <- intercepts[best_index]

coef_vec <- coef_vec[coef_vec != 0]

formula_str <- paste0(
  "logit(p) = ",
  round(intercept, 4),
  " + ",
  paste(
    paste0(round(coef_vec, 4), " * ", names(coef_vec)),
    collapse = " + "
  )
)

logit <- x_varsT[, names(coef_vec), drop = FALSE] %*% coef_vec + intercept
prob <- 1 / (1 + exp(-logit))
pred_class <- ifelse(prob > 0.5, 1, 0)

accuracy_fs <- mean(pred_class == y_varT)
print(paste("FastSparse Accuracy:", round(accuracy_fs * 100, 2), "%"))

table(Predicted = pred_class, Actual = y_varT)

accuracy_fs

# Print the number of variables
num_vars_fs <- length(coef_vec)

print(num_vars_fs)
comparison_df <- data.frame(
  Method <- c("glmnet(Lasso)", "FastSparse"),
  Accurracy <- c(accuracy_lasso, accuracy_fs),
  NumberOfVariables <- c(num_vars_lasso, num_vars_fs)
)
