# =====================================================================
# Do drugs with fewer reported side effects have higher ratings?
# Primary analysis: ordered beta regression of rating (0-10) on
# n_side_effects, adjusting for review volume and medical condition.
# =====================================================================

library(tidyverse)
library(ordbetareg)
library(randomForest)
library(DHARMa)
library(emmeans)
library(lme4)
library(performance)

setwd("Documents/programming/drugs_and_side_effects")

set.seed(42)

df <- read_csv("stats_df.csv")

# One row per drug x condition x rating x reviews cell
# n_side_effects = distinct side effects listed for that drug
drug_level <- df %>%
  group_by(drug_name, medical_condition, drug_classes, rx_otc,
           pregnancy_category, csa, rating, no_of_reviews, drug_class) %>%
  summarise(n_side_effects = n_distinct(side_effects), .groups = "drop") %>%
  mutate(
    p_rating = rating / 10,
    medical_condition = fct_lump_n(medical_condition, n = 10,
                                   other_level = "Other")
  )

# Ratings are missing when there are no reviews, therefore the downstream analysis
# contains drugs that have received at least one review

# Analysis sample for primary model
dat <- drug_level %>%
  filter(complete.cases(rating, n_side_effects, medical_condition, no_of_reviews))
cat("Rows:", nrow(dat), " Distinct drugs:", n_distinct(dat$drug_name),
    " Rows per drug (mean):", round(nrow(dat) / n_distinct(dat$drug_name), 2), "\n")

# Bounded outcome with piles at 0 and 10
# suggests ordered beta rather than Gaussian
ggplot(dat, aes(x = p_rating)) +
  geom_histogram(binwidth = 0.05, boundary = 0) +
  labs(x = "Rating (rescaled to 0-1)", y = "Count") +
  theme_basic()

ggplot(dat, aes(x = n_side_effects, y = p_rating)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "gam", formula = y ~ s(x)) +
  labs(x = "Number of side effects", y = "Rating (0-1)") +
  theme_basic()

# Primary model: ordered beta regression
# Handles exact 0s and 10s as censored boundary values
m1 <- ordbetareg(
  bf(p_rating ~ n_side_effects + log(no_of_reviews + 1) + medical_condition,
     phi      ~ n_side_effects + log(no_of_reviews + 1) + medical_condition),
  phi_reg = "both",   
  data = dat)
summary(m1)

# Simulated residuals vs. fitted model
sim_res <- simulateResiduals(m1, plot = TRUE)

# Effect of number of side effects on the original 0-10 scale
emm1 <- emmeans(m1, ~ n_side_effects, 
                at = list(n_side_effects = c(5, 10,15, 20,25, 30)))
emm1_df <- as.data.frame(emm1) %>%
  mutate(rating_10 = emmean * 10,
         lower_10  = lower.HPD * 10,
         upper_10  = upper.HPD * 10) %>%
  select(-emmean, -lower.HPD, -upper.HPD)
print(emm1_df)

ggplot(emm1_df, aes(x = n_side_effects, y = rating_10)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower_10, ymax = upper_10),
                width = 1) +
  labs(x = "Number of side effects", 
       y = "Expected rating on 0-10 scale (95% CI)") +
  theme_basic()

# How much variance do the attributes explain?
set.seed(123)
rf_data <- dat %>%
  select(p_rating, medical_condition, drug_class, no_of_reviews,
         pregnancy_category, csa, rx_otc, n_side_effects) %>%
  na.omit()

folds <- sample(rep(1:5, length.out = nrow(rf_data)))
cv_r2 <- map_dbl(1:5, function(k) {
  train <- rf_data[folds != k, ]
  test <- rf_data[folds == k, ]
  rf <- randomForest(p_rating ~ medical_condition + drug_class +
                       no_of_reviews + pregnancy_category + csa +
                       rx_otc + n_side_effects, 
                     data = train)
  pred <- predict(rf, test)
  1 - sum((test$p_rating - pred)^2) / sum((test$p_rating - mean(test$p_rating))^2)
})
print(round(cv_r2, 3))
cat("Mean 5-fold CV R2:", round(mean(cv_r2), 3))

rf_full <- randomForest(p_rating ~ medical_condition + drug_class +
                          no_of_reviews + pregnancy_category + csa +
                          rx_otc + n_side_effects, 
                        data = rf_data, importance = TRUE)
importance(rf_full)
