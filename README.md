# Drugs and Their Side Effects
## Data obtained from Kaggle:
[Data](https://www.kaggle.com/datasets/nailasrivastava/drugs-side-effects-and-medical-conditions?select=drugs_side_effects_drugs_com.csv)

[Licence](https://creativecommons.org/publicdomain/zero/1.0/)

## About
The dataset contains data about drugs scraped from Drugs.com by [Nalia Srivastava](https://github.com/Naila-Srivastava). Columns included are drug names (both generic and branded), ratings, number of reviews, side effects, prescription status and more. The aim of this project was to perform Exploratory Data Analysis (EDA) to generate preliminary graphs and visualisations, then perform more in depth analysis about the side effects associated with each drug and medical condition.

## Tools
 - Python (Pandas, NumPy, Seaborn, Matplotlib)
 - R (tidyverse, randomForest, ordbetareg)
 - Git & GitHub

## Data Cleaning
The dataset was first cleaned and filtered, using Pandas to drop the stray and unnamed columns due to misformatting and also ensuring the rating and review columns are numerical. This was then saved to a new dataframe.

The side effect column also had to be cleaned to create a long dataset comprising of one row for each side effect in each drug. Terms were clustered with similar terms using DBSCAN (cosine distance). For example, 'difficulty breathing' and 'breathing difficulty' were compressed into one cluster. The smallest term by character was used and every term in the cluster was changed to this.

This clustering method was repeated three times until I was happy the clusters formed were all valid clusters. The final merge required manual checking and deduplication to achieve the one row for each side effect dataset. The full code with notes written can be found [here](/text_cleaning.py).

## EDA
Visualisations were created to explore the data, including most reviewed drugs, most common side effects, cumulative proportion of side effects, medical conditions with the most drugs, distribution of ratings, and the top 20 drugs by side effect count and what they treat.

I also examined how many rows contained the top 200 most common side effects (77.4% of rows) and side effects that appeared over 10 times (88.3% of rows). I decided to restrict to the top 200 side effects for PCA, which I performed on the top 10 most common drug classes by side effect profile.

## Statistics
I performed ordered beta regression with rating as the response variable to see if any of the variables could explain the variation in rating. The ordered beta regression instead of classic GLM restricts the predictions to 0-10 as those are the only available ratings possible.

Random Forest model found that the variables available only explained 25.76% of the variation in rating, which is not surprising as humans are very messy. 

