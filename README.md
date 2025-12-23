# Career Recommendation Based on Personality and Aptitude Traits

## Project Overview
The goal of this project was to create a **machine learning solution** to recommend the best career based on an individual's personality and aptitude scores. The dataset contains OCEAN personality traits and various aptitude scores for multiple careers.

## Key Finding
Upon exploring the dataset, we discovered that most careers have **only one sample**, making it impossible to train a meaningful ML model.  

## Approach
Instead of ML, we use a **similarity-based recommendation system**:

1. Represent each career as a point in multi-dimensional trait space.
2. Compute **Euclidean distances** from a new individual to all careers.
3. Recommend the **top N closest careers** as the best fit.

Optionally, a **KNN classifier** with `n_neighbors=1` can be used; this is functionally equivalent to the distance-based approach.
