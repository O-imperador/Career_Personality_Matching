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

## Usage

```python
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

df = pd.read_csv("careerFixed.csv")
X = df.drop("Career", axis=1)
y = df["Career"]

new_person = np.array([[7.5, 8.0, 6.0, 7.0, 4.5, 6.5, 5.0, 6.0, 7.0, 7.5]])
distances = euclidean_distances(new_person, X)

top_idx = np.argsort(distances[0])[:5]
for rank, idx in enumerate(top_idx, start=1):
    print(f"Rank {rank}: {y.iloc[idx]}, distance = {distances[0][idx]:.4f}")
