# Participant Exercises

Try these after the instructor demo.

## Task 1 — Explore the data
1. Print the number of rows and columns.
2. Print all feature names.
3. Find the number of samples in each species.
4. Why is the target column not included in `X`?

## Task 2 — Change the model
1. In `src/train_model.py`, change `n_neighbors=5` to `n_neighbors=3`.
2. Run the script again.
3. Compare the accuracy and confusion matrix. Did the result change?

## Task 3 — Understand the split
1. Change `test_size=0.2` to `0.3`.
2. Explain what changed in the number of training and test samples.
3. Why should we evaluate on data the model did not train on?

## Task 4 — KNN logic
1. In `src/knn_from_scratch.py`, change `k=5` to `k=3`.
2. Describe what `np.argsort(distances)[:k]` does.
3. What could happen if `k=1`? What if `k` is too large?

## Task 5 — Stretch goal
Make a simple bar chart of the class counts, or print one new prediction for a set of flower measurements.

## Discussion questions
- Is accuracy always the best metric?
- What is the difference between training and prediction?
- Why do we scale features for KNN?
- What is data leakage, and how does fitting the scaler on all data cause it?  
