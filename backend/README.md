# How the ML part is implemented?
Check the [`viz.ipynb`](viz.ipynb) notebook for a step-by-step exploration of the data and the design of the machine learning approach.

As a overview, we will exploit the following domain knowledge to classify the animals:

```mermaid
quadrantChart
    title Features that could be used for the classification
    x-axis 2 legs --> 4 legs
    y-axis Small size --> Large size
    quadrant-1 Elephant
    quadrant-2 Kangaroo
    quadrant-3 Chicken
    quadrant-4 Dog
```

For an actual implementation of the machine learning model, check the [`ml.ipynb`](ml.ipynb) notebook.

With this complete, let's implement the backend using the trained model.