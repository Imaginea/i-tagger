# Models

[Tensorflow Estimators](https://www.tensorflow.org/programmers_guide/estimators) are used.

- Each model will have its own config handler
     - Which takes in user model parameter
     - Stores the config in model directory
     - Retrieves the model, when the model directory is given
- Each model inherits the estimator interaface and an feature type