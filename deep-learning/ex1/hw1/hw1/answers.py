r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. True, in order to estimate our in-sample error, we first train our model, and then we would need to test it over the unseen data (test set).
2. False, in order to train our model properly, we would need to reduce our error rate, therefore we would need a majority set of our train data (and not just a random split).
3. True, the test-set should not be used during cross-validation, since the cross-validation is used for determining our models' hyperparameters and the test is used for our model's evaluation.
4. False, Cross-Validation is used for selecting the model's hyperparameters and not for the model's generalization error. For the former step, we'd use the unseen dataset (test-set).
"""

part1_q2 = r"""
**Your answer:**
This approach is not justified. Performing hyperparameter tuning outside of cross-validation can lead to biased-high estimates of external validity because the dataset that is used to measure performance is the same as tuning the features.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:** Increasing k might lead to improved generalization for unseen data since the areas predicting each class will be more "smoothed", because it's the majority of the k-nearest neighbours which decide the class of any point, however, if k is selected to be too large, the model becomes too generalized and fails to accurately predict the data points in both train and test sets. This situation is known as underfitting.

Alternatively, if k is selected to be too low, the model becomes too specific and fails to generalize well. It also tends to be sensitive to noise. The model accomplishes a high accuracy on train set but will be a poor predictor on new, previously unseen data points. Therefore, we are likely to end up with an overfit model.
"""

part2_q2 = r"""
**Your answer:**
1. Evaluating model performance with the data used for training is not acceptable in data mining because it can easily generate overoptimistic and overfitted models. Additionally, we cannot check whether our model works or not by the given desired results, we should evaluate it using the unseen dataset.
2. When we have very little data, splitting it into training and test set might leave us with a very small test set (this issue might be even worse when we have a multi-class problem). Therefore, when using cross-validation in this case, we build K different models, so we are able to make predictions on all of our data.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The magnitude of the weights W has a direct effect on the scores. As we shrink all values inside W, the score differences will be lower, and as we scale up the weights, the score differences will all become higher.

Therefore, the exact value of the margin between the scores is in some sense meaningless because the weights can shrink or stretch the differences arbitrarily. 
Hence, the only tradeoff is how large we allow the weights to grow (through the regularization strength).
"""

part3_q2 = r"""
**Your answer:**
1. Since the images are stretched into high-dimensional column vectors, we can interpret each image as a single point in this space. Analogously, the entire dataset is a (labeled) set of points. Therefore, some of the classification errors might occur due to some similarities between certain points. An additional interpretation for the weights W is that each row of W corresponds to a template for one of the classes. 
The score of each class for an image is then obtained by comparing each template with the image using a dot product one by one to find the one that "fits" best. 
With this terminology, the linear classifier is doing template matching, where the templates are learned. 

2. As mentioned, the entire dataset is a labeled set of points. We can define the score of each class as a weighted sum of all image pixels. 
Each score is a linear function over this space, which is a generalized representation learned from the labeled data, that can classify our dataset by each of these lines. 
This method is different from the kNN that uses the closest similarity between the points (e.g. distance) and compares each new data with existing data and tries to seek which group claims the greatest proximity to it.
"""

part3_q3 = r"""
**Your answer:**
1. The learning rate we have chosen is quite good since we can observe that the training set loss is decreasing during the epochs and we received more than 90% accuracy. If the learning rate is too low, then training is more reliable, but optimization will take a lot of time because steps towards the minimum of the loss function are tiny, and if the learning rate is too high, then training may not converge or even diverge. Weight changes can be so big that the optimizer overshoots the minimum and makes the loss worse.

2. The model is slightly overfitting to the training set since there is a tiny percent difference in the accuracies between the training set to the test set. Nevertheless, we still received good accuracy on the test set, hence it is not highly overfitted. 
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
A residual plot is a graph that presents the residuals on the vertical axis and the independent variable on the horizontal axis. If the points in a residual plot are randomly dispersed around the horizontal axis, a linear regression model is appropriate for the data, otherwise, a nonlinear model is more appropriate.

Therefore, based on the residual plots we got above, we can assume that the fitness of our final trained model got good results since the points are closer to the zero axis than the previous model.  Additionally, it appears that our model dispersed around the zero, compared to the top-5 features that there is a particular relation between the y-predictions and the residuals. 
"""

part4_q2 = r"""
**Your answer:**
1. Yes, since nonlinear regression is also a form of regression analysis in which data is fit to a model and then expressed as a mathematical function. (i.e. simple linear regression relates two variables with a straight line, while nonlinear regression relates the two variables in a nonlinear relationship).

2. Yes, we can fit any nonlinear function of the original features by performing feature engineering and applying our linear regression model on these transformed features.

3. Adding non-linear features might get better results by attempting some nonlinear functional forms for our hyperplane. With this approach, we can classify also nonlinear points which cannot be obtained using linear features, and it still will be a hyperplane but on higher dimensions. 
"""

part4_q3 = r"""
**Your answer:**
1. Defining the range as np.logspace allows us to obtain (quickly) a more comprehensive range of values.

2. According to the implementation below:

`degree_range = np.arange(1, 4)` \
`lambda_range = np.logspace(-3, 2, base=10, num=20)` \
`best_hypers = hw1linreg.cv_best_hyperparams(model, x_train, y_train, k_folds=3, degree_range=degree_range, lambda_range=lambda_range)`

We got, degree_range = 3, lambda_range = 20, k_folds=3. \
Therefore, we fitted the model 180 (= 3 * 20 * 3) times on the data.
"""

# ==============
