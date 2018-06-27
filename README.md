# Amazon-review-sentiment-prediction
*CS 155 Machine Learning Data Mininig course project, Caltech, winter 2017. Equal contributor: Botao Hu, Jian Xu, Yukai Liu.*

## Overview
Use bag-of-words representations of Amazon reviews to predict the sentiments that the reviews express. The bag-of-words representation is constructed from counts of the top 1,000 words appearing in the 20,000 reviews. All reviews are classified into either "good" for 4 or 5-star reviews, or "bad" for or 2-star review. 3-star reviews are excluded from the dataset. 

The goal is given a bag-of-words representation of test reviews, the trained model should predict the emotion of the review: good or bad.

## Approach
* __XGBoost__: We used the open-source XGBoost library, and achieved the best validation accuracy 85.5%.
* __AdaBoost__: We used AdaBoostClassifier in sklearn, and achived the best validation accuracy 84.7%.
* __Logistic Regression__: We used Logistic Regression in sklearn, and achived the best validation accuracy 84.9%.
* __Neural Network__: We used the Sequential model from Keras library, and achieved best validation accuracy of 85.6%.
* __KNearest Neighbors__: We trained KNeighborsClassifier object from sklearn library, and achieved best validation accuracy of 71.4% with PCA pre-process.
* __SVM__:We used svm.SVC, resulting in a cross validation accuracy of 84.8%.
* __Random Forest__: We used RandomForestClassifier in sklearn, resulting in a cross validation accuracy of 84.7%.
* __Techniques: PCA__: We tried to implement PCA on dataset before training, but it turned out not to be helpful in the cross validation and the final testing score.
* __Techniques: Model Stacking__: We stacked all 7 models together, and used logistic regression to get final model. This technique improved the cross-validation accuracy of the best model by 0.2%.

## Usage
Explore the code for approaches, and see `./data` for results.
