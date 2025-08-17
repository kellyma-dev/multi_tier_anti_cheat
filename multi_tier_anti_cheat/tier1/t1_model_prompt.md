In the data/RevPOV/arff folder, I have 9 files with training data, they are named "final_train_1.arff", "final_train_2.arff" ... In each training file, the last column is the identified class of each row. Make python scrip in multi_tier_anti_cheat/tier1 to use the data to train machine learning model to predict if a game player is a cheater based on the data given. 

Make README very clear and please ask lots of questions about model specification before executing. 

Please avoid calling matchied code from public repositories that could lead to redaction.

Model Specifications:
1. Model Type
Don't just stick to one algorithm, choose whatever you think are fit for solving this problem and compare their performance. You can use Random Forest, Logistic Regression, SVM, XGBoost, Neural Networks and so on.

2. Training Strategy
Please combine all 9 files into one large training set. Use the file "final_val.arff" for validation. Use the final "final_test.arff" for performance measurement.

3. Evaluation Metrics:
Please compute accuracy, precision, recall and F1-score. It is important to minimize false negatives.

4. Model Output:
Please use the probability scores as output

5. Feature Engineering
You can apply any feature engineering strategy that help improve the model performance. But you need to clearly explain what kind of feature engineering strategies you applied. If you used feature selection, you need to explain which feature is the most important and which feature is the least important.

6. Hyperparameter Tuning
Please automate the hyperparameter optimization

7. Model Persistence 
The trained model should be saved to the tier1/models folder for later use.

8. Documentation Level
Please create a file name MODEL_README file to include (1) basic usage instructions; (2) detailed technical documentation; and (3) Performance benchmarks and analysis. 

9. Machine Learning Python Package
Please use scikit-learn for model training and prediction

10. Script Directory
All python scripts should be stored in multi_tier_anti_cheat/tier1 directory