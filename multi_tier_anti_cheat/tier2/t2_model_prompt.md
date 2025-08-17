In the data/RevStats/dataset folder, I have 3 csv files used for machine learning moder training, validation and testing. In each file:
* The first row is the header
* The first column is the id of each player and this data should not be used in training and testing
* The last column is the label of each plater or the class of each play. 

Make python scrips in the folder mtac-ml/tier2 and use the data stored in the folder data/RevStats/dataset to train machine learning models to predict if a game player is a cheater. The training data are stored in the file "final_train.csv", validation data are stored in the file "final_val.csv" and please use the data stored in the file "final_test.csv" to measure the final performance of the model.  

Make README very clear and please ask lots of questions about model specification before executing. I want you to consider the following deep neural network algorithms:
* multi-layer perceptron
* TabNet
* Wide & Deep Networks

Please avoid calling matched code from public repositories that could lead to redaction.