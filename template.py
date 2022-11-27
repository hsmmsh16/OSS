#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/hsmmsh16/OSS/blob/main/template.py

import sys
import pandas as pd
import sklearn.metrics as mt

def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    return dataset

def dataset_stat(dataset_df):	
    feats = dataset_df.shape[1]
    # class0 = data_df.loc['class 0']
    # class1 = data_df.loc['class 1']
    n_class0 = dataset_df[0].value_counts()
    n_class1 = dataset_df[1].value_counts()
    return feats, n_class0, n_class1

def split_dataset(dataset_df, testset_size):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(dataset_df.data, dataset_df.target, testset_size)
    return X_train, X_test, Y_train, Y_test
    
def decision_tree_train_test(x_train, x_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(x_train, y_train)
    y_pred = dt_cls.predict(x_test)
    acc = mt.accuracy_score(y_test, y_pred)
    prec = mt.precision_score(y_test, y_pred)
    recall = mt.recall_score(y_test, y_pred)
    return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    rf_cls = RandomForestClassifier()
    rf_cls.fit(x_train, y_train)
    y_pred = rf_cls.predict(x_test)
    acc = mt.accuracy_score(y_test, y_pred)
    prec = mt.precision_score(y_test, y_pred)
    recall = mt.recall_score(y_test, y_pred)
    return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
    from sklearn.svm import SVC
    svm_cls = SVC()
    svm_cls(x_train, y_train)
    y_pred = svm_cls.predict(x_test)
    acc = mt.accuracy_score(y_test, y_pred)
    prec = mt.precision_score(y_test, y_pred)
    recall = mt.recall_score(y_test, y_pred)
    return acc, prec, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
