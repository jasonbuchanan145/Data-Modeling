from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
from sklearn.model_selection import KFold
import numpy as np
import statistics

if __name__ == "__main__":
    iris = datasets.load_iris()
    num_features = iris.data.shape[1]
    print("Number of features:", num_features)
    feature_names = iris.feature_names
    print("Feature names:", feature_names)

    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
    ax.set(xlabel=iris.feature_names[2], ylabel=iris.feature_names[3])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
        )
    #plt.show()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df = df.replace({"species":  {"setosa":1,"versicolor":2, "virginica":3}})
    normalized_df = df[iris['feature_names']].apply(zscore)

    X = iris.data  # feature vectors
    y = iris.target  # labels

    # Split the dataset into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Create an SVM classifier with a radial basis function (RBF) kernel
    classifier = SVC(kernel = 'linear', random_state = 0)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)
#    print(classification_report(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))
    # Looking at the source for sklearn/svm/_classes.py the options for the kernel are
    # kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,
    # {'scale', 'auto'} or float, default='scale'
    # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    #if ``gamma='scale'`` (default) is passed then it uses
    #1 / (n_features * X.var()) as value of gamma,
    #if 'auto', uses 1 / n_features
    #if float, must be non-negative.
    kernelOptions = list(("linear", "poly", "rbf", "sigmoid"))
    gamaOptions = list(("scale","auto"))
    results = list()
    for gama in gamaOptions:
        for kernel in kernelOptions:
            svm_classifier = make_pipeline(StandardScaler(), SVC(kernel=kernel, gamma=gama))
            kf = KFold(n_splits=5, shuffle=True)
            accuracy = list()
            f1_score_result = list()
            recall_score_result = list()
            precision_score_result = list()
            for train_index, test_index in kf.split(range(len(X))):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                print(gama,kernel)
                svm_classifier.fit(X_train, y_train)
                y_pred = svm_classifier.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                f1_score_result.append(f1_score(y_test, y_pred, average='macro'))
                recall_score_result.append(recall_score(y_test, y_pred, average='macro'))
                precision_score_result.append(precision_score(y_test, y_pred, average='macro'))
                accuracy.append(score)
            acc_avg = statistics.mean(accuracy)
            f1_score_avg = statistics.mean(f1_score_result)
            recall_score_avg = statistics.mean(recall_score_result)
            precision_score_avg = statistics.mean(precision_score_result)
            result = {"gama":gama,"kernel":kernel,"accuracy":acc_avg,"f1_score":f1_score_avg,
            "recall":recall_score_avg, "precision":precision_score_avg, "sum":acc_avg+f1_score_avg
                                                                              +precision_score_avg
                                                                              +recall_score_avg}
            results.append(result)

    sorted_table = pd.DataFrame(results).sort_values(by='sum', ascending=False)
    print(sorted_table)
#    print(normalized_df.head())
    # Add a column for the target species, using the target_names for labeling
    if df.isnull().any().any():
        print("There are missing values in the dataset.")
        print(df.isnull().sum())  # This will print the count of missing values per column
    else:
        print("There are no missing values in the dataset.")
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    # Print the DataFrame
    print(df)