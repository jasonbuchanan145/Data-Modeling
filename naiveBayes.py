import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, log_loss
from sklearn.naive_bayes import GaussianNB

sns.set_style("darkgrid")

def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior

def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    return p_x_given_y

def naive_bayes_gaussian(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)

def testGuause(data):
    train, test = train_test_split(data, test_size=.3, random_state=41)
    X_train = train.iloc[:, :-1].values
    Y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    Y_test = test.iloc[:, -1].values
    Y_pred = naive_bayes_gaussian(train, X=X_test, Y="diagnosis")
    matrix = confusion_matrix(Y_test, Y_pred)


    #Copy paste from the fertilityModeling.py file from the other part of the assignment
    y_true = np.array(Y_test)

    model = GaussianNB()
    model.fit(X_train, Y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    false_positive, true_positive, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    plt.plot(false_positive, true_positive)
    plt.show()

    print(matrix)
    getPrecision(matrix)
    print("FScore:"+str(f1_score(Y_test, Y_pred)))
    getAccuracy(matrix)
    getRecall(matrix)

    print("AUC: " + str(auc(false_positive, true_positive)))
    print("Log Loss: " + str(log_loss(y_true, y_scores)))

#Copy from fertility project
def getAccuracy(matrix):
    tp = matrix[1, 1]
    tn = matrix[0, 0]
    fp = matrix[0, 1]
    fn = matrix[1, 0]
    accuracy = (tp + tn) / float(tp + tn + fp + fn)
    print("Accuracy: "+str(accuracy))

def getRecall(matrix):
    tp = matrix[1,1]
    fn = matrix[1,0]
    actual_yes = float(tp+fn)
    recall = float(tp/actual_yes)
    print("Recall: "+str(recall))
    return recall



def getPrecision(matrix):
    TP = matrix[1, 1]
    predYes = TP + matrix[0, 1]
    precision=float(TP / predYes)
    print("Precision: "+str(precision))
    return precision

if __name__=="__main__":
    data = pd.read_csv("./BreastCancerData.csv")
    data.head(10)
    data["diagnosis"].hist()
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    corr = data.iloc[:, :-1].corr(method="pearson")
    cmap = sns.diverging_palette(250, 354, 80, 60, center='dark', as_cmap=True)
    sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidths=.2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    sns.histplot(data, ax=axes[0], x="mean_radius", kde=True, color='r')
    sns.histplot(data, ax=axes[1], x="mean_smoothness", kde=True, color='b')
    sns.histplot(data, ax=axes[2], x="mean_texture", kde=True)
    testGuause(data)
