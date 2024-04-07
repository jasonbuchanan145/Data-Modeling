import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, log_loss
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def scatter(variable, target, df, ax):
    #Show how many people are in each group, helps especially with binary classifications
    group_sizes = df.groupby([variable, target]).size().reset_index(name='counts')
    dfMerged = df.merge(group_sizes, on=[variable, target])
    grouped = dfMerged.groupby(target)
    for key, group in grouped:
        #This outputs each bubble as a different size, which is ok, but really doesn't show the difference between
        #small differences
        ax.scatter(group[variable], group[target], s=group['counts'] * 10, label=key)
        #Prints the number ontop of the bubble
        for i, row in group.iterrows():
            ax.text(row[variable], row[target], str(row['counts']),
                    ha='center', va='center', color='red')

    plt.xlabel(variable)
    plt.ylabel(target)
    plt.show()


def confusionAndAccuracy(target, df, plt):
    x_train, x_test, y_train, y_test = train_test_split(df.drop('diagnosis', axis=1), df[target], test_size=0.3,random_state=52)


    #Results are heavily bias towards a negative diagnoses, force the weights to be balanced so that column 2 has something
    model = LogisticRegression(class_weight='balanced')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_predict)
    print(matrix)
    getAccuracy(matrix)
    precision=getPrecision(matrix)
    recall = getRecall(matrix)
    getF1(precision,recall)

    #ROC
    y_true = np.array(y_test)
    y_scores = model.predict_proba(x_test)[:, 1]
    false_positive, true_positive,thresholds = roc_curve(y_true,y_scores, pos_label='O')
    plt.plot(false_positive,true_positive)
    plt.show()

    print("AUC: "+str(auc(false_positive,true_positive)))
    print("Log Loss: "+str(log_loss(y_true,y_scores)))


# https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
# For most of these
def getAccuracy(matrix):
    tp = matrix[1, 1]
    tn = matrix[0, 0]
    fp = matrix[0, 1]
    fn = matrix[1, 0]
    accuracy = (tp + tn)/float(tp + tn + fp + fn)
    print("Accuracy: "+str(accuracy))

def getRecall(matrix):
    tp = matrix[1,1]
    fn = matrix[1,0]
    actual_yes = float(tp+fn)
    recall = float(tp/actual_yes)
    print("Recall: "+str(recall))
    return recall

def getPrecision(matrix):
    tp = matrix[1, 1]
    predYes = tp + matrix[0, 1]
    precision=float(tp / predYes)
    print("Precision: "+str(precision))
    return precision

#implemented from https://en.wikipedia.org/w/index.php?title=F-score#Definition
def getF1(precision,recall):
    f1 = 2*((precision*recall)/(precision+recall))
    print("F1:"+str(f1))
    return f1


if __name__ == "__main__":
    target = 'diagnosis'
    fertility = fetch_ucirepo(id=244)
    X = fertility.data.features
    Y = fertility.data.targets
    df = pd.DataFrame(X)
    df[target] = Y
    variable = 'hrs_sitting'

    fig, ax = plt.subplots()
    scatter(variable,target,df,ax)
    confusionAndAccuracy(target, df, plt)
