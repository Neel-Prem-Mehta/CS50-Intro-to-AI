import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    document = open(filename, "r")

    text = document.readlines()
    text = text[1:]

    data = []
    for shopper in text:
        data.append(shopper.split(","))

    evidence = []
    labels = []
    
    for shopper in data:
        temp = []
        for i in range(len(shopper)-1):
            if i in [0, 2, 4, 11, 12, 13, 14]:
                temp.append(int(shopper[i]))
            if i in [1, 3, 5, 6, 7, 8, 9]:
                temp.append(float(shopper[i]))
            if i == 10:
                temp.append(month_num(shopper[i]))
            if i == 15:
                if shopper[i] == "Returning_Visitor":
                    temp.append(1)
                else:
                    temp.append(0)
            if i == 16:
                if shopper[i][0] == "T":
                    temp.append(1)
                else:
                    temp.append(0)
        evidence.append(temp)

        if shopper[17][0] == "T":
            labels.append(1)
        else:
            labels.append(0)
    
    return (evidence, labels)


def month_num(month):
    master = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return master.index(month)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sens_numer = 0.0
    sens_denom = 0.0
    spec_numer = 0.0
    spec_denom = 0.0
    
    for x in range(len(labels)):
        #print(labels[x])
        if labels[x] == 1:
            sens_denom += 1
            if predictions[x] == 1:
                sens_numer += 1
        if labels[x] == 0:
            spec_denom += 1
            if predictions[x] == 0:
                spec_numer += 1

    sensitivity = 1
    specification = 1
    if sens_denom != 0:
        sensitivity = sens_numer/sens_denom
    if spec_denom != 0:
        specification = spec_numer/spec_denom

    print(sens_denom)
    print(spec_denom)
    
    return (sensitivity, specification)


if __name__ == "__main__":
    main()
