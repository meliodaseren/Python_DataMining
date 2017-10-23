def classify(features_train, labels_train):

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    # fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    # pred = clf.predict(test_data)

    # return the fit classifier
    return clf
