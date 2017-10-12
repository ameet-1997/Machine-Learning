RDA = RegularizedDiscriminantAnalysis(alpha=alpha, gamma=gamma)
RDA.fit(train_data, train_labels)

# Predict the labels
rda_predicted = RDA.predict(test_data)
[prec, rec, fsc, sup] = precision_recall_fscore_support(test_labels, rda_predicted)
