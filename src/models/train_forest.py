def train_forest() -> None:
    if (attribute != 'group') or (attribute != 'ID'):
        print('Error: unexpectable attribute')
        return None
    x_train, x_test, y_train, y_test = train_test_split(self.embaddings[:]['profile'],
                                                        self.embaddings[:][attribute], train_size=0.7)
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    if attribute == 'group':
        self.classification_matrix_group = confusion_matrix(y_test, y_pred)
        self.classification_report_group = classification_report(y_test, y_pred)

        self.forest_group = classifier
    else:
        self.classification_matrix_ID = confusion_matrix(y_test, y_pred)
        self.classification_report_ID = classification_report(y_test, y_pred)

        self.forest_ID = classifier
    return None