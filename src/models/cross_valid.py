from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from src.data.CustomDataSet import CustomDataSet

def cross_valid(data_path: str,
                model_path: str,
                attribute: str = 'group',
                amoumt: int = 1000) -> None:
    valid_set = pd.read_csv(data_path, sep=';')
    valid_set = CustomDataSet(valid_set.drop('group', axis=1).drop('ID', axis=1),
                              valid_set['group'],
                              valid_set['ID'])

    autoencoder = torch.load(model_path)
    valid_loader = DataLoader(valid_set, batch_size=len(valid_set))

    embaddings = autoencoder(valid_set[0])
    for i in tqdm(range(amoumt)):
        train_value, test_value = train_test_split(self.embaddings,
                                                   train_size=0.7,
                                                   shuffle=True)

        classifier_group = RandomForestClassifier()
        classifier_ID = RandomForestClassifier()
        classifier_group.fit(train_value[:]['profile'], train_value[:]['group'])
        classifier_ID.fit(train_value[:]['profile'], train_value[:]['ID'])
        pred_ID = classifier_ID.predict(test_value[:]['profile'])
        pred_group = classifier_group.predict(test_value[:]['profile'])
        accuracies_ID.append(accuracy_score(test_value[:]['ID'], pred_ID))
        accuracies_group.append(accuracy_score(train_value[:]['group'], pred_group))

    accuracies_ID = pd.Series(accuracies_ID)
    accuracies_group = pd.Series(accuracies_group)

    accuracies_ID.to_csv(rpath_ID, sep=';', header=True, index=True, )
    accuracies_group.to_csv(rpath_group, sep=';', header=True, index=True, )

    df = accuracies_group.to_numpy()
    fig = px.histogram(df, marginal="box", \
                       labels={'count': 'amount of bootstraps', 'value': 'accuracy'}, hover_data=df.transpose())
    fig.update_layout(showlegend=False)
    fig.show()

    fig.write_image(r'C:\education\ML\reports&results\bootstrap_' + attribute + '_' + str(noise) + '%.png')
    return None