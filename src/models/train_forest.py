import torch
import pandas as pd
import click
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.data.CustomDataSet import CustomDataSet

@click.command()
@click.argument("model_path", type=click.Path())
@click.argument("data_path", type=click.Path())
@click.argument("output_path_group", type=click.Path())
@click.argument("output_path_group_rep", type=click.Path())
@click.argument("output_path_id", type=click.Path())
@click.argument("output_path_id_rep", type=click.Path())
def train_forest(model_path: str,
                 data_path: str,
                 output_path_group: str,
                 output_path_group_rep: str,
                 output_path_id: str,
                 output_path_id_rep: str):
    device = torch.device('cpu')
    train_set = pd.read_csv(data_path, sep=';')
    train_set = CustomDataSet(train_set.drop('group', axis=1).drop('ID', axis=1).to_numpy(dtype=float),
                              train_set['group'],
                              train_set['ID'])
    size = len(train_set)

    autoencoder = torch.load(model_path).to(device)
    train_set.profile = autoencoder(train_set.profile).detach().numpy()

    idx_train, idx_test = train_test_split(list(range(size)), train_size=0.7)
    x_train, y_train_group, y_train_id = train_set.subset(idx_train)
    x_test, y_test_group, y_test_id = train_set.subset(idx_test)

    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train_group)
    y_pred = classifier.predict(x_test)
    classification_report_group = classification_report(y_test_group,
                                                        y_pred,
                                                        output_dict=True)
    classification_report_group = pd.DataFrame(classification_report_group).T
    classification_report_group.to_csv(output_path_group_rep)
    with open(output_path_group, 'wb') as f:
        pickle.dump(classifier, f)

    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train_id)
    y_pred = classifier.predict(x_test)
    classification_report_id = classification_report(y_test_id,
                                                        y_pred,
                                                        output_dict=True)
    classification_report_id = pd.DataFrame(classification_report_id).T
    classification_report_id.to_csv(output_path_id_rep)
    with open(output_path_id, 'wb') as f:
        pickle.dump(classifier, f)

if __name__ == "__main__":
    train_forest()

# train_forest("..\\..\\models\\DAE_norm_noise_40%.pkl",
#             "..\\..\\data\\processed\\sets\\test_set_normal_noise_40%.csv",
#             "..\\..\\models\\forest_40%_group",
#             "..\\..\\reports\\forest_40%_group.csv",
#             "..\\..\\models\\forest_40%_ID",
#             "..\\..\\reports\\forest_40%_ID.csv",
#              )