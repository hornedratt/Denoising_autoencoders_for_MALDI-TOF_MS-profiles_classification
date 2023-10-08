import torch
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

from src.data.CustomDataSet import CustomDataSet

def cross_valid(data_path: str,
                model_path: str,
                output_path_csv: str,
                output_path_hist: str,
                train_size: float=0.7,
                amoumt: int=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    valid_set = pd.read_csv(data_path, sep=';')
    valid_set = CustomDataSet(valid_set.drop('group', axis=1).drop('ID', axis=1).to_numpy(dtype=float),
                              valid_set['group'],
                              valid_set['ID'])

    autoencoder = torch.load(model_path)
    # valid_loader = DataLoader(valid_set, batch_size=len(valid_set))
    valid_set.profile = autoencoder(valid_set.profile)
    accuracies_ID = []
    accuracies_group = []
    for i in tqdm(range(amoumt)):
        train_value, test_value = train_test_split(valid_set,
                                                   train_size=train_size,
                                                   shuffle=True)
        classifier_group = RandomForestClassifier()
        classifier_ID = RandomForestClassifier()

#       тренируем очередной лес
        embaddings, group, id = train_value[:]
        classifier_group.fit(embaddings, group)
        classifier_ID.fit(embaddings, id)

#       тестируем полученный лес
        embaddings, group, id = test_value[:]
        pred_ID = classifier_ID.predict(embaddings)
        pred_group = classifier_group.predict(embaddings)
        accuracies_ID.append(accuracy_score(id, pred_ID))
        accuracies_group.append(accuracy_score(group, pred_group))
    accuracies_ID = pd.DataFrame(accuracies_ID)
    accuracies_group = pd.DataFrame(accuracies_group)
    result = pd.concat([accuracies_group, accuracies_ID], axis=0)
    result.index = ['group', 'ID']

    accuracies_ID.to_csv(output_path_csv, sep=';', header=True, index=True)

    df = accuracies_group.to_numpy()
    fig = px.histogram(df, marginal="box", \
                       labels={'count': 'amount of trains', 'value': 'accuracy'}, hover_data=df.transpose())
    fig.update_layout(showlegend=False)
    fig.show()

    fig.write_image(output_path_hist)
    return None
# if __name__ == "__main__":
#     cross_valid()
cross_valid(os.path.join("..", "..", "data\\processed\\sets\\test_set_normal_noise_40%.csv"),
            os.path.join("..", "..", "models\\old_models\\DAE_norm_noise_40%.pkl"),
            os.path.join("..", "..", "reports\\cross_valid_40%_result.csv"),
            os.path.join("..", "..", "reports\\figures\\cross_valid_40%_result.png"))
