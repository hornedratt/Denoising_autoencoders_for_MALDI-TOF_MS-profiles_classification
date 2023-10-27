import torch
import pandas as pd
import click
import progressbar as pb
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.io as pio


from src.data.CustomDataSet import CustomDataSet
@click.command()
@click.argument("data_path", type=click.Path())
@click.argument("model_path", type=click.Path())
@click.argument("output_path_csv", type=click.Path())
@click.argument("output_path_hist_g", type=click.Path())
@click.argument("output_path_hist_i", type=click.Path())
@click.option("--train_size", default=0.7, type=float)
@click.option("--amount", default=10, type=int)
def cross_valid(data_path: str,
                model_path: str,
                output_path_csv: str,
                output_path_hist_g: str,
                output_path_hist_i: str,
                train_size: float=0.7,
                amount: int=10):
    """Кросс-валидация
    :param data_path: путь до сэта на котором будем делать кросс-валидацию
    :param model_path: путь до кодера,который будем использовать для получения скрытых
     состояний
    :param output_path_csv: путь куда сохраним результаты
    :param output_path_hist: путь куда сохраним графики
    :param train_size: объем выборки для тренировки
    :param amoumt: количество итераций в кросс-валидации (сколько раз тренируем новую
     модель классификации)
    """
    device = torch.device('cpu')

    valid_set = pd.read_csv(data_path, sep=';')
    valid_set = CustomDataSet(valid_set.drop('group', axis=1).drop('ID', axis=1).to_numpy(dtype=float),
                              valid_set['group'],
                              valid_set['ID'])

    autoencoder = torch.load(model_path).to(device)
    valid_set.profile = autoencoder(valid_set.profile)
    accuracies_ID = []
    accuracies_group = []
    for i in pb.progressbar(range(amount)):
        train_idx, test_idx = train_test_split(list(range(len(valid_set))),
                                                   train_size=train_size,
                                                   shuffle=True)
        classifier_group = RandomForestClassifier()
        classifier_ID = RandomForestClassifier()

#       тренируем очередной лес
        embaddings, group, id = valid_set.subset(train_idx)
        with torch.no_grad():
            embaddings = embaddings.numpy()
        classifier_group.fit(embaddings, group)
        classifier_ID.fit(embaddings, id)

#       тестируем полученный лес
        embaddings, group, id = valid_set.subset(test_idx)
        with torch.no_grad():
            embaddings = embaddings.numpy()
        pred_ID = classifier_ID.predict(embaddings)
        pred_group = classifier_group.predict(embaddings)
        accuracies_ID.append(accuracy_score(id, pred_ID))
        accuracies_group.append(accuracy_score(group, pred_group))
    accuracies_ID = pd.DataFrame(accuracies_ID).T
    accuracies_group = pd.DataFrame(accuracies_group).T
    result = pd.concat([accuracies_group, accuracies_ID], axis=0)
    result.index = ['group', 'ID']

    result.to_csv(output_path_csv, sep=';', header=True, index=True)

    df = accuracies_group.to_numpy()
    fig = px.histogram(df,
                       title='accuracy group',
                       marginal='box',
                       labels={'count': 'amount of trains', 'value': 'accuracy'},
                       color_discrete_sequence=['indianred']
                       )
    fig.update_layout(showlegend=False)
    img_bytes = pio.to_image(fig, format="png")
    with open(output_path_hist_g, "wb") as f:
        f.write(img_bytes)

    df = accuracies_ID.to_numpy()
    fig = px.histogram(df,
                       title='accuracy ID',
                       marginal='box',
                       labels={'count': 'amount of trains', 'value': 'accuracy'},
                       color_discrete_sequence=['indianred']
                       )
    fig.update_layout(showlegend=False)
    img_bytes = pio.to_image(fig, format="png")
    with open(output_path_hist_i, "wb") as f:
        f.write(img_bytes)

    # return None

if __name__ == "__main__":
    cross_valid()

# cross_valid(os.path.join("..", "..", "data\\processed\\sets\\test_set_normal_noise_40%.csv"),
#             os.path.join("..", "..", "models\\DAE_norm_noise_40%.pkl"),
#             os.path.join("..", "..", "reports\\cross_valid_40%_result.csv"),
#             os.path.join("..", "..", "reports\\figures\\cross_valid_40%_result_group.png"),
#             os.path.join("..", "..", "reports\\figures\\cross_valid_40%_result_id.png")
#             )
