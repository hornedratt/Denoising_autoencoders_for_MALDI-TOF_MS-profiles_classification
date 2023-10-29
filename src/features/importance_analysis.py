import torch
import pandas as pd
import numpy as np
import pickle
import click
import matplotlib.pyplot as plt
import os

from src.data.CustomDataSet import CustomDataSet
@click.command()
@click.argument("data_path", type=click.Path())
@click.argument("model_path", type=click.Path())
@click.argument("forest_group_path", type=click.Path())
@click.argument("forest_id_path", type=click.Path())
@click.argument("output_path_forest_group_importances", type=click.Path())
@click.argument("output_path_forest_id_importances", type=click.Path())
@click.argument("output_path_group_mz_features", type=click.Path())
@click.argument("output_path_id_mz_features", type=click.Path())
def importance_analysis(model_path: str,
                        data_path: str,
                        forest_group_path: str,
                        forest_id_path: str,
                        output_path_forest_group_importances: str,
                        output_path_forest_id_importances: str,
                        output_path_group_mz_features: str,
                        output_path_id_mz_features: str
                        ):
    """Поиск важных для класификации фичей (классификация и по группам, и по штаммам).
     Сначала ищем важные фичи в скрытом сотоянии (через индекс gini, берем топ 10) потом нахожу максималные веса, ведущие в
     соответствующие этим фичам нейроны между внутренними слоями. На первом слое, среди весов,
     ищу выбросы по формуле из статьи nature (см. отчет)
    :param model_path: откуда берем автоенкодер
    :param data_path: откуда берем сэт
    :param forest_group_path: откуда берем лес для групп
    :param forest_id_path: откуда берем лес для штаммов
    :param output_path_forest_group_importances: куда сохраним график "важностей" каждой из фичей в лесу для групп
    :param output_path_forest_id_importances: куда сохраним график "важностей" каждой из фичей в лесу для штаммов
    :param output_path_group_mz_features: куда сохраним важные, для классификации по группам, пики в исходных профилях
    :param output_path_шв_mz_features: куда сохраним важные, для классификации по штаммам, пики в исходных профилях
    :return:
    """
    attributes = ['group', 'ID']
    paths = pd.DataFrame([[forest_group_path, output_path_forest_group_importances, output_path_group_mz_features],
                          [forest_id_path, output_path_forest_id_importances, output_path_id_mz_features]],
                         index=attributes,
                         columns=['forest', 'importances', 'mz_features'])

    for attribute in attributes:
        with open(paths.at[attribute, 'forest'], 'rb') as f:
            classifier = pickle.load(f)

        device = torch.device('cpu')

        with open(data_path, 'rb') as file:
            valid_set = pickle.load(file)

        autoencoder = torch.load(model_path).to(device)
        valid_set.profile = autoencoder(valid_set.profile)

        importances = classifier.feature_importances_
        forest_importances = pd.Series(importances, index={i for i in range(50)})

        std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
                     axis=0)  # deviation - мера отклонения

        fig, ax = plt.subplots(figsize=(10, 10))
        forest_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        plt.setp(ax.get_xticklabels(), rotation=90, rotation_mode="anchor")
        fig.tight_layout()
        plt.savefig(paths.at[attribute, 'importances'])

        top = forest_importances.nlargest(10)
        max_index = np.zeros(len(top.index))  # массив для хранения индексов макимальных весов

        # ищем максимумы между 3 и 4 слоями енкодера
        for j, i in enumerate(top.index):
            maximum, max_index[j] = torch.max(autoencoder[4].weight[i], dim=0)

        # ищем максимумы между 2 и 3 слоями енкодера
        for j in range(len(max_index)):
            maximum, max_index[j] = torch.max(autoencoder[2].weight[int(max_index[j])], dim=0)

        # ищем максимумы между 1 и 2 слоями енкодера
        mz_features = np.array([], dtype=int)
        for j in range(len(max_index)):
            current = (autoencoder[0]).weight[int(max_index[j])]
            # beta - гиперпараметр, его подобрали так, чтобы метод отбирал необходимое количество фичей
            beta = 5
            T = torch.mean(current) + beta * torch.std(current)
            for i in range(len(autoencoder[0].weight[0])):
                if (autoencoder[0]).weight[int(max_index[j]), i] >= float(T):
                    mz_features = np.append(mz_features, i)
        np.savetxt(paths.at[attribute, 'mz_features'], mz_features, fmt='%d')
        # return None

if __name__ == "__main__":
    importance_analysis()

# importance_analysis(os.path.join("..", "..", "models\\DAE_norm_noise_40%.pkl"),
#                     os.path.join("..", "..", "data\\processed\\sets\\test_set_normal_noise_40%.csv"),
#                     os.path.join("..", "..", "models\\forest_40%_group"),
#                     os.path.join("..", "..", "models\\forest_40%_ID"),
#                     os.path.join("..", "..", "reports\\figures\\forest_40%_importances_group.png"),
#                     os.path.join("..", "..", "reports\\figures\\forest_40%_importances_ID.png"),
#                     os.path.join("..", "..", "reports\\mz_features_40%_group.txt"),
#                     os.path.join("..", "..", "reports\\mz_features_40%_ID.txt"))