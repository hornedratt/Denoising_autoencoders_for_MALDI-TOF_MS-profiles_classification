import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from src.data.CustomDataSet import CustomDataSet
def importance_analysis(model_path: str,
                        data_path: str,
                        forest_group_path: str,
                        forest_id_path: str,
                        output_path_forest_group_importances: str,
                        noise: int = 40) -> None:
    with open(forest_group_path, 'rb') as f:
        classifier_group = pickle.load(f)
    with open(forest_id_path, 'rb') as f:
        classifier_id = pickle.load(f)

    device = torch.device('cpu')

    valid_set = pd.read_csv(data_path, sep=';')
    valid_set = CustomDataSet(valid_set.drop('group', axis=1).drop('ID', axis=1).to_numpy(dtype=float),
                              valid_set['group'],
                              valid_set['ID'])

    autoencoder = torch.load(model_path).to(device)
    valid_set.profile = autoencoder(valid_set.profile)

    importances = classifier_group.feature_importances_
    forest_importances = pd.Series(importances, index={i for i in range(50)})

    std = np.std([tree.feature_importances_ for tree in classifier_group.estimators_],
                 axis=0)  # deviation - мера отклонения

    fig, ax = plt.subplots(figsize=(10, 10))
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    plt.setp(ax.get_xticklabels(), rotation=90, rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig(output_path_forest_group_importances)

    top = forest_importances.nlargest(10)
    J = np.zeros(len(top.index))  # массив для хранения индексов макимальных весов
    # ищем максимумы между 3 и 4 слоями енкодера
    for j, i in enumerate(top.index):
        maximum, J[j] = torch.max(autoencoder_inference[0][4].weight[i], dim=0)
    # ищем максимумы между 2 и 3 слоями енкодера
    for j in range(len(J)):
        maximum, J[j] = torch.max((autoencoder_inference[0][2]).weight[int(J[j])], dim=0)
    # ищем максимумы между 1 и 2 слоями енкодера
    mz_features = np.array([], dtype=int)
    for j in range(len(J)):
        current = (autoencoder_inference[0][0]).weight[int(J[j])]
        # beta = random.uniform(1, 2.5)
        beta = 5
        T = torch.mean(current) + beta * torch.std(current)
        for i in range(len((autoencoder_inference[0][0]).weight[0])):
            if (autoencoder_inference[0][0]).weight[int(J[j]), i] >= float(T):
                mz_features = np.append(mz_features, i)
    np.savetxt('mz_features_' + attribute + '_' + str(noise) + '%_txt', mz_features, fmt='%d')
    return None