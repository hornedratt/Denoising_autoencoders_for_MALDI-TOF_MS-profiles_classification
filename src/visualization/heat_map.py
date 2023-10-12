import torch
import numpy as np
import pandas as pd
import matplotlib as plt
import click

from src.data.CustomDataSet import CustomDataSet

@click.command()
@click.argument("model_path", type=click.Path())
@click.argument("data_path", type=click.Path())
@click.argument("output_path_group", type=click.Path())
@click.argument("output_path_id", type=click.Path())
def heat_map(model_path: str,
             data_path: str,
             output_path_group: str,
             output_path_id: str):
    """Строит heatmap для эмбеддингов dataset'а с определенным упрвнем шума с использованием соответствующего
    автоенкодера. Каждая точка в heatmap - евклидово расстояние между среднегрупповыми/среднештаммовыми расстояниями
    :param model_path: откуда берем автоенкодер
    :param data_path: откуда берем сэт
    :param output_path_group: куда сохраним heatmap для среднегрупповых расстояний
    :param output_path_ID: куда сохраним heatmap для среднештаммовых расстояний
    """
    def euclid(x):
        x = x ** 2
        l = np.sum(x)
        l = np.sqrt(l)
        return l

    device = torch.device('cpu')

    valid_set = pd.read_csv(data_path, sep=';')
    valid_set = CustomDataSet(valid_set.drop('group', axis=1).drop('ID', axis=1).to_numpy(dtype=float),
                              valid_set['group'],
                              valid_set['ID'])

    autoencoder = torch.load(model_path).to(device)
    valid_set.profile = autoencoder(valid_set.profile)

    embaddings_heat = pd.DataFrame(valid_set.profile.numpy(), dtype=float)
    embaddings_heat['group'] = valid_set.group
    embaddings_heat['ID'] = valid_set.name
    for attribute in ['group', 'ID']:
        embaddings_heat_mean = embaddings_heat.groupby(attribute).mean()
        heat_map = np.zeros((len(embaddings_heat_mean.index), len(embaddings_heat_mean.index)))
        heat_map = pd.DataFrame(heat_map, index=embaddings_heat_mean.index,
                                columns=embaddings_heat_mean.index, dtype=float)
        for i in embaddings_heat_mean.index:
#           для каждой группы ищем средне-групповое расстояние
#           (ищем разность каждого ветора с каждым и делим на количество разностей)
            mean = embaddings_heat.loc[embaddings_heat[attribute] == i].drop(columns=['group', 'ID'], axis=1).to_numpy()
            s = np.zeros((len(mean[0])))
            count = 0
            for j in range(len(mean[:, 0])):
                for k in range(j + 1, len(mean[:, 0])):
                    s = s + mean[j] - mean[k]
                    count = count + 1
            s = s / count
            e = euclid(s)
            heat_map.at[i, i] = e

            # ищем расстояние до средних векторов остальных групп mylist.index(element)
            for j in embaddings_heat_mean.index.values[np.where(embaddings_heat_mean.index.values == i)[0][0]:]:
                s = embaddings_heat_mean.loc[i].to_numpy() - embaddings_heat_mean.loc[j].to_numpy()
                e = euclid(s)
                heat_map.at[i, j] = e

        heat_map_t = heat_map.values.copy()
        for i in range(len(heat_map['Bacillus_licheniformis'])):
            heat_map_t[i, i] = 0
        heat_map_t = heat_map_t.transpose()
        heat_map.loc[:, :] = heat_map.values + heat_map_t

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xticks(np.arange(heat_map.shape[1]))
        ax.set_yticks(np.arange(heat_map.shape[0]))
        ax.set_xticklabels(heat_map)
        ax.set_yticklabels(heat_map)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        if attribute == 'group':
            ax.set_title("карта расстояний между группами культур")
            plt.savefig(output_path_group)
        if attribute == 'ID':
            ax.set_title("карта расстояний между штаммами культур")
            plt.savefig(output_path_id)


if __name__ == "__main__":
    heat_map()