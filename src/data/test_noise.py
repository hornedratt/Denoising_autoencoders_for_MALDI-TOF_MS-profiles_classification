import os.path

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
import click

@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.option("--noise", default=40, type=int)
@click.option("--amount-additional-profiles", default=200, type=int)
def test_noise(input_path: str,
                    output_path: str,
                    noise: int = 40,
                    amount_additional_profiles: int = 200):
    """Делает зашумленный validset, таким же образом как и в trainloop: генирируем 12000(размерность
    профилей) слуйчайных величин из нормального распределения с нулевым средним и дисперсией, как в
    рассматриваемом векторе и этот ветор, домноженный на необходимый процент шума, прибавим к рассматриваемому
    вектору
    :param input_path: путь до папки, в которой лежаит csv с оригинальным набором профилей
    :param output_path: путь, куда сохраним сгенирированный csv
    :param noise: константа на которую домножаем сгенерированный вектор
    :param amount_additional_profiles: сколько зашумленных векторов сделаем из каждого оригинального
    :return: None
    """

#   считали как DataFrame чтобы имена колонок
    original_profiles = pd.read_csv(input_path, sep=';')
    noise_factor = noise/100
    final = pd.DataFrame({k: pd.Series(dtype=float) for k in original_profiles.columns})
    for i in range(len(original_profiles)):
        main = original_profiles.loc[i].T
        num = main[:'15000.0'].to_numpy(dtype=float)
        final_tmp = np.array([num])
        for j in range(amount_additional_profiles):
            tmp = num.copy()
            tmp = tmp + np.random.normal(loc=0,
                                         scale=noise_factor * tmp,
                                         size=(len(tmp)))
            tmp = abs(tmp)
            np.place(tmp, tmp > 1, 1)
            final_tmp = np.append(final_tmp, np.array([tmp]), axis=0)
        final_tmp_pd = pd.DataFrame(final_tmp)
        final_tmp_pd['group'] = main['group']
        final_tmp_pd['ID'] = main['ID']
        final_tmp_pd.columns = original_profiles.columns
        final = pd.concat([final, final_tmp_pd], axis=0)
    final.to_csv(output_path, sep=';', header=True, index=True)

if __name__ == "__main__":
    test_noise()

# test_noise(os.path.join("..", "..", "data\\processed\\original_MS_profiles.csv"),
#            os.path.join("..", "..", "data\\processed\\sets\\test_set_normal_noise_40%.csv"))
