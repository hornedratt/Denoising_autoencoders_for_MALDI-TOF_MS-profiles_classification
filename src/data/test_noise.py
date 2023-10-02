import os.path

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
import click

# @click.command()
# @click.argument("input_path", type=click.Path())
# @click.argument("output_path", type=click.Path())
# @click.option("--noise", default=40, type=int)
# @click.option("--amount-additional-profiles", default=200, type=int)
def test_noise(input_path: str,
                    output_path: str,
                    noise: int = 40,
                    amount_additional_profiles: int = 200) -> None:
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

    dtypes = [(f'{i}', 'f8') for i in range(12002)] + [('12002', 'U10'), ('12003', 'U10')]
    original_profiles = np.getfromtxt(input_path, delimiter=';', dtype=dtypes)

    #считали как DataFrame чтобы имена колонок
    profiles_for_columns = pd.read_csv(input_path, sep=';')
    columns = profiles_for_columns.columns

    profile_columns = [f'col{i}' for i in range(12002)]
    noise_factor = noise/100
    for i in range(len(original_profiles)):

        main = [row[name] for name in column_names]
        main = original_profiles[i]
        main[:12002] = main[:12002].astype(float)
        for m, j in enumerate(range(amount_additional_profiles)):
            tmp = main.copy()
            tmp[:12002] = tmp[:12002] + np.random.normal(loc=0,
                                                         scale=noise_factor * tmp[:12002],
                                                         size=(len(tmp[:12002])))
            tmp = abs(tmp)
            np.place(tmp[:12002], tmp[:12002] > 1, 1)
            if (m == 0) and (i == 0):
                final = np.array([tmp])
            else:
                final = np.append(final, np.array([tmp]), axis=0)
    profiles_inference = pd.DataFrame(final, columns=original_profiles_columns)
    profiles_inference.to_csv(output_path, sep=';', header=True, index=True )
    return None

# if __name__ == "__main__":
#     test_noise()

test_noise(os.path.join("..", "..", "data\\processed\\original_MS_profiles.csv"),
           os.path.join("..", "..", "data\\processed\\set\\test_set_normal_noise_40%.csv"))
