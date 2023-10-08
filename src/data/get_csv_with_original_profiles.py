import click
import pandas as pd
import os


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def get_csv(input_path: str, output_path: str):

    """Собирает все профиля в одном csv файле. Изначально каждый профиль лежит
    в отдельном файле
    :param input_path: путь до папки, в которой лежат файлы с профилями и csv
    файл с колонками: название файла; штамм; ID
    :param output_path: путь до csv файла, где вместо названий файлов уже находятся
    сами профиля
    :return: None
    """
    # считали ключи (название файла - бактерия)
    profiles_ID = pd.read_csv(os.path.join(input_path, 'profiles_group_ID.csv'),
                              sep=';',
                              header=None)

    #считали один профиль для шаблона для колонок
    t = pd.read_csv(os.path.join(input_path, 'MS_profiles', '0a2c8dcb-9e19-47a4-9f35-0d7a5728eaa7'),
                    sep=';',
                    index_col=0,
                    header=None)
    t = t.T
    t['group'] = 0
    t['ID'] = 0

    # пустой словарь для удобного считывания
    original_profiles = pd.DataFrame({k: pd.Series(dtype=float) for k in t.columns})

    # считываем каждый файл
    for i in profiles_ID.index:
        path = os.path.join(input_path, 'MS_profiles', profiles_ID.at[i, 0])
        s = pd.read_csv(path, sep=';', index_col=0, header=None)
        s = s.T
        s['group'] = profiles_ID.at[i, 1]
        s['ID'] = profiles_ID.at[i, 2]
        original_profiles = pd.concat([original_profiles, s])

    # делаем красивые индексы
    original_profiles.index = profiles_ID.index
    original_profiles.to_csv(output_path, sep=';', index=False)

if __name__ == "__main__":
    get_csv()


# get_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'MS_profiles'),
#         os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'original_MS_profiles.csv'))
