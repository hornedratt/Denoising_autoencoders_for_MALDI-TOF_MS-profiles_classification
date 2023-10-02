import pandas as pd
import numpy as np
def inference_noise(input_path: str,
                    output_path: str,
                    noise: int = 40,
                    amount_additional_profiles: int = 200) -> None:
    """Делает зашумленный validset, таким же образом как и в trainloop: генирируем 12000(размерность
    профилей) слуйчайных величин из нормального распределения с нулевым средним и дисперсией, как в
    рассматриваемом векторе и этот ветор, домноженный на необходимый процент шума, прибавим к рассматриваемому
    вектору
    :param input_path: путь до папки, в которой лежаит csv с оригинальным набором профилей
    :param output_path: путь, куда сохраним сгенирированный csv
    :param noise: конста на которую домножаем сгенерированный вектор
    :param amount_additional_profiles: сколько зашумленных векторов сделаем из каждого оригинального
    :return: None
    """
    original_profiles = pd.read_csv(input_path)
    noise_factor = noise/100
    for n, i in tqdm(enumerate(pd.unique(original_profiles.at[:,'group']))):
        mask = original_profiles.at[:, 'group'] == i
        s = original_profiles.at[mask, :15000.0].copy()
        s.index = {j for j in range (len(s.index))}  # красивые индексы
        s = s.to_numpy()
        main = s.copy()
        for m, j in enumerate(range(amount_additional_profiles)):
            s = main.copy()
            tmp = s.astype(float)
            tmp = tmp + np.random.normal(loc=0.0,
                                         scale=noise_factor * tmp,
                                         size=(len(s)))
            tmp = abs(tmp)
            s = tmp
            np.place(s, s > 1, 1)
            if (m == 0) and (n == 0):
                S = np.array([s])
            else:
                S = np.append(S, np.array([s]), axis = 0)
    MS_profiles_inference = pd.DataFrame(S, columns = MS_profiles.columns)
    MS_profiles_inference.index = {i for i in range(len(MS_profiles_inference.index))}
    path = r'C:\education\ML\MS_profiles\sets\set_group_distribution_ ' +str(noise ) +'%.csv'
    MS_profiles_inference.to_csv(path, sep=';', header=True, index=True ,)
    return None