# Denoising autoencoder в распозновании масс-спектров MALDI-TOF
 В данном репозитории представленны методы классификации масс-спектров MALDI-TOF с использованием denoising автоенкодера (далее DAE), для снижения размерности профилей, и модели случайного леса, для классификации скрытых состояний. Также, присутствуют методы зашумления профилей (так как была острая нехватка исходных данных (~ 150 профилей) и, вместе с этим, это необходимо для обучения DAE) и методы анализирующие некоторые характеристики моделей и данных. Все представленные результаты слишком идеальные, так как, как я уже упоминала исходный датасэт был всего 150 профилей, поэтому их не нужно воспринимать как настоящие результаты, скорее это просто пример того что делает данный репозиторий.  

### Данные
 Данные представляют собой вектора размерером 12000, где каждая компонента - нормированная методами от MALDI-TOF интенсивность белка с определенной массой, т. е. это вещественное число из интервала [0, 1], с двумя лейблами - группа и штамм, пример исходных данных есть в data/raw.
 
### Шум
 Метод add_normal_noise генирирует набор зашумленных профилей для построения модели случайного леса (10 зашумленных из одного оригинального) по формуле:
$$C_n = |C_o + \xi|, \xi \in N(0, \sigma * d)$$
где $C_o$ - некоторая кордината оригинального профиля, $\sigma$ - средне-квадратичное отклонение координат в рассматриваемом оригинальном профиле, d - доля (уровень) шума от оригинального. В предложенном пайплайне $d \in \\{ 0.1, 0.2, 0.3, 0.4 \\}$. Все $C_n > 1$ заменяются на 1.
В автоенкодере зашумление происходит аналогичным образом на этапе формирования батча(по умолчанию размер набора для данных для автоенкодера - 2500 шт. (если ваш набор больше этот параметр нужно будет увеличить, см "Требования и запуск пайплайна"), batch_size=64, train_size=0.7) (все последующие примеры графиков и отчетов будут для $d=0.4$)

### Pipeline
Пайплайн был написан с помощью snakemake, его DAG:
![DAG](reports/figures/DAG.png?raw=true)

### Обучение DAE
 Vanilla autoencoder - полносвязный автоенкодер. По умолчанию количество эпох - 50. В методе train строиться график функции потерь от эпохи на train и valid выборках:
<p align="center">
 <img src='reports/figures/DAE_norm_noise_40%25.png' width=320>
</p>
 Также, в пайплайне строиться heat map,для разбиения по группам и штаммам, соответственно:
<p align="center">
 <img src='reports/figures/heat_map_group_40%25.png' width=420>
</p>
<p align="center">
 <img src='reports/figures/heat_map_ID_40%25.png' width=420>
</p>
 Каждая точка в heatmap - евклидово расстояние между средними групп/штаммов, на диагонали стоят среднегрупповые/среднештаммовые расстояния. Так как строики/столбцы у каждой группы/штамма хорошо различимы и минимальные значения стоят на диагоналях, можно предположить, что полученное скрытое пространство хорошо подходит для дальнейшего решения задачи классификации.
 
 ### Обучение случайного леса
  Реализация случайного леса была взята из библиотеки scikit-learn cо следующими параметрами: n_estimators=10, criterion='gini', min_samples_split=8, min_samples_leaf=4). Train_forest сохраняет отчет о работе построенного леса (прмиер с группами, со штаммами есть аналогичный отчет):
  
|                |   precision |   recall |   f1-score |    support |
|:----------------------------|------------:|---------:|-----------:|-----------:|
| Anoxybacillus_flavithermus  |    1        | 1        |   1        |   8        |
| Bacillus_altitudinis        |    1        | 0.985075 |   0.992481 |  67        |
| Bacillus_aryabhattai        |    1        | 1        |   1        |   4        |
| Bacillus_atrophaeus         |    1        | 1        |   1        |  10        |
| Bacillus_berkeleyi          |    1        | 1        |   1        |   7        |
| Bacillus_cereus             |    1        | 1        |   1        |  46        |
| Bacillus_chungangenis       |    1        | 1        |   1        |   5        |
| Bacillus_clausii            |    1        | 1        |   1        |   8        |
| Bacillus_coagulans          |    1        | 1        |   1        |   6        |
| Bacillus_flexus             |    1        | 1        |   1        |  14        |
| Bacillus_licheniformis      |    1        | 1        |   1        |  72        |
| Bacillus_megaterium         |    1        | 1        |   1        |  50        |
| Bacillus_mycoides           |    1        | 1        |   1        |   3        |
| Bacillus_pumilus            |    0.991071 | 1        |   0.995516 | 111        |
| Bacillus_simplex            |    1        | 1        |   1        |  40        |
| Bacillus_subtilis           |    1        | 1        |   1        |   4        |
| Bacillus_thuringiensis      |    1        | 1        |   1        |   5        |
| Bacillus_weihenstephanensis |    1        | 1        |   1        |   7        |
| E-Coli                      |    1        | 1        |   1        |   7        |
| Geobacillus_subterraneus    |    1        | 1        |   1        |  18        |
| accuracy                    |    0.997967 | 0.997967 |   0.997967 |   0.997967 |
| macro avg                   |    0.999554 | 0.999254 |   0.9994   | 492        |
| weighted avg                |    0.997986 | 0.997967 |   0.997964 | 492        |
  
   Также, в пайплайне строиться гистограмма точностей при кроссвалидации (по умолчанию валидируемся 1000 раз):
<p align="center">
 <img src='reports/figures/cross_valid_40%25_result_group.png' width=420>
</p>
 importance analysis, на основании критерия gini строит график наиболее важных для классификации фичей в скрытом состоянии(прмиер с группами, со штаммами есть аналогичный график):
<p align="center">
 <img src='reports/figures/forest_40%25_importances_group.png' width=420>
</p>
 после чего по самым большим весам кодера идем до его входного слоя, на котором, через $\omega_{j d}$ - все веса, ищутся самые важные фичи для классификации уже в исходном пространстве, по формуле:
 
$$
\omega_{j k}- mean\left(\omega_{j d}\right)>\beta * std\left(\omega_{j d}\right)
$$

т. е. согласно этой формуле k-ая фича - является важной. Здесь $\beta$ - гиперпараметр, который подбирался так, чтобы мы находили ~ 150 важных фичей. Номера этих фичей лежат в reports/mz_features_40%_group.txt
### Cross noise
 Последний метод в пайплайне берет все полученные модели и сэты и строит матрицы с точностями и f1-мерами при применении моделей к сэтам с уровнями шума, отличными от тех, на которых эти модели обучались. Строки отвечают за уровень шума при обучении (Train Noise), а столюцы на уровень шума на входе (Input Noise). Пример с f1-мерой для групп:
 
|   |   | Input Noise        | Input Noise     | Input Noise   | Input Noise    |
|:-------------|:-------------|:-------------------|:-------------------|:-------------------|:-------------------|
|         |        | 10%                | 20%                | 30%                | 40%                |
| Train Noise  | 10%          | 1.0                | 1.0                | 1.0                | 1.0                |
| Train Noise  | 20%          | 0.9982857142857142 | 0.9982857142857142 | 0.9982857142857142 | 0.9982857142857142 |
| Train Noise  | 30%          | 0.99728002920774   | 0.99728002920774   | 0.99728002920774   | 0.99728002920774   |
| Train Noise  | 40%          | 0.9993998449037391 | 0.9993998449037391 | 0.9993998449037391 | 0.9993998449037391 |

### Требования и запуск пайплайна
Все запуски проводились на домашней машине с RTX3060 и ryzen 5 3600 6 в кондовской виртуальной среде. Для установки environment.yml требуется cuda версии 12.1. Команды для установки среды и запуска пайплайна:
```shell
conda env export > environment.yml
snakemake --cores 4 --resources mem_mb=4000
```
 ограничения на использование памяти и числа ядер необходимы, так как в пайплайне много задач и без них они моментально перегружают память и проц, и устанавливаются в зависимости от машины. 
 Могу добавить, что, если ваш датасэт больше 2500 шт профилей, то shell команды в правиле train_autoencoder я бы заменила на:
 ```shell
python -m src.models.train {output} --noise_factor {wildcards.noise} --set_size=size_of_your_set
```
а в правиле add_normal_noise на:
 ```shell
python -m src.data.test_noise {input} {output} --noise {wildcards.noise} --amount_additional_profiles 1
```
### Структура репозитория
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
