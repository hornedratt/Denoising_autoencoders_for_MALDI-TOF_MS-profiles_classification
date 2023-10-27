rule all:
    input:
        "data\\processed\\original_MS_profiles.csv",
        "data\\processed\\sets\\test_set_normal_noise_40%.csv",
        "reports\\cross_valid_40%_result.csv",
        "reports\\figures\\cross_valid_40%_result_group.png",
        "reports\\figures\\cross_valid_40%_result_ID.png",
        'models\\DAE_norm_noise_40%.pkl',
        'reports\\figures\\DAE_norm_noise_40%.png',
        "reports\\figures\\heat_map_group_40%.png",
        "reports\\figures\\heat_map_ID_40%.png",
        "models\\forest_40%_group",
        "reports\\forest_40%_group.csv",
        "models\\forest_40%_ID",
        "reports\\forest_40%_ID.csv",
        "reports\\figures\\forest_40%_importances_group.png",
        "reports\\figures\\forest_40%_importances_ID.png",
        "reports\\mz_features_40%_group.txt",
        "reports\\mz_features_40%_ID.txt"
rule make_original_profiles_csv:
    input:
        "data\\raw"
    output:
        "data\\processed\\original_MS_profiles.csv"
    shell:
        "python -m src.data.get_csv_with_original_profiles {input} {output}"
rule add_normal_noise:
    input:
        "data\\processed\\original_MS_profiles.csv"
    output:
        "data\\processed\\sets\\test_set_normal_noise_40%.csv"
    shell:
        "python -m src.data.test_noise {input} {output} --noise 40"
rule train_autoencoder:
    output:
        'models\\DAE_norm_noise_40%.pkl',
        'reports\\figures\\DAE_norm_noise_40%.png'
    shell:
        "python -m src.models.train {output}" #--n_epochs 2"
rule heat_map:
    input:
        "models\\DAE_norm_noise_40%.pkl",
        "data\\processed\\sets\\test_set_normal_noise_40%.csv"
    output:
        "reports\\figures\\heat_map_group_40%.png",
        "reports\\figures\\heat_map_ID_40%.png"
    shell:
        "python -m src.visualization.heat_map {input} {output}"
rule train_forest:
    input:
        "models\\DAE_norm_noise_40%.pkl",
        "data\\processed\\sets\\test_set_normal_noise_40%.csv"
    output:
        "models\\forest_40%_group",
        "reports\\forest_40%_group.csv",
        "models\\forest_40%_ID",
        "reports\\forest_40%_ID.csv"
    shell:
        "python -m src.models.train_forest {input} {output}"
rule cross_validation:
    input:
        "data\\processed\\sets\\test_set_normal_noise_40%.csv",
        "models\\DAE_norm_noise_40%.pkl"
    output:
        "reports\\cross_valid_40%_result.csv",
        "reports\\figures\\cross_valid_40%_result_group.png",
        "reports\\figures\\cross_valid_40%_result_ID.png"
    shell:
        "python -m src.models.cross_valid {input} {output}"

rule importance_analysis:
    input:
        "data\\processed\\sets\\test_set_normal_noise_40%.csv",
        "models\\DAE_norm_noise_40%.pkl",
        "models\\forest_40%_group",
        "models\\forest_40%_ID"
    output:
        "reports\\figures\\forest_40%_importances_group.png",
        "reports\\figures\\forest_40%_importances_ID.png",
        "reports\\mz_features_40%_group.txt",
        "reports\\mz_features_40%_ID.txt"
    shell:
        "python -m src.features.importance_analysis {input} {output}"
