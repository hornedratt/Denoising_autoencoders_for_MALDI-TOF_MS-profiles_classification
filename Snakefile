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
        "reports\\figures\\heat_map_ID_40%.png"
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
        "python -m src.models.train {output}"
rule heat_map:
    input:
        "data\\processed\\sets\\test_set_normal_noise_40%.csv",
        "models\\DAE_norm_noise_40%.pkl"
    output:
        "reports\\figures\\heat_map_group_40%.png",
        "reports\\figures\\heat_map_ID_40%.png"
    shell:
        "python -m src.visualization.heat_map {input} {output}"
rule cross_validation:
    input:
        "data\\processed\\sets\\test_set_normal_noise_40%.csv",
        "models\\old_models\\DAE_norm_noise_40%.pkl"
    output:
        "reports\\cross_valid_40%_result.csv",
        "reports\\figures\\cross_valid_40%_result_group.png",
        "reports\\figures\\cross_valid_40%_result_ID.png"
    shell:
        "python -m src.models.cross_valid {input} {output}"
