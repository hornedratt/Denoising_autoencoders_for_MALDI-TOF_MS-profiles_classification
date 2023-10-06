rule all:
    input:
       "data\\processed\\original_MS_profiles.csv",
#       "data\\processed\\sets\\test_set_normal_noise_50%.csv"
       "data\\processed\\sets\\test_set_normal_noise_50%.csv"
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
        "data\\processed\\sets\\test_set_normal_noise_50%.csv"
    shell:
        "python -m src.data.test_noise {input} {output} --noise 50"
