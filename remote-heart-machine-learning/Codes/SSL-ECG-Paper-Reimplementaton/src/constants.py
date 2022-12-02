class Constants:
    # cache_base_path = "D:/EMOTION PREDICT/SSL-ECG-Paper-Reimplementaton/cache/"
    # data_base_path = "D:/EMOTION PREDICT/WESAD/"
    # data_base_path = "/Users/joergsimon/Documents/work/datasets_cache/
    # data_base_path = "/home/jsimon/Desktop/knownew/600 Datasets/human-telemetry/other_datasets_joerg/"
    # model_base_path = "D:/EMOTION PREDICT/SSL-ECG-Paper-Reimplementaton/model_data"
    cache_base_path = "/home/lucas.lisboa/scratch/EMOTION_PREDICT/vialab21-desafio2/Codes/SSL-ECG-Paper-Reimplementaton/cache/"
    data_base_path = "/home/lucas.lisboa/scratch/EMOTION_PREDICT/raw_datasets/"
    model_base_path = "/home/lucas.lisboa/scratch/EMOTION_PREDICT/vialab21-desafio2/Codes/SSL-ECG-Paper-Reimplementaton/model_data/"
    results_path = "/home/lucas.lisboa/scratch/EMOTION_PREDICT/vialab21-desafio2/Codes/SSL-ECG-Paper-Reimplementaton/results"
    
    sample_name = "mahnob_gt" # "meta_v4_420"

    use_ray = True

    loss = 'CrossEntropyLoss' # MSELoss or CrossEntropyLoss 