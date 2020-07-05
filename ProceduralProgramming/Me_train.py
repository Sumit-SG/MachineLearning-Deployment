def data_prep(data):
    data['cabin'] = data['cabin'].apply(lambda line: pf.extract_letter(line))

    for var in config.CATEGORICAL_IMPUTE:
        data[var] = pf.impute_na(data=data,var = var,impute_value="Missing")

    for var in config.NUMERIC_IMPUTE:
        data[var+"_na"] = pf.add_missing_indicator(data=data,var=var)

    for var in config.NUMERIC_IMPUTE:
        data[var] = pf.impute_na(data=data,var=var,impute_value=config.NUMERIC_MEDIAN[var])

    for var in config.CATEGORICAL_VARIABLES:
        data[var] = pf.remove_rare_labels(data=data,var=var,frequency_ls=config.FREQUENCY_LABELS[var])

    data = pf.check_variables(data=data,column_names=config.VARIABLE_LIST)

    data = pf.scale_features(data=data, scaler_file=config.SCALER_PATH)

    return data


if __name__=="__main__":
    import Me_config as config
    import Me_preprocessing_functions as pf
    from sklearn.metrics import accuracy_score, roc_auc_score

    data = pf.load_data(DATA_PATH=config.DATA_PATH)
    X_train, X_test, y_train, y_test = pf.data_split(data=data, target=config.TARGET)
    X_train = data_prep(data=X_train)

    pf.train_model(data=X_train, target=y_train, model_path=config.MODEL_PATH)
    print("Finished Training")

    pred_acc, pred_roc =pf.predict_model(data=X_train,model_path=config.MODEL_PATH)

    print(f'Train Accuracy:{accuracy_score(y_train,pred_acc):0.5f}')
    print(f'Train ROC:{roc_auc_score(y_train,pred_roc[:,1]):0.5f}')
    print()











