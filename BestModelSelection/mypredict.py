import joblib
import myconfig as config


def make_prediction(input_data):
    # _titanic_pipe = joblib.load(filename=config.PIPELINE_PATH)
    _titanic_pipe  = joblib.load(config.PIPELINE_PATH)
    # print(_titanic_pipe)
    pred_acc = _titanic_pi   pe.predict(input_data)
    pred_roc = _titanic_pipe.predict_proba(input_data)[:,1]
    return pred_acc,pred_roc


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    data = pd.read_csv(config.DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=0)

    pred_acc,pred_roc = make_prediction(X_test)

    acc = accuracy_score(y_test, pred_acc)
    roc = roc_auc_score(y_test, pred_roc)

    print(f'Test Accuracy:{acc:0.5f}')
    print(f'Test ROC:{roc:0.5f}')
    print()





