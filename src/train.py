import argparse
import pandas as pd
import config
import model_dispacher
from sklearn import metrics
import helper
import numpy as np
import sklearn

print(f"sklearn version : {sklearn.__version__}")


def training(df, fold):
    print("#" * 40)
    print(f"FOLD : {fold}")
    df_train = df[df.kfold != fold].reset_index()
    df_valid = df[df.kfold == fold].reset_index()

    xtrain = df_train.drop("Product_Sold", axis=1)
    ytrain = df_train["Product_Sold"]

    xvalid = df_valid.drop("Product_Sold", axis=1)
    yvalid = df_valid["Product_Sold"]
    model = model_dispacher.models[args.model]
    model.fit(xtrain, ytrain)
    train_preds = model.predict(xtrain)
    test_preds = model.predict(xvalid)
    train_rmse_score = np.sqrt(metrics.mean_squared_error(ytrain, train_preds))
    test_rmse_score = np.sqrt(metrics.mean_squared_error(yvalid, test_preds))
    train_r2_score = metrics.r2_score(ytrain, train_preds)
    test_r2_score = metrics.r2_score(yvalid, test_preds)

    print(f"\tTraining RMSE : {train_rmse_score}")
    print(f"\tTesting RMSE : {test_rmse_score}")
    print(f"\tTraining R2 Score : {train_r2_score}")
    print(f"\tTesting R2 Score : {test_r2_score}\n")

    print("#" * 40)
    return train_rmse_score, test_rmse_score, train_r2_score, test_r2_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument(
        "--model", type=str, default="lr", choices=["lr", "dt", "lasso", "ridge"]
    )

    args = parser.parse_args()
    avg_train_rmse = 0
    avg_test_rmse = 0
    avg_train_r2 = 0
    avg_test_r2 = 0
    df = pd.read_csv(config.FOLD_10_FILE)
    df_new = helper.create_engineered_features(df)
    all_coefs = []

    for f in range(args.fold):
        train_rmse, test_rmse, train_r2, test_r2 = training(df_new, f)
        avg_train_rmse += train_rmse
        avg_test_rmse += test_rmse
        avg_train_r2 += train_r2
        avg_test_r2 += test_r2

    avg_train_rmse /= args.fold
    avg_test_rmse /= args.fold
    avg_train_r2 /= args.fold
    avg_test_r2 /= args.fold

    print(f"\nAverage Training RMSE : {avg_train_rmse}")
    print(f"Average Testing RMSE : {avg_test_rmse}")
    print(f"\nAverage Training R2 Score : {avg_train_r2}")
    print(f"Average Testing R2 Score : {avg_test_r2}")
