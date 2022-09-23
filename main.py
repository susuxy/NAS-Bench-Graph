from nas_bench_graph.readbench import light_read
from nas_bench_graph.architecture import Arch
import pandas as pd
from tqdm import tqdm
from utils import key2structure
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt 
import os
import logging


dataset_name = 'arxiv'

bench = light_read(dataset_name)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(dataset_name + '.log')
logger.addHandler(handler)
with open(dataset_name + '.log', 'w') as f:
    pass


def read_data(file_name):
    df_file_path = os.path.join('nas_bench_graph', 'save_df', file_name + '.pkl')
    if os.path.exists(df_file_path):
        df = pd.read_pickle(df_file_path)
    else:
        # contruct dataframe
        bench = light_read(file_name)
        df = pd.DataFrame(columns=['in_0', 'in_1', 'in_2', 'in_3', 'op_0', 
        'op_1', 'op_2', 'op_3', 'params', 'latency', 'valid_acc', 'test_acc'])
        for key_idx in tqdm(bench):
            info = bench[key_idx]
            structure = key2structure(key_idx)
            sample = []
            sample += structure[0]
            sample += structure[1]

            bench_sample = [info['para'], info['latency'], info['valid_perf'], info['perf']]
            sample += bench_sample

            df.loc[df.shape[0]] = sample
        df.to_pickle(df_file_path)
    return df

df_total = read_data('arxiv')
logger.info(f"total dataframe size is {df_total.shape}")


# data preparation

def category_label_encode(df, origin_col_names, output_col_names):
    # category label encoding
    ops_array = df[origin_col_names].values.reshape(1, -1)
    ops_array = np.squeeze(ops_array, axis=0)


    le = LabelEncoder()
    le.fit(ops_array)

    df[output_col_names] = df[origin_col_names].apply(le.transform)

    return df, le


df_total, le_ops = category_label_encode(df_total, ['op_0', 'op_1', 'op_2', 'op_3'], ['op_0_encode', 'op_1_encode', 'op_2_encode', 'op_3_encode'])

# one-hot label encoding
# df = pd.get_dummies(df, prefix=['in_0', 'in_1', 'in_2', 'in_3'], columns=['in_0', 'in_1', 'in_2', 'in_3'])
# df = pd.get_dummies(df, prefix=['op_0', 'op_1', 'op_2', 'op_3'], columns=['op_0', 'op_1', 'op_2', 'op_3'])






def split_dataset(df, feat_col_name, pred_col_name, test_size, random_seed):

    X = df[feat_col_name]
    y = df[pred_col_name]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

col_names_x = ['in_0', 'in_1', 'in_2', 'in_3', 'op_0_encode', 'op_1_encode', 'op_2_encode', 'op_3_encode', 'params']
col_names_y = ['test_acc']
X_train, X_test, y_train, y_test = split_dataset(df_total, col_names_x, col_names_y, 0.2, 13)
logger.info(f"training data size for X and y is {X_train.shape} and {y_train.shape}")
logger.info(f"testing data size for X and y is {X_test.shape} and {y_test.shape}")



def model_training(params, X, y):
    model = xgboost.XGBRegressor(**params)
    model.fit(X.values, y.values)
    return model


params = {
    "n_estimators": 500,
    "max_depth": 4,
}
model = model_training(params, X_train, y_train)


def model_testing(model, X):
    y = model.predict(X)
    return y

def model_results(model, X, y_true):
    y_pred = model_testing(model, X)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse

r2, mse = model_results(model, X_test.values, y_test)
print([r2, mse])
logger.info(f"R2 score {r2:.7f}")
logger.info(f"MSE score {mse:.7f}")


def plot_feat_importance(model, feat_names, plot_save_path):
    feat_important = model.feature_importances_
    plt.figure()
    plt.bar(feat_names, feat_important)
    plt.title('feature importance')
    plt.xlabel('feature name')
    plt.ylabel('importance scores')
    plt.savefig(plot_save_path)

all_feats = ['in_0', 'in_1', 'in_2', 'in_3', 'op_0', 'op_1', 'op_2', 'op_3', 'params']
plot_feat_importance(model, all_feats, 'importance.png')


