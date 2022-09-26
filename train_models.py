import os
from tracemalloc import start
import pandas as pd
import numpy as np
from nas_bench_graph.readbench import light_read
from tqdm import tqdm
from utils import key2structure
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import xgboost
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import time

class ML_model:
    def __init__(self, args):
        self.args = args
        self.log_initialize()
        
    def log_initialize(self):
        log_file_name = '_'.join(self.args.data)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file_name + '.log')
        self.logger.addHandler(handler)
        with open(log_file_name + '.log', 'w') as f:
            pass
        self.logger.info(self.args)

    def run(self, category_col_names):
        # read data
        self.read_data()

        # encode category variables
        if self.args.encode == 'category':
            self.category_label_encode(category_col_names)
        elif self.args.encode == 'one_hot':
            self.one_hot_label_encode(category_col_names)
        
        # data split
        col_names_x = [col for col in self.df if col.startswith('in') or col.startswith('op')]
        col_names_x = col_names_x + ['params']
        col_names_y = ['test_acc']
        self.split_dataset(col_names_x, col_names_y)

        # model training
        params = {
            "n_estimators": 500,
            "max_depth": 4,
        }
        start_time = time.time()
        self.model_training(params)
        self.logger.info(f"model training time is {time.time() - start_time}")

        # model testing
        self.model_results()

        # feature importances
        self.plot_feat_importance(col_names_x)



    def read_data(self):
        df_list = []
        for each_file_name in self.args.data:
            df_file_path = os.path.join('nas_bench_graph', 'save_df', each_file_name + '.pkl')
            if os.path.exists(df_file_path):
                df = pd.read_pickle(df_file_path)
            else:
                # contruct dataframe
                bench = light_read(each_file_name)
                print(f"for dataset {each_file_name}, bench information is {bench[list(bench.keys())[0]].keys()}")
                df = pd.DataFrame(columns=['in_0', 'in_1', 'in_2', 'in_3', 'op_0', 'op_1', 'op_2', 'op_3', 'params', 'latency', 'test_acc'])
                for key_idx in tqdm(bench):
                    info = bench[key_idx]
                    structure = key2structure(key_idx)
                    sample = []
                    sample += structure[0]
                    sample += structure[1]

                    bench_sample = [info['para'], info['latency'], info['perf']]
                    sample += bench_sample

                    df.loc[df.shape[0]] = sample
                df.to_pickle(df_file_path)
            df_list.append(df)
        
        df = pd.concat(df_list)
        self.df = df
        self.logger.info(f"total dataframe column names {df.columns}")
        self.logger.info(f"total dataframe size is {df.shape}")

    def category_label_encode(self, col_names):
        # category label encoding
        ops_array = self.df[col_names].values.reshape(1, -1)
        ops_array = np.squeeze(ops_array, axis=0)


        self.le = LabelEncoder()
        self.le.fit(ops_array)

        self.df[col_names] = self.df[col_names].apply(self.le.transform)


    def one_hot_label_encode(self, col_names):
        self.df = pd.get_dummies(self.df, prefix = col_names, columns=col_names)

    def split_dataset(self, feat_col_name, pred_col_name, test_size=0.2, random_seed=13):

        X = self.df[feat_col_name]
        y = self.df[pred_col_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed
        )
        self.logger.info(f"training data size for X and y is {self.X_train.shape} and {self.y_train.shape}")
        self.logger.info(f"testing data size for X and y is {self.X_test.shape} and {self.y_test.shape}")

    def model_training(self, params):
        self.model = xgboost.XGBRegressor(**params)
        self.model.fit(self.X_train.values, self.y_train.values)

    def model_testing(self):
        self.y_pred = self.model.predict(self.X_test.values)

    def model_results(self):
        self.model_testing()
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        self.logger.info(f"R2 score {r2:.7f}")
        self.logger.info(f"MSE score {mse:.7f}")
        self.result = [r2, mse]

    def plot_feat_importance(self, feat_names, plot_save_path='importance.png', feat_num=5):
        feat_important = self.model.feature_importances_

        zipped_list = zip(feat_important, feat_names)
        sorted_zip_list = sorted(zipped_list, reverse=True)
        tuples = zip(*sorted_zip_list)
        feat_important_sort, feat_names_sort = [list(t) for t in tuples]
        self.logger.info(f"{feat_num} most important features {feat_names_sort[:feat_num]}")

        plt.figure(figsize=(16, 9))
        plt.bar(feat_names, feat_important)
        plt.title('feature importance')
        plt.xlabel('feature name')
        plt.ylabel('importance scores')
        plt.xticks(rotation='90')
        plt.savefig(plot_save_path)

