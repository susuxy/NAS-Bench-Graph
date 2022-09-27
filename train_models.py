import os
from tracemalloc import start
import pandas as pd
import numpy as np
from nas_bench_graph.readbench import light_read
from tqdm import tqdm
from utils import key2structure, dataset2info
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import xgboost
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import time
from scipy import stats
from sklearn.utils import shuffle

class ML_model:
    def __init__(self, args):
        self.args = args
        self.args.data = sorted(self.args.data)
        self.args.train_data = sorted(self.args.train_data)
        self.args.test_data = sorted(self.args.test_data)
        self.log_initialize()
        
    def log_initialize(self):
        if self.args.train_mode == 'normal':
            log_file_name = '_'.join(self.args.data) + '.log'
        elif self.args.train_mode == 'data_transfer':
            log_file_name = '_'.join(self.args.train_data) + '__' + '_'.join(self.args.test_data) + '.log'
        log_file_path = os.path.join('log_file', log_file_name)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file_path)
        self.logger.addHandler(handler)
        with open(log_file_path, 'w') as f:
            pass
        self.logger.info(self.args)

    def run(self, category_col_names):
        if self.args.train_mode == 'normal':
            # read data
            self.read_data(self.args.data)
        elif self.args.train_mode == 'data_transfer':
            self.read_data_datatransfer()

        # encode category variables
        if self.args.encode == 'category':
            self.category_label_encode(category_col_names)
        elif self.args.encode == 'one_hot':
            self.one_hot_label_encode(category_col_names)
        
        # data split
        col_names_x = [col for col in self.df if col.startswith('in') or col.startswith('op') or col.startswith('dataset')]
        # col_names_x = [col for col in self.df if col.startswith('in') or col.startswith('op')]
        col_names_x = col_names_x + ['params', 'layers']
        col_names_y = ['test_acc']
        if self.args.train_mode == 'normal':
            self.split_dataset(col_names_x, col_names_y, random_seed=self.args.random_seed)
        elif self.args.train_mode == 'data_transfer':
            self.split_dataset_datatransfer(col_names_x, col_names_y)

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

    def read_data_datatransfer(self):
        self.train_df = self.read_data(self.args.train_data, dataset_name='train')
        self.test_df = self.read_data(self.args.test_data, dataset_name='test')
        shuffle(self.train_df)
        shuffle(self.test_df)
        self.df = pd.concat([self.train_df, self.test_df])
        self.train_size = self.train_df.shape[0]
        self.test_size = self.test_df.shape[0]



    def read_data(self, dataset_list, dataset_name='total'):
        df_list = []
        for each_file_name in dataset_list:
            df_file_path = os.path.join('nas_bench_graph', 'save_df', each_file_name + '.pkl')
            if (not self.args.reload) and os.path.exists(df_file_path):
                df = pd.read_pickle(df_file_path)
            else:
                # contruct dataframe
                bench = light_read(each_file_name)
                print(f"for dataset {each_file_name}, bench information is {bench[list(bench.keys())[0]].keys()}")
                df = pd.DataFrame(columns=['in_0', 'in_1', 'in_2', 'in_3', 'op_0', 'op_1', 'op_2', 'op_3', 'params', 
                'latency', 'test_acc', 'layers', 'dataset_nodes', 'dataset_edges', 'dataset_feats', 'dataset_types'])
                for key_idx in tqdm(bench):
                    info = bench[key_idx]
                    structure = key2structure(key_idx)
                    sample = []
                    sample += structure[0]
                    sample += structure[1]

                    bench_sample = [info['para'], info['latency'], info['perf']]
                    sample += bench_sample

                    sample.append(max(structure[0]) + 1)
                    
                    dataset_sample = dataset2info(each_file_name)
                    sample += dataset_sample

                    df.loc[df.shape[0]] = sample
                df.to_pickle(df_file_path)
            df_list.append(df)
        
        df = pd.concat(df_list)
        self.df = df
        self.logger.info(f"{dataset_name} dataframe column names {df.columns}")
        self.logger.info(f"{dataset_name} dataframe size is {df.shape}")
        return df

    def category_label_encode(self, col_names):
        # category label encoding
        ops_array = self.df[col_names].values.reshape(1, -1)
        ops_array = np.squeeze(ops_array, axis=0)


        self.le = LabelEncoder()
        self.le.fit(ops_array)

        self.df[col_names] = self.df[col_names].apply(self.le.transform)


    def one_hot_label_encode(self, col_names):
        self.df = pd.get_dummies(self.df, prefix = col_names, columns=col_names)

    def split_dataset_datatransfer(self, feat_col_name, pred_col_name):
        train_set = self.df.iloc[:self.train_size, :]
        test_set = self.df.iloc[self.train_size:, :]
        assert test_set.shape[0] == self.test_size
        self.X_train = train_set[feat_col_name]
        self.y_train = train_set[pred_col_name]
        self.X_test = test_set[feat_col_name]
        self.y_test = test_set[pred_col_name]

        self.logger.info(f"training data size for X and y is {self.X_train.shape} and {self.y_train.shape}")
        self.logger.info(f"testing data size for X and y is {self.X_test.shape} and {self.y_test.shape}")

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
        self.plot_rank_spearman()
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        self.logger.info(f"R2 score {r2:.7f}")
        self.logger.info(f"MSE score {mse:.7f}")
        self.result = [r2, mse]

    def plot_rank_spearman(self):
        x,y = self.y_pred, self.y_test
        if self.args.train_mode == 'normal':
            score_path = os.path.join('spearman_plot', '_'.join(self.args.data) + '_score.png')
            rank_path = os.path.join('spearman_plot', '_'.join(self.args.data) + '_rank.png')
        elif self.args.train_mode == 'data_transfer':
            score_path = os.path.join('spearman_plot', '_'.join(self.args.train_data) + '__' + '_'.join(self.args.test_data) + '_score.png')
            rank_path = os.path.join('spearman_plot', '_'.join(self.args.train_data) + '__' + '_'.join(self.args.test_data) + '_rank.png')
        rank_x = stats.rankdata(x)
        rank_y = stats.rankdata(y)
        spearman_corr, pvalue = stats.spearmanr(rank_x, rank_y)

        plt.figure()
        plt.scatter(x, y, s=2)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.show()
        plt.savefig(score_path)

        plt.figure()
        plt.scatter(rank_x, rank_y, s=2)
        plt.xlabel("rank_y_pred")
        plt.ylabel("rank_y_true")
        plt.show()
        plt.savefig(rank_path)
        self.logger.info(f"Spearman corr is {spearman_corr}, num_samples is {len(rank_x)}")

    def plot_feat_importance(self, feat_names, feat_num=5):
        if self.args.train_mode == 'normal':
            plot_save_path = os.path.join('importance_plot', '_'.join(self.args.data) + '_importance.png')
        elif self.args.train_mode == 'data_transfer':
            plot_save_path = os.path.join('importance_plot', '_'.join(self.args.train_data) + '__' + '_'.join(self.args.test_data) + '_importance.png')
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

