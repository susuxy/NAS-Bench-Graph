import argparse
from secrets import choice

def load_args():
    parser = argparse.ArgumentParser('Nas-Bench-Graph')
    # data input
    parser.add_argument('--train_mode', choices=['normal', 'data_transfer'], default='normal', type=str, help="normal means randomly split dataset and train & test, data_transfer means trains on some datasets and test on others")
    parser.add_argument('--train_data', nargs='*', type=str, default=['citeseer', 'computers', 'cora', 'cs', 'photo', 'physics', 'proteins', 'pubmed'])
    parser.add_argument('--test_data', nargs='*', type=str, default=['arxiv'])
    parser.add_argument('--data', nargs='*', default = ['arxiv', 'citeseer', 'computers', 'cora', 'cs', 'photo', 'physics', 'proteins', 'pubmed'], type=str)
    parser.add_argument('--reload', action='store_true', help="regenerate the dataframe of each dataset used")
    
    # feature generation
    parser.add_argument('--encode', choices = ['one_hot', 'category'], default = 'one_hot', help = 'encoding method for category variables')
    # parser.add_argument('--data_type', choices = ['complete', 'short'], default = 'complete', help = 'use complete benchmarks or short look up tables')
    
    # data split
    parser.add_argument('--random_seed', default=13, type=int, help='random seed for data split')
    parser.add_argument('--test_split_random_seed', default=1, type=int, help='random seed for spliting part of the testing dataset')
    parser.add_argument('--data_leak', default=0, type=float, help='when use data transfer, leverage few shot idea to use x percent of data in training set')
    parser.add_argument('--xgb_incremental', action='store_true')

    # performance evaluation
    parser.add_argument('--percentage_overlap', default=10, type=int, help = "calcualte scores of top x overlap")
    args = parser.parse_known_args()[0]
    return args