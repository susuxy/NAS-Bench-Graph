from args_loader import load_args
from train_models import ML_model

args = load_args()
ml_model = ML_model(args)
category_col = ['in_0', 'in_1', 'in_2', 'in_3', 'op_0', 'op_1', 'op_2', 'op_3']
ml_model.run(category_col)


# # dataset_name = args.data

# # # bench = light_read(dataset_name)

# # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
# # logger = logging.getLogger(__name__)
# # handler = logging.FileHandler(dataset_name + '.log')
# # logger.addHandler(handler)
# # with open(dataset_name + '.log', 'w') as f:
# #     pass

# # logger.info(args)

# # df_total = read_data(dataset_name)
# # logger.info(f"total dataframe column names {df_total.columns}")
# # logger.info(f"total dataframe size is {df_total.shape}")


# # data preparation

# if args.encode == 'category':
#     df_total, le_ops = category_label_encode(df_total, ['op_0', 'op_1', 'op_2', 'op_3'], ['op_0', 'op_1', 'op_2', 'op_3'])
# elif args.encode == 'one_hot':
#     one_hot_col = ['in_0', 'in_1', 'in_2', 'in_3', 'op_0', 'op_1', 'op_2', 'op_3']
#     df_total = one_hot_label_encode(df_total, one_hot_col)



# # data split
# col_name_in_op = [col for col in df_total if col.startswith('in') or col.startswith('op')]
# col_names_x = col_name_in_op + ['params']

# # col_names_x = ['in_0', 'in_1', 'in_2', 'in_3', 'op_0_encode', 'op_1_encode', 'op_2_encode', 'op_3_encode', 'params']
# col_names_y = ['test_acc']
# X_train, X_test, y_train, y_test = split_dataset(df_total, col_names_x, col_names_y, 0.2, 13)
# logger.info(f"training data size for X and y is {X_train.shape} and {y_train.shape}")
# logger.info(f"testing data size for X and y is {X_test.shape} and {y_test.shape}")




# # model training and testing
# params = {
#     "n_estimators": 500,
#     "max_depth": 4,
# }
# model = model_training(params, X_train, y_train)


# r2, mse = model_results(model, X_test.values, y_test)
# # print([r2, mse])
# logger.info(f"R2 score {r2:.7f}")
# logger.info(f"MSE score {mse:.7f}")



# # plot feature importances
# # all_feats = ['in_0', 'in_1', 'in_2', 'in_3', 'op_0', 'op_1', 'op_2', 'op_3', 'params']
# plot_feat_importance(model, col_names_x, 'importance.png')


