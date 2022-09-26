from args_loader import load_args
from train_models import ML_model

args = load_args()
ml_model = ML_model(args)
category_col = ['in_0', 'in_1', 'in_2', 'in_3', 'op_0', 'op_1', 'op_2', 'op_3']
ml_model.run(category_col)

