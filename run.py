#CREDIT : Work is bases on the top kernel : https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features

import polars as pl
import lightgbm as lgb
import argparse

from pathlib import Path
from sklearn.metrics import roc_auc_score
from preproc import preprocessor
from sklearn.model_selection import StratifiedKFold


param = {'n_estimators': 1024,
         'learning_rate': 0.005,
         'metric': 'rmse',
         'random_state': 42,
         'force_col_wise': True,
         'verbosity': 0,}


def run(dir_path, scores_path):
	train_scores = pl.read_csv(scores_path + 'train_scores.csv')
	train_essays = preprocessor(dir_path)
	train_data = train_essays.join(train_scores, on='id', how='left', coalesce=True)
	X = train_data.drop(['id', 'score']).collect().to_pandas()
	y = train_data['score'].collect().to_pandas().values
	skf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
	preds = np.zeros(x.shape[0], 5)
	for train_idx , valid_idx in skf.split(X, y.astype(str)):
		X_train, y_train = train_data[train_idx], train_data[train_idx]
		X_valid, y_valid = train_data[valid_idx], train_data[valid_idx]
		model = lgb.LGBMRegressor(**param)
		model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)])
		preds[valid_idx] = model.predict(X_valid)
	return np.mean(preds, axis=1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='run the trainig models.')
	parser.add_argument('--dir_path', metavar='path', required=True, help='the path to data files')
	parser.add_argument('--scores_path', metavar='path', required=True, help='the path to score files')
	args = parser.parse_args()
	run(args.dir_path, args.scores_path)

	