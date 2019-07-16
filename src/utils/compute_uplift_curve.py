import pandas as pd
import numpy as np


def f_compute_uplift_curve(df_res: pd.DataFrame, delay_threshold: int = 60):
	"""
	Function to evaluate the model output.
	It receives a pandas DataFrame with colums:
	  - "ARRIVAL_DELAY": these are the true delays. In minutes.
	  - "Y_PROB": these are the output probabilities of the model; namely, P(T > threshold).
	:param df_res:
	:param delay_threshold:
	:return:
	"""
	col_prob = f'PROB_DELAYED_MORE_THAN_{delay_threshold}'
	col_true = f'DELAYED_MORE_THAN_{delay_threshold}'

	df = pd.DataFrame(np.vstack([df_res['ARRIVAL_DELAY'] > delay_threshold, df_res['Y_PROB']]).T, columns=[col_true, col_prob])
	df['BIN'] = pd.qcut(df[col_prob], q=10, duplicates='drop')
	df_eval = (
		df
		.groupby(by='BIN')
		.apply(lambda x: pd.Series([
			x[col_true].sum() / x.shape[0],
			x.shape[0]]))
		.fillna(0.)
		.rename(columns={0: 'RATIO', 1: 'SUPPORT'})
		)

	df_eval['SUPPORT'] = df_eval['SUPPORT'].astype(np.int)
	return df_eval.reset_index()
