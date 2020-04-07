import numpy as np
import math

def r_squared(y_pred, y_actual):
	tmp = y_pred - y_actual
	tmp = tmp * tmp
	ss_res = np.sum(tmp)
	mean = np.mean(y_actual)
	tmp = y_actual - mean
	tmp = tmp * tmp
	ss_tot = np.sum(tmp)
	return (1 - (ss_res / (ss_tot + np.finfo(np.float32).eps)))


def rmse(y_pred, y_actual):
	tmp = y_pred - y_actual
	tmp = tmp * tmp
	mean = np.mean(tmp)
	return math.sqrt(mean)

def mae(y_pred, y_actual):
	tmp = y_pred - y_actual
	tmp = np.abs(tmp)
	return np.mean(tmp)

def mape(y_pred, y_actual):
	tmp = y_actual - y_pred
	tmp = np.abs(tmp)
	tmp = tmp / (y_actual + np.finfo(np.float32).eps)
	return np.mean(tmp) * 100

def compute_metrics(y_pred, y_actual):
	return rmse(y_pred, y_actual), mae(y_pred, y_actual), mape(y_pred, y_actual), r_squared(y_pred, y_actual)