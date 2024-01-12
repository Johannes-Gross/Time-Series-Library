from utils.m4_summary import M4Summary

model_name = 'TimesNet_causal' # This is name of the folder in m4_results
file_path = './m4_results/' + model_name + '/'
root_path = './dataset/m4'

m4_summary = M4Summary(file_path, root_path, custom_eval=True)
smape_results, owa_results, mape, mase = m4_summary.evaluate()
print('smape:', smape_results)
print('mape:', mape)
print('mase:', mase)
print('owa:', owa_results)