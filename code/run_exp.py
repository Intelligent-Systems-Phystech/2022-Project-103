prefix = '../data/'
def run_exp(model, X_train, X_test, y_train):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return prediction

def calc_metrics(y_true, y_pred, estimators=[]):
    res = {}
    for name in estimators:
        res[name] = estimators[name](y_true, y_pred)
    return res

def plot_result(y_true, y_pred, out):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['axes.labelsize'] = 24
    
    plt.figure(figsize=(15, 9))
    grid = np.arange(len(y_pred))
    plt.plot(grid, y_true, label='True Value')
    plt.plot(grid, y_pred, label='Prediction')
    plt.xlabel('Complex id')
    plt.ylabel('Affinity')
    plt.legend()
    plt.savefig(prefix + out)
    plt.show()    