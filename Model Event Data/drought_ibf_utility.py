import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
import sklearn.metrics as sklm
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler

'''
Main Functions
'''

def fit_threshold_model(data, feature, label_name, Plot=True):
    reduced_data = reduce_data(data, label_name)

    balanced_data = balance_data(reduced_data, label_name, undersampling_seed=10)

    median_val = balanced_data[[feature, label_name]].groupby(label_name).median()
    median_diff = median_val.loc[True] - median_val.loc[False]
    threshold_up = (median_diff > 0).values[0]

    scores = balanced_data[feature]

    labels = balanced_data[label_name]

    thresholds = np.linspace(scores.min(), scores.max(), 1000)

    thresholds = thresholds[1:-1]

    All_f = []

    for threshold in thresholds:
        if threshold_up:
            pred = scores >= threshold
        else:
            pred = scores < threshold
        f = sklm.f1_score(labels, pred, labels=[True, False], average='macro')
        All_f.append(f)

    best_threshold = thresholds[np.argmax(All_f)]

    if Plot:
        plt.plot(thresholds, All_f, zorder=1)
        plt.axvline(best_threshold, ls='--', color='k', zorder=0)
        plt.xlabel('Thresholds')
        plt.ylabel('Average F1-score')

    if threshold_up:
        pred = scores >= best_threshold
    else:
        pred = scores < best_threshold

    return labels, pred, np.max(All_f), best_threshold

def fit_xgb_model(data, selected_features, label_name, undersampling_seed=10, train_frac=0.8, test_frac=0.15,
                  scoring='f1_macro', verbose=True):
    reduced_data = reduce_data(data, label_name)

    balanced_data = balance_data(reduced_data, label_name, undersampling_seed)

    X, y, X_train, y_train, X_val, y_val, X_test, y_test, train_ind, val_ind, test_ind = make_temporal_split(
        balanced_data,
        selected_features,
        label_name,
        train_frac,
        test_frac)

    opt_model = XGBClassifier()

    cv = ((np.array(train_ind), np.array(val_ind)),)

    # Optimizing 'max_depth' and 'min_child_weight'
    param_grid = {'max_depth': range(1, 25, 1),
                  'min_child_weight': [2, 4, 6, 8, 10, 20, 30, 40, 50]}

    opt_model = do_gridsearch(X, y, opt_model, param_grid, cv, scoring, verbose)

    # Optimizing gamma
    param_grid = {'gamma': [i / 100.0 for i in range(0, 100)]}

    opt_model = do_gridsearch(X, y, opt_model, param_grid, cv, scoring, verbose)

    # Optimizing subsample and colsample_bytree
    param_grid = {'subsample': [i / 10.0 for i in range(6, 11)],
                  'colsample_bytree': [i / 10.0 for i in range(6, 11)]}

    opt_model = do_gridsearch(X, y, opt_model, param_grid, cv, scoring, verbose)

    # Optimizing reg_alpha
    param_grid = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}

    opt_model = do_gridsearch(X, y, opt_model, param_grid, cv, scoring, verbose)

    opt_model.fit(X_train, y_train)

    y_pred_val = opt_model.predict(X_val)

    y_pred_test = opt_model.predict(X_test)

    return opt_model, X, y, X_train, y_train, X_val, y_val, X_test, y_test, y_pred_val, y_pred_test

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores, labels=[True,False])
    conf = sklm.confusion_matrix(labels, scores, labels=[True,False])
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print('Average F-score  %0.2f' % sklm.f1_score(labels, scores, labels=[True,False], average='macro'))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F-score    %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

def get_feature_importances(model,colnames,n):
    feature_importances = pd.DataFrame()
    feature_importances['features'] = colnames
    feature_importances['importance'] = model.feature_importances_
    feature_importances.sort_values(by='importance',ascending=False, inplace=True)
    feature_importances.reset_index(drop=True,inplace=True)
    feature_importances.set_index('features').iloc[0:n].plot.barh(figsize=(8,3))
    #plt.xticks(rotation='vertical')
    return feature_importances.head(n)

def plot_dist(data,target,drought_var):
    plt.figure()
    plt.subplot(1,2,1)
    sns.boxenplot(y=target,x=drought_var,data=data)
    plt.subplot(1,2,2)
    bins = np.arange(data[target].min(),data[target].max(),0.05*(data[target].max()-data[target].min()))
    ax=sns.distplot((data[(data[drought_var].notna())&(data[drought_var]==False)][target]).dropna()
                 ,bins=bins,label='False')
    sns.distplot(data[(data[drought_var].notna())&(data[drought_var]==True)][target].dropna(),
                bins=bins,label='True')
    ax.yaxis.set_label_position("right")
    plt.ylabel('distribution')
    ax.yaxis.tick_right()
    plt.legend(loc="best")
    return

'''
Helper Functions 
'''

def normalize_data(data, ids_list, grouping):
    grouped = data.groupby(grouping)
    normed_data = pd.DataFrame()
    Znorm = StandardScaler()
    colnames = list(data.drop(labels=ids_list, axis=1).columns)

    for name, group in grouped:
        group.reset_index(inplace=True, drop=True)
        temp = pd.DataFrame(Znorm.fit_transform(group[colnames]), columns=colnames)
        temp[ids_list] = group[ids_list]
        temp = temp[ids_list + colnames]
        normed_data = pd.concat([normed_data, temp])

    normed_data.reset_index(inplace=True, drop=True)

    return normed_data

def reduce_data(data, label_name):
    reduced_data = pd.DataFrame()

    for name, group in data.groupby('District'):
        drought_years = np.array(group[group[label_name]]['year'])
        keep_years = np.sort(np.unique(np.append(drought_years, [drought_years - 1, drought_years + 1])))
        temp = group[group.year.apply(lambda x: x in keep_years)].sort_values(by=['year', 'Season']).copy()
        reduced_data = pd.concat([reduced_data, temp])

    reduced_data.sort_values('year', inplace=True)
    reduced_data.reset_index(drop=True, inplace=True)

    return reduced_data


def balance_data(data, label_name, undersampling_seed=10):
    droughts_number = data[label_name].sum()

    negative_data = data[data[label_name] == False].copy()
    positive_data = data[data[label_name] == True].copy()
    balanced_data = pd.concat([negative_data.sample(n=droughts_number, replace=False,
                                                    random_state=undersampling_seed, axis=0), positive_data])
    balanced_data.sort_values('year', inplace=True)
    balanced_data.reset_index(drop=True, inplace=True)

    return balanced_data


def make_temporal_split(data, selected_features, label_name, train_frac=0.8, test_frac=0.15):
    data.sort_values('year', inplace=True)
    data.reset_index(drop=True, inplace=True)

    L = len(data)
    p1 = 1 - test_frac
    p2 = train_frac

    train_val_ind = list(range(int(p1 * L)))
    test_ind = list(range(np.max(train_val_ind) + 1, L))
    L2 = len(train_val_ind)
    train_ind = list(range(int(p2 * L2)))
    val_ind = list(range(np.max(train_ind) + 1, L2))

    X = data[selected_features]
    y = data[label_name]

    X_train = data[selected_features].loc[train_ind]
    y_train = data[label_name].loc[train_ind]

    X_val = data[selected_features].loc[val_ind]
    y_val = data[label_name].loc[val_ind]

    X_test = data[selected_features].loc[test_ind]
    y_test = data[label_name].loc[test_ind]

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test, train_ind, val_ind, test_ind


def do_gridsearch(X, y, opt_model, param_grid, cv, scoring, verbose=True):
    GS = ms.GridSearchCV(estimator=opt_model, param_grid=param_grid,
                         cv=cv,
                         scoring=scoring,
                         return_train_score=True, n_jobs=4)
    GS.fit(X, y);

    for key in param_grid.keys():
        exec ("opt_model." + key + "= GS.best_params_[key]")

    if verbose:
        print(GS.best_params_)
        print(scoring + ' = ', GS.best_score_)

    return opt_model

