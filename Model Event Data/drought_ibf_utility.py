import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as sklm
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

'''
Main Functions
'''


def prepare_Uganda_data(phath='./datasets/',
                        filename='Droughts_satelite_and_events.csv',
                        output_filename='Uganda_seasonal_normalized.csv',
                        save=False):

    full_data = pd.read_csv(phath + filename, index_col=False)
    Uganda_data = full_data[full_data.Country == 'Uganda'].copy()

    first_harvest = [6, 7]
    second_harvest = [11, 12]
    first_planting = [3, 4, 5]
    second_planting = [8, 9, 10]
    second_id = '_'.join(str(x) for x in second_harvest)
    first_id = '_'.join(str(x) for x in first_harvest)

    label_list = ['drought_reported', 'drought_news_article', 'drought_desinventar']
    feature_list = list(Uganda_data.drop(labels=['Country', 'District',
                                                 'year', 'month', 'day',
                                                 'date', ] + label_list,
                                         axis=1).columns)

    raw_features_noSPEI = Uganda_data[['District', 'year', 'month'] + feature_list[0:18] + [feature_list[-1]]].copy()
    raw_features_noSPEI = raw_features_noSPEI[
        raw_features_noSPEI.month.apply(lambda x: x in (first_planting + second_planting))]
    raw_features_noSPEI['Season'] = raw_features_noSPEI['month'].apply(
        lambda x: first_id if x in first_planting else second_id)
    raw_features_noSPEI.drop(labels='month', axis=1, inplace=True)
    features_noSPEI = raw_features_noSPEI.groupby(['year', 'District', 'Season']).mean().reset_index()

    normal_features = normalize_data(features_noSPEI, ids_list=['year', 'District', 'Season'],
                                     grouping=['District', 'Season'])

    spei_col = 'SPEI_3month'
    spei = Uganda_data[['year', 'District', 'month'] + [spei_col]].copy()
    spei = spei[spei['month'].apply(lambda x: x in ([first_planting[-1]] + [second_planting[-1]]))]
    spei['Season'] = spei['month'].apply(lambda x: first_id if x == first_planting[-1] else second_id)
    spei.drop(labels='month', axis=1, inplace=True)
    spei.reset_index(drop=True, inplace=True)
    normal_features = normal_features.merge(spei, on=['year', 'District', 'Season'])

    new_feature_list = sorted(list(normal_features.drop(labels=['year', 'District', 'Season'], axis=1).columns))
    normal_features = normal_features[['year', 'District', 'Season'] + new_feature_list]
    normal_features.sort_values(by=['year', 'District', 'Season'], inplace=True)

    label_col = 'drought_reported'
    labels = Uganda_data[['District', 'year', 'month'] + [label_col]].copy()
    labels = labels[labels.month.apply(lambda x: x in (first_harvest + second_harvest))]
    labels['Season'] = labels['month'].apply(lambda x: first_id if x in first_harvest else second_id)
    labels.drop(labels='month', axis=1, inplace=True)
    sum_labels = labels.groupby(by=['year', 'District', 'Season']).sum().reset_index()
    sum_labels.rename(columns={'drought_reported': 'number_drought_reported'}, inplace=True)
    sum_labels[label_col] = sum_labels['number_drought_reported'] > 0

    normal_data = normal_features.merge(sum_labels, on=['year', 'District', 'Season'])
    normal_data.drop(labels='number_drought_reported', axis=1, inplace=True)
    normal_data.dropna(inplace=True)
    if save:
        normal_data.to_csv(phath + output_filename, index=False)

    return normal_data

def fit_Logreg_model(data, selected_features, label_name, C_array,
                     n_splits=2, shuffle=True, shuffle_seed=10):
    reduced_data = reduce_data(data, label_name)

    X = reduced_data[selected_features]

    y = reduced_data[label_name]

    n_pos = len(y[y == True])
    n_neg = len(y[y == False])

    W_neg = (1.0 / n_neg) / (1.0 / n_pos + 1.0 / n_neg)
    W_pos = (1.0 / n_pos) / (1.0 / n_pos + 1.0 / n_neg)

    Weights = {True: W_pos, False: W_neg}

    opt_model = LogisticRegression(C=1.0,
                                   class_weight=Weights,
                                   penalty='l1',
                                   fit_intercept=True,
                                   solver='liblinear',
                                   random_state=0)

    param_grid = {'C': C_array}

    scoring = sklm.make_scorer(weighted_fscore)

    cv = ms.KFold(n_splits=n_splits,
                  shuffle=shuffle,
                  random_state=shuffle_seed)

    GS = ms.GridSearchCV(estimator=opt_model,
                         param_grid=param_grid,
                         cv=cv,
                         scoring=scoring,
                         return_train_score=False,
                         n_jobs=4,
                         iid=True)
    GS.fit(X, y);

    best_param = GS.best_params_['C']

    opt_model.C = best_param

    mean_test_scores = GS.cv_results_['mean_test_score']

    std_test_scores = GS.cv_results_['std_test_score']

    return X, y, opt_model, mean_test_scores, std_test_scores


def predict_Logreg_model(model, X, y, C, confusion_matrix=True):

    label_name = y.name

    model.C = C
    model.fit(X, y)
    y_pred = model.predict(X)

    coefs = pd.DataFrame()
    coefs['feature'] = X.columns
    coefs['coef'] = model.coef_.ravel()
    coefs['abs_coef'] = coefs['coef'].abs()
    coefs.sort_values('abs_coef', ascending=False, inplace=True, axis=0)
    coefs = coefs[['feature', 'coef']]
    coefs = coefs[coefs.coef.abs()>0]

    predictions = pd.DataFrame()
    predictions[label_name] = y
    predictions['logit_scores'] = (X.values.dot(model.coef_.T)).ravel() + model.intercept_
    predictions['predictions'] = y_pred

    pr = sklm.precision_recall_curve(y, predictions['logit_scores'],
                                           pos_label=True)

    roc = sklm.roc_curve(y, predictions['logit_scores'],
                                     pos_label=True)

    auc = sklm.auc(roc[0],roc[1])

    if confusion_matrix:
        print_metrics(y, y_pred)
        print('\n')
        print('Weighted Average F-score  %0.2f' % weighted_fscore(y, y_pred))

    return coefs, predictions, pr, roc, auc

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores, labels=[True,False])
    conf = sklm.confusion_matrix(labels, scores, labels=[True,False])
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    #print('Average F-score  %0.2f' % sklm.f1_score(labels, scores, labels=[True,False], average='macro'))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F-score    %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

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

def weighted_fscore(y, y_pred):
    n_pos = len(y[y == True])
    n_neg = len(y[y == False])
    W_neg = (1.0 / n_neg) / (1.0 / n_pos + 1.0 / n_neg)
    W_pos = (1.0 / n_pos) / (1.0 / n_pos + 1.0 / n_neg)

    f_pos = sklm.f1_score(y, y_pred, pos_label=True, average='binary')
    f_neg = sklm.f1_score(y, y_pred, pos_label=False, average='binary')

    f_ave = W_pos * f_pos + W_neg * f_neg

    return f_ave


def positive_fscore(y, y_pred):
    f_pos = sklm.f1_score(y, y_pred, pos_label=True, average='binary')

    return f_pos