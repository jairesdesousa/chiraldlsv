import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import platform


def find_most_simil(des_idx, test_df, train_df):
    distances = []
    des = test_df[des_idx]

    for i in range(len(train_df)):
        dist = np.linalg.norm(des - train_df[i])
        distances.append(dist)

    m_dist = min(distances)
    idx = train_idx[distances.index(m_dist)]

    return idx


df_y = pd.read_csv('class_all.csv', index_col=0)
train_idx = df_y.loc[df_y['TR/TE'] == 'TR'].index
test_idx = df_y.loc[df_y['TR/TE'] == 'TE'].index


results = open(f'results.csv', 'w')
results.write('model,'
              'descriptor,'
              'classification,'
              'oob_score,accuracy,'
              'undetected_pairs,'
              'correctly_predicted_pairs,'
              'total_pairs,'
              '%undetectedpairs,'
              '%correctly_predicted_pairs\n')


for model in ('fingpr', 'transf', 'cddd'):
    f = open(f'{model}/des_names.csv', 'r')
    des_names = f.read().strip().split(',')
    f.close()

    des_df_ori = pd.read_csv(f'{model}/ADH2_can.csv', low_memory=False)
    des_df_opp = pd.read_csv(f'{model}/ADH2_opposite.csv', low_memory=False)
    des_df_nos = pd.read_csv(f'{model}/ADH2_nostereo.csv', low_memory=False)

    lsv = des_df_ori.loc[:, des_names].values
    dlsv_opp = lsv - des_df_opp.loc[:, des_names].values
    dlsv_nos = lsv - des_df_nos.loc[:, des_names].values

    for x, des in zip((lsv, dlsv_opp, dlsv_nos), ('lsv', 'dlsv_opp', 'dlsv_ns')):
        x_train = x[train_idx]
        x_test = x[test_idx]
        for y in ('F-L_class', '@-@@_class', 'R-S_class'):
            y_train = df_y.loc[train_idx, y]
            y_test = df_y.loc[test_idx, y]
            rf = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True, verbose=2)
            rf.fit(x_train, y_train)
            predictions = rf.predict(x_test)
            oob_score = round(rf.oob_score_, 3)
            acc = round(accuracy_score(y_test, predictions), 3)

            df_results = pd.DataFrame(columns=['smi', 'smi_opp', 'smi_ns', 'pred', 'real', 'prob', 'descriptors'])
            df_results['smi'] = df_y.loc[test_idx, 'SMILES']
            df_results['smi_opp'] = df_y.loc[test_idx, 'SMILES_opp']
            df_results['smi_ns'] = df_y.loc[test_idx, 'SMILES_ns']
            df_results['real'] = df_y.loc[test_idx, y]
            df_results['pred'] = predictions
            df_results['prob'] = np.amax(rf.predict_proba(x_test), axis=1)
            df_results['descriptors'] = f'{model} {des} {y}'

            plot = sns.violinplot(
                x='descriptors',
                split=True,
                inner='quart',
                y='prob',
                hue='pred',
                data=df_results)

            plt.savefig(f'{model}/pred_probs/{des}_{y}.png')
            plt.clf()

            n_pairs = df_results.smi_ns.nunique()
            pairs = df_results.smi_ns.unique()

            und = 0
            cor = 0

            for pair in pairs:
                split = df_results.loc[df_results['smi_ns'] == pair]
                if split.pred.nunique() == 1:
                    und += 1
                else:
                    if split.pred.to_list() == split.real.to_list():
                        cor += 1

            und_p = round(und/n_pairs*100, 3)
            cor_p = round(cor/n_pairs*100, 3)

            outliers = df_results[(df_results['prob'] > 0.8) & (df_results['pred'] != df_results['real'])]
            most_sim_idxs = []
            most_sim_smiles = []
            most_sim_labels = []
            for i in outliers.index:
                most_sim_idx = find_most_simil(i, x_test, x_train)
                most_sim_smi = df_y.loc[most_sim_idx, 'SMILES']
                most_sim_label = df_y.loc[most_sim_idx, y]

                most_sim_idxs.append(most_sim_idx)
                most_sim_smiles.append(most_sim_smi)
                most_sim_labels.append(most_sim_label)

            outliers['most_sim_idx'] = most_sim_idxs
            outliers['most_sim_SMILES'] = most_sim_smiles
            outliers['most_sim_label'] = most_sim_labels
            outliers.to_csv(f'{model}/outliers/{des}_{y}.csv')

            results.write(f'{model},{des},{y},{oob_score},{acc},{und},{cor},{n_pairs},{und_p},{cor_p}\n')

results.write(f'python version: {platform.python_version()}\n')
results.write(f'pandas version: {pd.__version__}\n')
results.write(f'sci-kit learn version: {sklearn.__version__}\n')
results.write(f'NumPy version: {np.__version__}\n')
results.write(f'matplotlib version: {matplotlib.__version__}\n')
results.write(f'seaborn version: {sns.__version__}\n')
results.close()
