# %% Import packages
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.optimizers import Adam
import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from scipy.stats.distributions import chi2
from itertools import combinations
# %% Auxiliar functions
def get_scores(y_true, y_pred, score_fun):
    nclasses = np.shape(y_true)[1]
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_true[:, k], y_pred[:, k]) for k in range(nclasses)]]
    return np.array(scores).T
def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc
def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)
def affer_results(y_true, y_pred):
    """Return true positives, false positives, true negatives, false negatives.

    Parameters
    ----------
    y_true : ndarray
        True value
    y_pred : ndarray
        Predicted value

    Returns
    -------
    tn, tp, fn, fp: ndarray
        Boolean matrices containing true negatives, true positives, false negatives and false positives.
    cm : ndarray
        Matrix containing: 0 - true negative, 1 - true positive,
        2 - false negative, and 3 - false positive.
    """
    # True negative
    tn = (y_true == y_pred) & (y_pred == 0)
    # True positive
    tp = (y_true == y_pred) & (y_pred == 1)
    # False positive
    fp = (y_true != y_pred) & (y_pred == 1)
    # False negative
    fn = (y_true != y_pred) & (y_pred == 0)
    # Generate matrix of "tp, fp, tn, fn"
    m, n = np.shape(y_true)
    cm = np.zeros((m, n), dtype=int)
    cm[tn] = 0
    cm[tp] = 1
    cm[fn] = 2
    cm[fp] = 3
    return tn, tp, fn, fp, cm
def main():
    # start of predict.py portion of code: so far I have changed it so that it does not save the output to a file,
    # instead, the result is still kept in a numpy array so the generate_figures_and_tables portion can use it
    # TO DO: add argument for evaluation csv
    #
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('--tracings', default="./ecg_tracings.hdf5",  # or date_order.hdf5
                        help='HDF5 containing ecg tracings.')
    parser.add_argument('--model', default="./model.hdf5",  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                        help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--annotations', default="/data/isip/data/tnmg_code/v1.0.0b/annotations/gold_standard.csv", help='file containing disease annotations (for validating)')
    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")
    # import ecg tracings from hdf5
    #
    with h5py.File(args.tracings, "r") as f:
        x = np.array(f['tracings'])
    # import model weight file
    #
    model = load_model(args.model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score_predict = model.predict(x, batch_size=args.bs, verbose=1)
    # Generate dataframe
    # start of generate_figures_and_tables
    #
    score_fun = {'Precision': precision_score,
                'Recall': recall_score, 'Specificity': specificity_score,
                'F1 score': f1_score}
    diagnosis = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
    nclasses = len(diagnosis)
    predictor_names = ['DNN']
    # %% Read datasets
    # Get two annotators
    #
    trueAnnotationsPath = args.annotations
    evaluationSetPath = "/data/isip/tnmg_code/v1.0.0b/annotations/gold_standard.csv"
    annotationsPath = "/data/isip/data/tnmg_code/v1.0.0b/annotations/"
    y_true = pd.read_csv(trueAnnotationsPath).values
    y_cardiologist1 = pd.read_csv(annotationsPath + 'cardiologist1.csv').values
    y_cardiologist2 = pd.read_csv(annotationsPath + 'cardiologist2.csv').values
    y_cardio = pd.read_csv(annotationsPath + 'cardiology_residents.csv').values
    y_emerg = pd.read_csv(annotationsPath + 'emergency_residents.csv').values
    y_student = pd.read_csv(annotationsPath + 'medical_students.csv').values
    y_score_list = [np.load('/data/isip/exp/tnmg_code/exp_0008/automatic-ecg-diagnosis/OUTPUTS/dnn_predicts/other_seeds/model_' + str(i+1) + '.npy') for i in range(10)]
    # %% Get average model model
    # Get micro average precision
    micro_avg_precision = [average_precision_score(y_true[:, :6], y_score_predict[:, :6], average='micro')
                            for y_score in y_score_list]
    # get ordered index
    index = np.argsort(micro_avg_precision)
    print(index)
    print('Micro average precision')
    print(np.array(micro_avg_precision)[index])
    # get 6th best model (immediatly above median) out 10 different models
    k_dnn_best = index[5]
    print(k_dnn_best)
    y_score_best = y_score_list[k_dnn_best]
    y_score_list = []
    y_score_list.append(y_score_predict)
    y_score_best = y_score_predict
    # Get threshold that yield the best precision recall
    _, _, threshold = get_optimal_precision_recall(y_true, y_score_best)
    mask = y_score_best > threshold
    # Get neural network prediction
    # This data was also saved in './data/csv_files/dnn.csv'
    y_neuralnet = np.zeros_like(y_score_best)
    y_neuralnet[mask] = 1
    y_neuralnet[mask] = 1
    scores_list = []
    scores = get_scores(y_true, y_neuralnet, score_fun)
    scores_df = pd.DataFrame(scores, index=diagnosis, columns=score_fun.keys())
    scores_list.append(scores_df)
    print(score_fun.keys())
    # %% Generate table with scores for the average model (Table 2)
    scores_list = []
    for y_pred in [y_neuralnet]:
        # Compute scores
        scores = get_scores(y_true, y_neuralnet, score_fun)
        # Put them into a data frame
        scores_df = pd.DataFrame(scores, index=diagnosis, columns=score_fun.keys())
        # Append
        scores_list.append(scores_df)
    # Concatenate dataframes
    scores_all_df = pd.concat(scores_list, axis=1, keys=['DNN'])
    # Change multiindex levels
    scores_all_df = scores_all_df.swaplevel(0, 1, axis=1)
    scores_all_df = scores_all_df.reindex(level=0, columns=score_fun.keys())
    # Save results
    scores_all_df.to_excel("./outputs/tables/scores5000.xlsx", float_format='%.3f')
    scores_all_df.to_csv("./outputs/tables/scores5000.csv", float_format='%.3f')
    # %% Plot precision recall curves (Figure 2)
    for k, name in enumerate(diagnosis):
        precision_list = []
        recall_list = []
        threshold_list = []
        average_precision_list = []
        fig, ax = plt.subplots()
        lw = 2
        t = ['bo', 'rv', 'gs', 'kd']
        for j, y_score in enumerate(y_score_list):
            # Get precision-recall curve
            precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
            recall[np.isnan(recall)] = 0  # change nans to 0
            precision[np.isnan(precision)] = 0  # change nans to 0
            # Plot if is the choosen option
            # Compute average precision
            average_precision = average_precision_score(y_true[:, k], y_score[:, k])
            precision_list += [precision]
            recall_list += [recall]
            average_precision_list += [average_precision]
            threshold_list += [threshold]
        # Plot shaded region containing maximum and minimun from other executions
        recall_all = np.concatenate(recall_list)
        recall_all = np.sort(recall_all)  # sort
        recall_all = np.unique(recall_all)  # remove repeated entries
        recall_vec = []
        precision_min = []
        precision_max = []
        for r in recall_all:
            p_max = [max(precision[recall == r]) for recall, precision in zip(recall_list, precision_list)]
            p_min = [min(precision[recall == r]) for recall, precision in zip(recall_list, precision_list)]
            recall_vec += [r, r]
            precision_min += [min(p_max), min(p_min)]
            precision_max += [max(p_max), max(p_min)]
        ax.plot(recall_vec, precision_min, color='blue', alpha=0.3)
        ax.plot(recall_vec, precision_max, color='blue', alpha=0.3)
        ax.fill_between(recall_vec, precision_min, precision_max,
                        facecolor="blue", alpha=0.3)
        # Plot iso-f1 curves
        f_scores = np.linspace(0.1, 0.95, num=15)
        # Plot values in
    # %% Confusion matrices (Supplementary Table 1)
    M = [[confusion_matrix(y_true[:, k], y_pred[:, k], labels=[0, 1])
       for k in range(nclasses)] for y_pred in [y_neuralnet]]
    M_xarray = xr.DataArray(np.array(M),
                            dims=['predictor', 'diagnosis', 'true label', 'predicted label'],
                            coords={'predictor': ['DNN'],
                                    'diagnosis': diagnosis,
                                    'true label': ['not present', 'present'],
                                    'predicted label': ['not present', 'present']})
    confusion_matrices = M_xarray.to_dataframe('n')
    confusion_matrices = confusion_matrices.reorder_levels([1, 2, 3, 0], axis=0)
    confusion_matrices = confusion_matrices.unstack()
    confusion_matrices = confusion_matrices.unstack()
    confusion_matrices = confusion_matrices['n']
    confusion_matrices.to_excel("./outputs/tables/confusion matrices.xlsx", float_format='%.3f')
    confusion_matrices.to_csv("./outputs/tables/confusion matrices.csv", float_format='%.3f')
    #%% Compute scores and bootstraped version of these scores
    bootstrap_nsamples = 1000
    percentiles = [2.5, 97.5]
    scores_resampled_list = []
    scores_percentiles_list = []
    for y_pred in [y_neuralnet]:
        # Compute bootstraped samples
        np.random.seed(123)  # NEVER change this =P
        n, _ = np.shape(y_true)
        samples = np.random.randint(n, size=n * bootstrap_nsamples)
        # Get samples
        y_true_resampled = np.reshape(y_true[samples, :], (bootstrap_nsamples, n, nclasses))
        y_doctors_resampled = np.reshape(y_pred[samples, :], (bootstrap_nsamples, n, nclasses))
        # Apply functions
        scores_resampled = np.array([get_scores(y_true_resampled[i, :, :], y_doctors_resampled[i, :, :], score_fun)
                                    for i in range(bootstrap_nsamples)])
        # Sort scores
        scores_resampled.sort(axis=0)
        # Append
        scores_resampled_list.append(scores_resampled)
        # Compute percentiles index
        i = [int(p / 100.0 * bootstrap_nsamples) for p in percentiles]
        # Get percentiles
        scores_percentiles = scores_resampled[i, :, :]
        # Convert percentiles to a dataframe
        scores_percentiles_df = pd.concat([pd.DataFrame(x, index=diagnosis, columns=score_fun.keys())
                                        for x in scores_percentiles], keys=['p1', 'p2'], axis=1)
        # Change multiindex levels
        scores_percentiles_df = scores_percentiles_df.swaplevel(0, 1, axis=1)
        scores_percentiles_df = scores_percentiles_df.reindex(level=0, columns=score_fun.keys())
        # Append
        scores_percentiles_list.append(scores_percentiles_df)
    # Concatenate dataframes
    scores_percentiles_all_df = pd.concat(scores_percentiles_list, axis=1, keys=predictor_names)
    # Change multiindex levels
    scores_percentiles_all_df = scores_percentiles_all_df.reorder_levels([1, 0, 2], axis=1)
    scores_percentiles_all_df = scores_percentiles_all_df.reindex(level=0, columns=score_fun.keys())
    #%% Print box plot (Supplementary Figure 1)
    # Convert to xarray
    print()
    scores_resampled_xr = xr.DataArray(np.array(scores_resampled_list),
                                    dims=['predictor', 'n', 'diagnosis', 'score_fun'],
                                    coords={
                                        'predictor': predictor_names,
                                        'n': range(bootstrap_nsamples),
                                        'diagnosis': ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'],
                                        'score_fun': list(score_fun.keys())})
    # Remove everything except f1_score
    for sf in score_fun:
        fig, ax = plt.subplots()
        f1_score_resampled_xr = scores_resampled_xr.sel(score_fun=sf)
        # Convert to dataframe
        f1_score_resampled_df = f1_score_resampled_xr.to_dataframe(name=sf).reset_index(level=[0, 1, 2])
        # Plot seaborn
        ax = sns.boxplot(x="diagnosis", y=sf, hue="predictor", data=f1_score_resampled_df)
        # Save results
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("")
        plt.ylabel("", fontsize=16)
        if sf == "F1 score":
            plt.legend(fontsize=17)
        else:
            ax.legend().remove()
        plt.tight_layout()
        plt.savefig('./outputs/figures/boxplot_bootstrap_{}.pdf'.format(sf))
    scores_resampled_xr.to_dataframe(name='score').to_csv('./outputs/figures/boxplot_bootstrap_data.txt')
if __name__ == "__main__":
    main()