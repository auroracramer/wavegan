import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
import pickle
from pathlib import Path
from sklearn.feature_selection import SelectKBest, chi2
from scipy import stats


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze a fiwGAN on a given latent code c')
    parser.add_argument(
        '--latent_v_dir',
        type=str,
        help='Path to directory where latent variable is stored')
    parser.add_argument('--s_dir',
                        type=str,
                        help='path to directory where s data is stored')
    parser.add_argument('--job_id', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_arguments()
    latent_v_dir = args['latent_v_dir']
    latent_v_dir = Path(latent_v_dir)
    s_dir = args['s_dir']
    s_dir = Path(s_dir)
    output_dir = args['output_dir']
    output_dir = Path(output_dir)
    job_id = args['job_id']

    vlist = []
    for files in latent_v_dir.iterdir():
        with open(latent_v_dir / files, 'rb') as out_data:
            latent_v = pickle.load(out_data)
            vlist.append(latent_v)
    vlist = np.concatenate(vlist)

    s_data = pd.read_csv(s_dir)
    tf = np.array(s_data['s'])
    tf[tf > 0] = 1

    slope = []
    p_value = []
    for i in range(vlist.shape[1]):
        z = vlist[:, i]
        z = z.reshape((*z.shape, 1))
        #tf = tf.reshape((*tf.shape, 1))
        #print(tf.shape)
        X_train, X_test, y_train, y_test = train_test_split(z,
                                                            tf,
                                                            test_size=0.5)

        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_pred = linreg.predict(X_test)

        fig, ax = plt.subplots()
        plt.scatter(X_test, y_test, color='b')
        plt.plot(X_test, y_pred, color='k')
        plt.savefig(output_dir / job_id / f"{i}.png")
        plt.close()

        #calculate the p value in linear regression & print out the table
        newX = pd.DataFrame({
            "Constant": np.ones(len(X_train))
        }).join(pd.DataFrame(X_train))
        MSE = (sum((y_train - y_pred)**2)) / (len(newX) - len(newX.columns))
        params = np.append(linreg.intercept_, linreg.coef_)
        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [
            2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX.columns))))
            for i in ts_b
        ]
        #print(f"p_values {p_values}")
        #exit()
        p_value.append(p_values)
        sd_b = np.round(sd_b, 3)
        ts_b = np.round(ts_b, 3)
        p_values = np.round(p_values, 3)
        params = np.round(params, 4)

        #myDF3 = pd.DataFrame()
        #myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
        #print(myDF3)
        #calculate the slope value

        slopes = linreg.coef_[0]
        absolute_slope = abs(slopes)
        slope.append(absolute_slope)

    #draw firguer for slope

    slope = np.array(slope)
    p_value = np.array(p_value)
    p_value = p_value[:, 1]
    ind_slope_pvalue = np.array([[i, slope[i], p_value[i]]
                                 for i in range(len(slope))],
                                dtype=object)
    ind_slope_pvalue = ind_slope_pvalue[np.argsort(ind_slope_pvalue[:, 1])]

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(ind_slope_pvalue[:, 1], label="slope", marker="o")
    plt.xticks([i for i in range(len(slope))],
               ind_slope_pvalue[:, 0].astype(int),
               rotation=90)
    plt.xlabel("latent variable")
    plt.ylabel("slopes in absolute values (|β|)")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(ind_slope_pvalue[:, 2], label="p_value", marker="o")
    plt.xticks([i for i in range(len(slope))],
               ind_slope_pvalue[:, 0].astype(int),
               rotation=90)
    plt.xlabel("latent variable")
    plt.ylabel("p value")
    plt.legend()
    plt.savefig(output_dir / job_id / f"slope.png")

    ##draw figure for chi squre

    chi_model = SelectKBest(chi2, k=2)
    res_feaetures = chi_model.fit_transform(vlist - np.min(vlist), tf)
    scores = chi_model.scores_
    pvalues = chi_model.pvalues_
    #indices = np.arange(len(scores))

    ind_chi_p = np.array([[i, scores[i], pvalues[i]]
                          for i in range(len(scores))])

    ind_chi_p = ind_chi_p[np.argsort(ind_chi_p[:, 1])]

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(ind_chi_p[:, 1], label="scores", marker="o")
    plt.xticks([i for i in range(len(scores))],
               ind_chi_p[:, 0].astype(int),
               rotation=90)
    plt.xlabel("latent variable")
    plt.ylabel("χ2")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(ind_chi_p[:, 2], label="pvalues", marker="o")
    plt.xticks([i for i in range(len(scores))],
               ind_chi_p[:, 0].astype(int),
               rotation=90)
    plt.xlabel("latent variable")
    plt.ylabel("p value")
    plt.legend()
    plt.savefig(output_dir / job_id / f"chi.png")
    #plt.show()