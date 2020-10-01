# coding=utf-8
# Author: Rion B Correia
# Date: April 17, 2020
#
# Description: Plot risk of interaction per age_group
#
#
import numpy as np
import pandas as pd
from utils import ensurePathExists
import statsmodels.api as sm
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt


if __name__ == '__main__':

    gender = 'Female'
    gender_symbol = 'F' if gender == 'Female' else 'M'

    path = '../../data/'
    dfi = pd.read_csv(path + 'indianapolis/age-gender.csv', index_col='age_group')
    dfi = dfi.loc[dfi['gender'] == gender, :]

    dfb = pd.read_csv(path + 'blumenau/age-gender.csv', index_col='age_group')
    dfb = dfb.loc[dfb['gender'] == gender, :]

    # Format Catalonia data
    dfc = pd.read_csv(path + 'Catalonia/Age-gender.txt', sep='\t', index_col='age_group')
    if gender == 'Female':
        dfc['patient'] = dfc['At_least_1_fem']
        dfc['patient-coadmin'] = dfc['Coad_fem']
    elif gender == 'Male':
        dfc['patient'] = dfc['At_least_1_mal']
        dfc['patient-coadmin'] = dfc['Coad_mal']

    dfc = dfc.loc[:, ['patient-coadmin', 'patient']]

    # Compute RI
    dfi['RC^{[y1,y2]}'] = dfi['patient-coadmin'] / dfi['patient']
    dfb['RC^{[y1,y2]}'] = dfb['patient-coadmin'] / dfb['patient']
    dfc['RC^{[y1,y2]}'] = dfc['patient-coadmin'] / dfc['patient']

    #
    # Curve Fitting
    #
    y_rc_indy = dfi['RC^{[y1,y2]}'].values
    y_rc_bnu = dfb['RC^{[y1,y2]}'].values
    y_rc_cat = dfc['RC^{[y1,y2]}'].values
    x = np.arange(len(y_rc_indy))
    x_ = np.linspace(x[0], x[-1], len(x) * 10)

    # RI Cubic Model
    Xi = sm.add_constant(np.column_stack([x**3, x**2, x]))
    rc_c_model_indy = sm.OLS(y_rc_indy, Xi)
    rc_c_model_bnu = sm.OLS(y_rc_bnu, Xi)
    rc_c_model_cat = sm.OLS(y_rc_cat, Xi)
    rc_c_model_indy_result = rc_c_model_indy.fit()
    rc_c_model_bnu_result = rc_c_model_bnu.fit()
    rc_c_model_cat_result = rc_c_model_cat.fit()

    # print(rc_c_model_result.summary())
    Xi_ = sm.add_constant(np.column_stack([x_**3, x_**2, x_]))
    y_rc_indy_ = np.dot(Xi_, rc_c_model_indy_result.params)
    y_rc_bnu_ = np.dot(Xi_, rc_c_model_bnu_result.params)
    y_rc_cat_ = np.dot(Xi_, rc_c_model_cat_result.params)
    #
    rc_c_model_indy_R2 = rc_c_model_indy_result.rsquared_adj
    rc_c_model_bnu_R2 = rc_c_model_bnu_result.rsquared_adj
    rc_c_model_cat_R2 = rc_c_model_cat_result.rsquared_adj

    #
    # Plot
    #
    fig, ax = plt.subplots(figsize=(4.3, 3), nrows=1, ncols=1)
    #
    fit_color_indy = markerfacecolor_indy = '#aec7e8'
    markeredgecolor_indy = '#1f77b4'
    #
    fit_color_bnu = markerfacecolor_bnu = '#98df8a'
    markeredgecolor_bnu = '#2ca02c'
    #
    fit_color_cat = markerfacecolor_cat = '#ff9896'
    markeredgecolor_cat = '#d62728'

    # color_ddi = #d62728 / #ff9896

    ax.set_title(r'$RC^{{[y_1,y_2],{g:s}}}$'.format(g=gender_symbol))
    #
    age_inds = np.arange(0, len(dfi))
    age_labels = dfi.index.tolist()

    # Plot
    rc_indy, = ax.plot(age_inds, dfi['RC^{[y1,y2]}'].tolist(), marker='o', ms=6, lw=0, markerfacecolor=markerfacecolor_indy, markeredgecolor=markeredgecolor_indy, zorder=5)
    rc_bnu, = ax.plot(age_inds, dfb['RC^{[y1,y2]}'].tolist(), marker='^', ms=6, lw=0, markerfacecolor=markerfacecolor_bnu, markeredgecolor=markeredgecolor_bnu, zorder=5)
    rc_cat, = ax.plot(age_inds, dfc['RC^{[y1,y2]}'].tolist(), marker='p', ms=6, lw=0, markerfacecolor=markerfacecolor_cat, markeredgecolor=markeredgecolor_cat, zorder=5)

    # Plot Cubic Fit
    rc_indy_cubic, = ax.plot(x_, y_rc_indy_, color=fit_color_indy, ms=0, lw=2, zorder=3)
    rc_bnu_cubic, = ax.plot(x_, y_rc_bnu_, color=fit_color_bnu, ms=0, lw=2, zorder=3)
    rc_cat_cubic, = ax.plot(x_, y_rc_cat_, color=fit_color_cat, ms=0, lw=2, zorder=3)

    # R^2
    ax.text(x=0.97, y=0.23, s=r'$R^{{2}}_{{Indy}}={r2:.2f}$'.format(r2=rc_c_model_indy_R2), ha='right', va='bottom', transform=ax.transAxes)
    ax.text(x=0.97, y=0.13, s=r'$R^{{2}}_{{Bnu}}={r2:.2f}$'.format(r2=rc_c_model_bnu_R2), ha='right', va='bottom', transform=ax.transAxes)
    ax.text(x=0.97, y=0.03, s=r'$R^{{2}}_{{Cat}}={r2:.2f}$'.format(r2=rc_c_model_cat_R2), ha='right', va='bottom', transform=ax.transAxes)

    # Legend
    Ls = ax.legend(
        [
            rc_indy,
            rc_bnu,
            rc_cat
        ],
        [
            r'$RC^{{[y_1,y_2],{g:s}}}_{{Indy}}$'.format(g=gender_symbol),
            r'$RC^{{[y_1,y_2],{g:s}}}_{{Bnu}}$'.format(g=gender_symbol),
            r'$RC^{{[y_1,y_2],{g:s}}}_{{Cat}}$'.format(g=gender_symbol)
        ],
        loc='upper left', handletextpad=0.25, columnspacing=0, handlelength=1, ncol=1)

    ax.set_xticks(age_inds)
    ax.set_xticklabels(age_labels, rotation=90)
    ax.grid()

    # Save
    plt.tight_layout()
    gender_str = gender.lower()
    wIMGfile = 'images/img-rc-age-{gender:s}.pdf'.format(gender=gender_str)
    ensurePathExists(wIMGfile)
    fig.savefig(wIMGfile)
    plt.close()
