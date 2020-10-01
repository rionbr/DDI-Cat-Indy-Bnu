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
    #
    dfb = pd.read_csv(path + 'blumenau/age-gender.csv', index_col='age_group')
    dfb = dfb.loc[dfb['gender'] == gender, :]

    # Format Catalonia data
    dfc = pd.read_csv(path + 'Catalonia/Age-gender.txt', sep='\t', index_col='age_group')
    if gender == 'Female':
        dfc['patient'] = dfc['At_least_1_fem']
        dfc['patient-inter'] = dfc['Coad_DDI_fem']
    elif gender == 'Male':
        dfc['patient'] = dfc['At_least_1_mal']
        dfc['patient-inter'] = dfc['Coad_DDI_mal']
    dfc = dfc.loc[:, ['patient-inter', 'patient']]

    # Compute RI
    dfi['RI^{[y1,y2],g}'] = dfi['patient-inter'] / dfi['patient']
    dfb['RI^{[y1,y2],g}'] = dfb['patient-inter'] / dfb['patient']
    dfc['RI^{[y1,y2],g}'] = (dfc['patient-inter'] / dfc['patient'])

    #
    # Curve Fitting
    #
    y_ri_indy = dfi['RI^{[y1,y2],g}'].values
    y_ri_bnu = dfb['RI^{[y1,y2],g}'].values
    y_ri_cat = dfc['RI^{[y1,y2],g}'].values
    x = np.arange(len(y_ri_indy))
    x_ = np.linspace(x[0], x[-1], len(x) * 10)

    # RI Cubic Model
    Xi = sm.add_constant(np.column_stack([x**3, x**2, x]))
    ri_c_model_indy = sm.OLS(y_ri_indy, Xi)
    ri_c_model_bnu = sm.OLS(y_ri_bnu, Xi)
    ri_c_model_cat = sm.OLS(y_ri_cat, Xi)
    ri_c_model_indy_result = ri_c_model_indy.fit()
    ri_c_model_bnu_result = ri_c_model_bnu.fit()
    ri_c_model_cat_result = ri_c_model_cat.fit()

    # print(rc_c_model_result.summary())
    Xi_ = sm.add_constant(np.column_stack([x_**3, x_**2, x_]))
    y_ri_indy_ = np.dot(Xi_, ri_c_model_indy_result.params)
    y_ri_bnu_ = np.dot(Xi_, ri_c_model_bnu_result.params)
    y_ri_cat_ = np.dot(Xi_, ri_c_model_cat_result.params)
    #
    ri_c_model_indy_R2 = ri_c_model_indy_result.rsquared_adj
    ri_c_model_bnu_R2 = ri_c_model_bnu_result.rsquared_adj
    ri_c_model_cat_R2 = ri_c_model_cat_result.rsquared_adj

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

    ax.set_title(r'$RI^{{[y_1,y_2],{g:s}}}$'.format(g=gender_symbol))
    #
    age_inds = np.arange(0, len(dfi))
    age_labels = dfi.index.tolist()

    # Plot
    ri_indy, = ax.plot(age_inds, dfi['RI^{[y1,y2],g}'].tolist(), marker='o', ms=6, lw=0, markerfacecolor=markerfacecolor_indy, markeredgecolor=markeredgecolor_indy, zorder=5)
    ri_bnu, = ax.plot(age_inds, dfb['RI^{[y1,y2],g}'].tolist(), marker='^', ms=6, lw=0, markerfacecolor=markerfacecolor_bnu, markeredgecolor=markeredgecolor_bnu, zorder=5)
    ri_cat, = ax.plot(age_inds, dfc['RI^{[y1,y2],g}'].tolist(), marker='p', ms=6, lw=0, markerfacecolor=markerfacecolor_cat, markeredgecolor=markeredgecolor_cat, zorder=5)

    # Plot Cubic Fit
    ri_indy_cubic, = ax.plot(x_, y_ri_indy_, color=fit_color_indy, ms=0, lw=2, zorder=3)
    ri_bnu_cubic, = ax.plot(x_, y_ri_bnu_, color=fit_color_bnu, ms=0, lw=2, zorder=3)
    ri_cat_cubic, = ax.plot(x_, y_ri_cat_, color=fit_color_cat, ms=0, lw=2, zorder=3)

    # R^2
    ax.text(x=0.97, y=0.23, s=r'$R^{{2}}_{{Indy}}={r2:.2f}$'.format(r2=ri_c_model_indy_R2), ha='right', va='bottom', transform=ax.transAxes)
    ax.text(x=0.97, y=0.13, s=r'$R^{{2}}_{{Bnu}}={r2:.2f}$'.format(r2=ri_c_model_bnu_R2), ha='right', va='bottom', transform=ax.transAxes)
    ax.text(x=0.97, y=0.03, s=r'$R^{{2}}_{{Cat}}={r2:.2f}$'.format(r2=ri_c_model_cat_R2), ha='right', va='bottom', transform=ax.transAxes)

    # Legend
    Ls = ax.legend(
        [
            ri_indy,
            ri_bnu,
            ri_cat
        ],
        [
            r'$RI^{{[y_1,y_2],{g:s}}}_{{Indy}}$'.format(g=gender_symbol),
            r'$RI^{{[y_1,y_2],{g:s}}}_{{Bnu}}$'.format(g=gender_symbol),
            r'$RI^{{[y_1,y_2],{g:s}}}_{{Cat}}$'.format(g=gender_symbol)
        ],
        loc='upper left', handletextpad=0.25, columnspacing=0, handlelength=1, ncol=1)

    ax.set_xticks(age_inds)
    ax.set_xticklabels(age_labels, rotation=90)
    ax.grid()

    # Save
    plt.tight_layout()
    gender_str = gender.lower()
    wIMGfile = 'images/img-ri-age-{gender:s}.pdf'.format(gender=gender_str)
    ensurePathExists(wIMGfile)
    fig.savefig(wIMGfile)
    plt.close()
