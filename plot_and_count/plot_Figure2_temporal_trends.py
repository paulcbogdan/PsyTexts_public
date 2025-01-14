from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from utils import read_csv_fast

pd.options.mode.chained_assignment = None


def pool_neighoring_years(df_s, grp_range=1):
    years = df_s['year'].unique()
    for i in range(-grp_range, grp_range + 1):
        if i == 0: continue
        df_s_cop = df_s.copy()
        df_s_cop['year'] = df_s_cop['year'] + i
        df_s = pd.concat([df_s, df_s_cop])
    df_s = df_s[df_s.year.isin(years)]
    return df_s


def plot_key(df, key='marginal', group_keys=None, ax=None, ):
    if ax is not None:
        plt.sca(ax)
    plt.gca().spines[['right', 'top']].set_visible(False)

    df = df.dropna(subset=[key])

    df = pool_neighoring_years(df)

    group2full = {'General': 'General_Psychology',
                  'Exp. & Cog.': 'Experimental_and_Cognitive_Psychology',
                  'Dev. & Edu.': 'Developmental_and_Educational_Psychology',
                  'Social': 'Social_Psychology',
                  'Clinical': 'Clinical_Psychology',
                  'Applied': 'Applied_Psychology'}

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3)),
                  (0, (3, 1, 1, 1, 1, 1))]

    for subject in group_keys:
        df_s = df[df[group2full[subject]] == True]
        yrs = []
        Ms = []
        SE_low = []
        SE_high = []
        for year, df_sy in df_s.groupby('year'):
            if key in ['d', 't_N']:
                vals = df_sy[key].values
                low = np.nanquantile(vals, .05)
                high = np.nanquantile(vals, .95)
                vals = vals[vals > low]
                vals = vals[vals < high]
                M = np.nanmedian(vals)
                SE = np.nanstd(vals) / np.sqrt(np.sum(~np.isnan(vals)))
            else:
                M = df_sy[key].mean()
                SE = df_sy[key].std() / (df_sy[key].count() ** 0.5)
            SE_low.append(M - SE)  # *1.96)
            SE_high.append(M + SE)  # *1.96)
            Ms.append(M)
            yrs.append(year)
        plt.plot(yrs, Ms, label=subject,
                 linewidth=1, linestyle=linestyles.pop(0))
        plt.fill_between(yrs, SE_low, SE_high, alpha=0.2)
    plt.xlim(min(df.year), max(df.year))

    plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.yticks(fontsize=11)

    if key == 'p_fragile':  # 0.2998, 0.2998
        plt.title('Mean p-fragile (%)\n'
                  '(.01 ≤ p < .05)',
                  fontsize=12.5, pad=9)
        plt.plot([min(yrs), max(yrs)], [.26, .26], '--', color='k',
                 linewidth=1, zorder=-1)
        plt.text(2012, 0.253, '(Expected if 80% power)',
                 fontsize=11, ha='center', va='top')
        plt.ylim(.21, .34)
    elif key == 'p_fragile_implied':  # 0.2998, 0.2998
        plt.title('Mean implied p-fragile (%)\n(.01 < p < .05)',
                  fontsize=12.5, pad=9)
        plt.plot([min(yrs), max(yrs)], [.26, .26], '--', color='k',
                 linewidth=1, zorder=-1)
        plt.text(2011.5, 0.253, '(Expected if\n  80% power)',
                 fontsize=11, ha='center', va='top')
        plt.ylim(.21, .34)
    elif key == 'p_bad':
        plt.title('Flagrantly unlikely to replicate\n(p-fragile ≥ 50%)',
                  fontsize=12.5, pad=9)
        plt.plot([min(yrs), max(yrs)], [.11, .11], '--', color='k',
                 linewidth=1, zorder=-1)
        plt.text(2014, 0.123, '(Expected if 80% power)',
                 fontsize=11, ha='center')
        plt.yticks([.1, .15, .2, .25, .3, .35])
    elif key == 'p_good':
        plt.title('Optimistically replicable\n(p-fragile < 32%)',
                  fontsize=12.5, pad=9)
        plt.plot([min(yrs), max(yrs)], [.66, .66], '--', color='k',
                 linewidth=1, zorder=-1)
        plt.text(2014, 0.64, '(Expected if 80% power)',
                 fontsize=11, ha='center')
    elif key == 't_N':
        plt.title('Median sample sizes',
                  fontsize=12.5, pad=9)
        plt.ylim(0, 250)
    elif key == 'd':
        plt.title('Median Cohen’s d',
                  fontsize=12.5, pad=9)
        plt.ylim(0, 0.6)

    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=10.75)

    if 'p_fragile' in key or key in ['marginal']:
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    if ax is None:
        plt.show()


def plot_subject_over_time(bias_adjustment=.023):
    fp = fr'../dataframes/df_combined_semi_pruned_Aug24.csv'
    df = read_csv_fast(fp, easy_override=False)

    # Apply the adjustment after calculating the p-values, as otherwise
    #   the adjustment may cause 2-p-value or 3-p-value papers to flip
    df['p_bad'] = (df['p_fragile'] > .5 - 1e-6).astype(int)
    df['p_good'] = (df['p_fragile'] < .319).astype(int)
    df['p_fragile'] -= bias_adjustment

    group_keys = ['General', 'Exp. & Cog.', 'Dev. & Edu.', 'Social',
                  'Clinical', 'Applied']

    keys = ['p_fragile', 'p_bad', 'p_good',
            'p_fragile_implied', 't_N', 'd']

    pd.set_option('display.max_rows', 2000)
    for key in keys:
        df_m = df.groupby('year')[key].mean()
        print(df_m)

    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.3))

    for i, key in enumerate(keys):
        row = i // 3
        col = i % 3
        plt.sca(axs[row, col])

        plot_key(df, key=key, ax=plt.gca(), group_keys=group_keys)

    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    leg = fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=12.5,
                     frameon=False, columnspacing=0.8, handletextpad=0.3,
                     markerscale=2, handlelength=1.5)
    for line in leg.get_lines():
        line.set_linewidth(1.5)

    plt.subplots_adjust(left=0.075,
                        bottom=0.125,
                        right=0.975,
                        top=0.915,
                        wspace=0.24,
                        hspace=.45
                        )
    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)
    plt.savefig('../figs_and_tables/Figure_2_temporal_trends.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    plot_subject_over_time()
