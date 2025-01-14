import numpy as np
import scipy.stats as stats

from utils import read_csv_fast


def sample_from_num_ps():
    # read_csv_fast caches, so its not loaded every time this func is called
    fp = fr'../dataframes/df_combined_semi_pruned_Aug24.csv'
    df = read_csv_fast(fp, easy_override=False)
    return df['num_ps'].sample(20, replace=True)


def sim_80_power_implied_low(beta=2.8, nsim=10_000):
    np.random.seed(0)
    powers = []
    p_fragiles = []

    p_sig_cutoff = .05
    z_sig_cutoff = stats.norm.isf(p_sig_cutoff / 2)
    p_fragile_cutoff = .01
    z_fragile_cutoff = stats.norm.isf(p_fragile_cutoff / 2)

    fp = fr'../dataframes/df_combined_semi_pruned_Aug24.csv'
    df = read_csv_fast(fp, easy_override=False)
    num_ps = df['num_ps'].astype(int).sample(nsim, replace=True)
    prop_sigs = []
    for num_p in num_ps:
        sig_zs = []
        while len(sig_zs) < 1:  # no sig
            zs = np.random.normal(beta, size=num_p)
            sig_zs = zs[zs > z_sig_cutoff]
            if len(sig_zs) == 0:
                prop_sigs.append(0)
            # prop_sigs will be over nsim but that's fine
        prop_sig = len(sig_zs) / len(zs)
        prop_sigs.append(prop_sig)
        p_fragile = np.mean(sig_zs < z_fragile_cutoff)
        p_fragiles.append(p_fragile)

    power = np.mean(prop_sigs)
    print(f'Calculated power: {power:.2%}')
    if beta == 2.8:
        assert 0.798 < power < 0.802
    p_fragiles = np.array(p_fragiles)
    M_fragile = np.mean(p_fragiles)
    print(f'Mean p-fragile expected: {M_fragile:.1%}')

    prop_fragile_over_50 = np.mean(p_fragiles > .5 - 1e-6)
    prop_fragile_under_32 = np.mean(p_fragiles < .319)

    print(f'Percentage of papers with over 50% expected: '
          f'{prop_fragile_over_50:.1%}')
    print(f'Percentage of papers with under 31.9% expected: '
          f'{prop_fragile_under_32:.1%}')


if __name__ == '__main__':
    sim_80_power_implied_low()

    # Show that 44% power causes 50% of significant p-values to be fragile
    sim_80_power_implied_low(beta=1.8)
