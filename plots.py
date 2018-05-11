import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st



def plot_nc_map_data(column, nc_map_data, congressional_data, title, step=2):
    kwargs = {'figsize': (24, 18)}
    if step > 0:
        if step == 1:
            ax = nc_map_data.plot(facecolor='none', edgecolor='black', linewidth=1, **kwargs)
        else:
            ax = nc_map_data.plot(column=column, cmap='bwr_r', edgecolor='black', linewidth=1, vmin=0., vmax=1., **kwargs)
        kwargs = {'ax': ax}
    ax = congressional_data.plot(facecolor='none', linewidth=4, edgecolor='black', **kwargs)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if step == 0:
        ax.set_title('North Carolina Congressional Districts')
    elif step == 1:
        ax.set_title('North Carolina Voting Districts and Congressional Districts')
    else:
        ax.set_title('Estimated {}Vote for Democratic Candidates'.format(title))

    return ax


def plot_rows(trace, district_df, n_rows, district):
    idxs = np.random.choice(np.arange(len(district_df.label)), n_rows)
    minority = trace['pct_minority_voting_dem'][:, idxs]
    majority = trace['pct_majority_voting_dem'][:, idxs]
    labels = district_df.label.iloc[idxs].values
    return plot_precincts(minority, majority, labels, district)


def plot_precincts(minority_pct, majority_pct, y_labels, district, n_x_pts=500, overlap=1.3):
    N = minority_pct.shape[1]
    fig, ax = plt.subplots(figsize=(12, N // 2))
    x = np.linspace(0, 1, n_x_pts)

    N = minority_pct.shape[1]
    if y_labels is None:
        y_labels = range(N)

    iterator = zip(
        y_labels,
        minority_pct.T,
        majority_pct.T,
    )

    for idx, (precinct, minority, majority) in enumerate(iterator, 1):
        pfx = '' if idx == 1 else '_'
        minority_kde = st.gaussian_kde(minority)
        majority_kde = st.gaussian_kde(majority)
        ax.plot([0], [precinct])
        trans = ax.convert_yunits(precinct)

        minority_y = minority_kde(x)
        minority_y = overlap * minority_y / minority_y.max()
        majority_y = majority_kde(x)
        majority_y = overlap * majority_y / majority_y.max()

        ax.fill_between(x, minority_y + trans, trans, color='steelblue', zorder=4*N-4*idx, label=pfx + 'Minority')
        ax.plot(x, minority_y + trans, color='black', linewidth=4, zorder=4*N-4*idx+1)

        ax.fill_between(x, majority_y + trans, trans, color='salmon', zorder=4*N-4*idx+2, label=pfx + 'Majority')
        ax.plot(x, majority_y + trans, color='black', linewidth=4, zorder=4*N-4*idx+3)

    ax.set_title('Inferring majority and minority bloc voting in North Carolina {}'.format(district))
    ax.set_xlabel('Pct vote for democrat')
    ax.legend()
    return fig, ax