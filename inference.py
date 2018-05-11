import pymc3 as pm


def ecological_inference_model(num_votes_for_group, voting_population, pct_minority, lam=0.5, group_name='dem'):
    """Create 2x2 ecological inference model with data

    Assumes there are two demographic groups of interest, and two politcal groups. Will
    construct a PyMC3 model with parameters `pct_minority_voting_dem` and `pct_majority_voting_dem`
    with one entry for each precinct.

    Parameters
    ----------
    num_votes_for_group : np.array
        Must be of length `n_precincts`, should be integer number of votes for each precinct.

    voting_population : np.array
        Must be of length `n_precincts`, total number of votes cast in the election for each precinct.

    pct_minority : np.array
        Must be of length `n_precincts`, percent of voting population that is minority in each precinct.

    lam : float (default 0.5)
        Hyperparameter controlling the exponential priors. Set to 0.5 in King's paper.

    group_name : str (default "dem")
        If you would like the variable names to be like "pct_minority_voting_pete", set this variable to "pete".

    Returns
    -------
    pm.Model defining the joint probability distribution.
    """
    n_precincts = len(voting_population)
    if not all(len(data) == n_precincts for data in (num_votes_for_group, voting_population, pct_minority)):
        raise TypeError('Each of `num_votes_for_group`, `voting_population`, and `pct_minority` must'
                        'be an iterable of the same length (which is the number of precincts)')

    with pm.Model() as model:
        alpha = pm.Exponential('alpha', lam=lam, shape=2)
        beta = pm.Exponential('beta', lam=lam, shape=2)

        pct_minority_voting_dem= pm.Beta('pct_minority_voting_{}'.format(group_name),
                                         alpha=alpha[0],
                                         beta=beta[0],
                                         shape=n_precincts)
        pct_majority_voting_dem = pm.Beta('pct_majority_voting_{}'.format(group_name),
                                          alpha=alpha[1],
                                          beta=beta[1],
                                          shape=n_precincts)

        est_voting_dem = pm.Deterministic('est_voting_{}'.format(group_name),
                                          (pct_minority * pct_minority_voting_dem +
                                           (1 - pct_minority) * pct_majority_voting_dem))

        pm.Binomial('votes_for_{}'.format(group_name),
                    n=voting_population,
                    p=est_voting_dem,
                    shape=n_precincts,
                    observed=num_votes_for_group)

    return model


def run_ecological_inference(district_df, **sample_kwargs):
    """Get samples from ecological inference.

    Parameters
    ----------
    district_df : pd.DataFrame
        Should have the columns 'Total' and 'White Alone' from the US Census data, and
        columns 'DEM' and 'REP' from election data.  A good start is the output from
        `make_north_carolina_data`, perhaps subset to a single congressional district.

    sample_kwargs :
        passed on to pm.sample

    Returns
    -------
    pm.MultiTrace
    """
    # Observed
    pct_minority = ((district_df['Total'] - district_df['White Alone']) / district_df['Total']).values
    voting_population = (district_df['DEM'] + district_df['REP']).values

    num_voting_for_dems = district_df['DEM'].values

    with ecological_inference_model(num_voting_for_dems, voting_population, pct_minority):
        trace = pm.sample(5000, nuts_kwargs={'target_accept': 0.98}, **sample_kwargs)
    return trace