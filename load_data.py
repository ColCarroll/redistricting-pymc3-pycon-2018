from io import BytesIO
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import requests
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
PRECOMPUTED_DATA = os.path.join(HERE, 'precomputed_data')
OPEN_ELECTIONS_URL = 'http://results.openelections.net/nc/raw/20161108__nc__general__precinct__raw.csv'
DEMOGRAPHIC_DATA_FILENAME = os.path.join(DATA, 'BlockGr.csv')
JOIN_DATA_FILENAME = os.path.join(DATA, 'Block_Level_GeoKeys.tab')
MANUAL_MAPPING = os.path.join(DATA, 'manual_mapping.tsv')
MAP_DATA = os.path.join(DATA, 'tl_2016_37_bg.shp')

PRECOMPUTED_NC_MAP = os.path.join(PRECOMPUTED_DATA, 'north_carolina_map_data.geojson')
PRECOMPUTED_NC_ELEC = os.path.join(PRECOMPUTED_DATA, 'nc_data.msg')


DEMOS = ['White Alone', 'Black or African American Alone', 'Hispanic or Latino', 'Total']
ALL_DEMOS = DEMOS[:] + ['Other']


if not os.path.isdir(DATA):
    os.mkdir(DATA)


def load_remote_data(url=None, filename=None):
    """Get data from a remote file, cacheing on first request.

    Parameters:
    ----------
    url : str, optional
        url to request data from (the default is None, which means the data must be local)
    filename : str, optional
        filename to load data from (the default is None, which parses a filename from the url, and
        saves in the default data path)

    Returns
    -------
    BytesIO
        In memory buffer of bytes.  Can usually be consumed with pandas.read_csv.
    """

    if filename is None:
        filename = os.path.join(DATA, url.split('/')[-1])

    if not os.path.exists(filename):
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, 'wb') as buff:
            buff.write(r.content)
    with open(filename, 'rb') as buff:
        return BytesIO(buff.read())


def get_north_carolina_election_data(office='US HOUSE OF REPRESENTATIVES'):
    """Load the North Carolina election data, returning a data frame.

    Data is from OpenElections, and includes races for all offices from 2016.
    """
    elections = pd.read_csv(load_remote_data(url=OPEN_ELECTIONS_URL))
    manual = pd.read_csv(MANUAL_MAPPING, sep='\t', header=-1, names=['division', 'manual_division'])

    # remove columns of all null and columns of only 1 value
    to_remove = []
    for col in elections.columns:
        if elections[col].isna().all() or len(elections[col].unique()) == 1:
            to_remove.append(col)
    elections = elections.drop(columns=to_remove)
    offices = elections.office.unique()
    if office not in offices:
        raise TypeError("'office' argument must be one of:\n{}".format('\n'.join(offices)))
    elections = elections[elections.office==office]
    elections['district'] = elections['district'].astype(int)
    elections = (
        elections
            .groupby(['party', 'division', 'district'])
            .agg({'votes': sum})
            .reset_index()
    )
    elections = elections.join(manual.set_index('division'), how='outer', on='division')
    elections.loc[~pd.isna(elections.manual_division), 'division'] = elections.loc[~pd.isna(elections.manual_division), 'manual_division']
    elections = (elections
        .drop(columns=['manual_division'])
        .reset_index(drop=True)
        .pivot_table(index=['division', 'district'], columns=['party'], values=['votes'], aggfunc='sum')
    )
    elections.columns = ['DEM', 'LIB', 'REP']
    elections = elections.fillna(0.).reset_index()

    return elections


def get_north_carolina_demographic_data():
    """Load the North Carolina voting age demographic data, returning a data frame.  There's some
    futzing with the columns to unpack the GEOID:

    The GEOID has format 15000USsscccttttttb, where ss = two-digit state FIPS code,
    ccc = three-digit county FIPS code, tttttt = six-digit tract group code, and
    b = one-digit block group code

    TODO: The url is actually on this page
    https://www.census.gov/rdo/data/voting_age_population_by_citizenship_and_race_cvap.html
    and returns a zip file.  The `load_remote_data` should return a fine form to unzip, but
    that needs to be done to automate this.
    """
    nc_code = '15000US37'
    demographics = pd.read_csv(load_remote_data(filename=DEMOGRAPHIC_DATA_FILENAME), encoding='latin1')
    nc = demographics[demographics.geoid.str.startswith(nc_code)]
    pattern = re.compile(r'^15000US(\d{12})$')
    def split_geoid(geoid):
        return pattern.match(geoid).group(1)
    nc['BG_Key'] = nc.geoid.apply(split_geoid)

    def agg_groups(j):
        cvap_col = 'CVAP_EST'
        return pd.Series({key: j[cvap_col][j.lntitle == key].sum() for key in DEMOS})

    nc = nc.groupby('BG_Key').apply(agg_groups)
    nc['Other'] = 2 * nc['Total'] - nc.sum(axis=1)
    nc = nc.reset_index()
    nc['BG_Key'] = nc.BG_Key.astype('int64')
    return nc


def get_north_carolina_join_data():
    """Load the table to join North Carolina demographic data with election results

    TODO: The url is actually on this page
    https://www.ncleg.net/representation/Content/BaseData/BD2016.aspx
    and returns a zip file.  The `load_remote_data` should return a fine form to unzip, but
    that needs to be done to automate this
    """
    join = pd.read_csv(load_remote_data(filename=JOIN_DATA_FILENAME), sep='\t')
    join = join[['Block_Key', 'Cnty_Name', 'VTD_Code', 'BG_Key', 'Cnty_Code']].drop_duplicates()
    join['division'] = 'ocd-division/country:us/state:nc/county:' + join.Cnty_Name.str.lower() + '/precinct:' + join.VTD_Code.str.lower()
    join = join.drop(columns=['Cnty_Name', 'VTD_Code', 'Cnty_Code']).drop_duplicates()
    return join


def get_north_carolina_congressional_join_data():
    """Load the table to join the other join data with congressional districts

    TODO: This url is fake for now.  Need to generalize a bit - the html page has a large zip download.
    """
    url = "https://www.census.gov/geographies/mapping-files/2017/dec/rdo/115-congressional-district-bef.html/block_cd.csv"
    cong_join = pd.read_csv(load_remote_data(url=url), header=-1, names=['Block_Key', 'district'])
    return cong_join


def make_north_carolina_data(office='US HOUSE OF REPRESENTATIVES', break_cache=False):
    if os.path.exists(PRECOMPUTED_NC_ELEC) and not break_cache:
        return pd.read_msgpack(PRECOMPUTED_NC_ELEC)
    election = get_north_carolina_election_data(office=office)
    join = get_north_carolina_join_data()
    demo = get_north_carolina_demographic_data()
    cong = get_north_carolina_congressional_join_data()

    join = join.merge(cong, on='Block_Key')

    rows = []
    for bg_key, gp in join.merge(demo, on='BG_Key').groupby('BG_Key'):
        base = gp[ALL_DEMOS].mean()
        blocks = gp.Block_Key.count()
        for (division, district), sub_blocks in gp.groupby(['division', 'district']).Block_Key.agg('count').items():
            row = dict(base * sub_blocks / blocks)
            row.update({'BG_Key': bg_key, 'division': division, 'district': district})
            rows.append(row)

    demo = pd.DataFrame(rows)
    demo = demo.groupby(['division', 'district'])[ALL_DEMOS].agg(sum).reset_index()
    demo = demo.merge(election, on=['division', 'district'])
    demo['label'] = demo.loc[:, 'division'].apply(division_to_label)
    return demo


def division_to_label(division):
    if type(division) is float:
        return 'None (None)'
    match = re.match(r'^ocd-division/country:us/state:nc/county:(?P<county>.+)/precinct:(?P<precinct>.+)$',
                     division)
    return '{county} ({precinct})'.format(**{k: v.title() for k, v in match.groupdict().items()})


def get_nc_map_data(nc_data, break_cache=False):
    """Data is from https://catalog.data.gov/dataset/tiger-line-shapefile-2016-state-north-carolina-current-block-group-state-based
    """
    if os.path.exists(PRECOMPUTED_NC_MAP) and not break_cache:
        return gpd.read_file(PRECOMPUTED_NC_MAP, driver='GeoJSON')
    nc = gpd.read_file(MAP_DATA)
    nc['GEOID'] = nc.GEOID.astype(int)

    join = get_north_carolina_join_data()
    join = join.drop(columns=['Block_Key']).drop_duplicates()

    nc = nc.merge(join, left_on='GEOID', right_on='BG_Key')
    nc = nc.dissolve(by='division').reset_index(drop=False)
    nc = nc.merge(nc_data)
    nc['pct_dem'] = nc.DEM / (nc.DEM + nc.REP + nc.LIB)
    return nc