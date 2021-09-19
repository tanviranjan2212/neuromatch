# @title Data retrieval

import pandas as pd
from os.path import expanduser as eu
import matplotlib.pylab as plt
import xarray as xr
import numpy as np


def get_data(fpath=eu('~/data/')):
    import os, requests
    fname = []

    if 'tanvi' in fpath:
        fpath=''
    
    for j in range(3):
        fname.append(fpath + 'steinmetz_part%d.npz' % j)
        url = ["https://osf.io/agvxh/download"]
        url.append("https://osf.io/uv3mw/download")
        url.append("https://osf.io/ehmw2/download")

    for j in range(len(url)):
        if not os.path.isfile(fname[j]):
            try:
                r = requests.get(url[j])
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")
            else:
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")
                else:
                    with open(fname[j], "wb") as fid:
                        fid.write(r.content)

    # @title Data retrieval for lfp and spike times

    fname = [fpath + 'steinmetz_st.npz']
    fname.append(fpath + 'steinmetz_wav.npz')
    fname.append(fpath + 'steinmetz_lfp.npz')

    url = ["https://osf.io/4bjns/download"]
    url.append("https://osf.io/ugm9v/download")
    url.append("https://osf.io/kx3v9/download")

    for j in range(len(url)):
        if not os.path.isfile(fname[j]):
            try:
                r = requests.get(url[j])
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")
            else:
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")
                else:
                    with open(fname[j], "wb") as fid:
                        fid.write(r.content)

    # @title Data loading
    import numpy as np

    alldat = np.array([])
    for j in range(len(fname)):
        alldat = np.hstack((alldat, np.load(fpath + 'steinmetz_part%d.npz' % j, allow_pickle=True)['dat']))

    dat_LFP = None  # To save time
    # dat_LFP = np.load(fpath + 'steinmetz_lfp.npz', allow_pickle=True)['dat']
    dat_ST = np.load(fpath + 'steinmetz_st.npz', allow_pickle=True)['dat']
    return alldat, dat_LFP, dat_ST


def concat_all_spikes(do_cluster_alltrial=True, do_reload=False, fpath=eu('~/data/')):
    # Events and spiketimes have to be done together to keep track pf trial starts
    def concat_spiketimes_sessions(dat_ST, alldat):
        # Concat spiketimes from all seessions
        # sp, neu, trial_starts = deps.concat_spiketimes(np.vstack(dat_ST['ss']))
        sp = []
        neu = []
        trial_starts = []
        events_df = []
        trial_start = 0
        last_neu = 0
        for i_sess, (dat_st, dat) in enumerate(zip(dat_ST, alldat)):
            # make combined spiketimes
            s, ne, trial_startz = concat_spiketimes(dat_st['ss'])
            trial_startz += trial_start
            trial_starts.extend(trial_startz + trial_start)
            sp.append(s + trial_start)
            neu.append(ne + last_neu)
            trial_start += max(trial_startz)
            last_neu += max(ne)

            # make combined events dataframe
            events_df_sess = convert_events_to_dataframe(
                dat,
                trial_startz,
                event_names=('gocue', 'response_time', 'feedback_time', 'trial_start'),
                condition_names=('response', 'contrast_right', 'contrast_left', 'feedback_type'))
            events_df_sess['i_session'] = i_sess

            events_df.append(events_df_sess)
        events_df = pd.concat(events_df)

        sp = np.hstack(sp)
        neu = np.hstack(neu)
        return sp, neu, events_df

    if not do_reload:
        saved_data = np.load(fpath + '/dat_concat.npz', allow_pickle=True)
        saved_orig_psth = np.load(fpath + '/dat_concat_orig_psth.npz', allow_pickle=True)
        brain_area = pd.read_pickle(fpath + '/dat_concat_orig_psth')
        spykes_times = pd.read_pickle(fpath + '/dat_concat_spykes_times.pickle')
        events_df = pd.read_pickle(fpath + '/dat_concat_events_df.pickle')
        return (saved_data['sp'], saved_data['neu'], spykes_times, events_df,
                saved_orig_psth['spks'],
                brain_area)
    alldat, dat_LFP, dat_ST = get_data()

    # Mean over trials and stack
    spks = [dat['spks'].mean(1) for dat in alldat]
    spks = np.vstack(spks)
    brain_area = [spykes_get_brain_regions(dat['brain_area'], i_session) for i_session, dat in enumerate(alldat)]
    brain_area = pd.concat(brain_area).reset_index()
    brain_area.to_pickle(fpath + '/dat_concat_orig_psth')

    np.savez(fpath + '/dat_concat_orig_psth.npz', spks=spks, allow_pickle=True)

    # Cluster/plot
    if do_cluster_alltrial:
        _ = cluster_trial_start(spks, brain_area)

    # Concat spiketimes from all seessions
    sp, neu, events_df = concat_spiketimes_sessions(dat_ST, alldat)

    # Continue making PSTHs
    spykes_times = spykes_get_times(sp, neu, thr_n_spikes=30)
    spykes_times = pd.concat([brain_area, spykes_times], axis=1, levels='i_neuron').dropna(0, 'any')

    # Dump to disk
    np.savez(fpath + '/dat_concat.npz', sp=sp, neu=neu, allow_pickle=True)
    spykes_times.to_pickle(fpath + '/dat_concat_spykes_times.pickle')
    events_df.to_pickle(fpath + '/dat_concat_events_df.pickle')
    return sp, neu, spykes_times, events_df, spks, brain_area


# Functions to work with spykes library

def get_psth(spikes, spykes_df, event, window=[-100, 100], bin_size=10, fr_thr=-1, conditions='response', **kwargs):
    """
    Calculates psth using spykes
    :param spikes:
    :param spykes_df:
    :param event:
    :param window:
    :param bin_size:
    :param fr_thr:
    :return:
    """
    import spykes
    import warnings
    warnings.filterwarnings('ignore')
    assert window[1] - window[0] > 0, 'Window size must be greater than zero!'
    # filter firing rate
    if type(spikes) is list:
        spikes = [spyke_obj for spyke_obj in spikes if spyke_obj.firingrate > fr_thr]
    else:
        spikes = [spyke_obj['spykes'] for i, spyke_obj in spikes.iterrows() if spyke_obj['spykes'].firingrate > fr_thr]
    pop = spykes.plot.popvis.PopVis(spikes)
    # calculate psth
    # TODO anything beyond trial_start+2.5 is nan
    mean_psth = pop.get_all_psth(event=event, df=spykes_df, window=window, binsize=bin_size, plot=False,
                                 conditions=conditions, **kwargs)
    # Check for sanity
    print(f"======= Event = {event}; Condition = {conditions} =======")
    assert mean_psth['data'][0].size > 0, 'Empty group PSTH!'
    return pop, mean_psth


def spyke2xar(all_psth, brain_group=None, brain_group_color=None):
    """
    Psth to xarray
    :param all_psth: output of pop.get_all_psth
    :return: xarray
    """

    arr = np.stack([all_psth['data'][key] for key in all_psth['data'].keys()
                    if len(all_psth['data'][key]) > 0])
    xar = xr.DataArray(arr,
                       dims=[all_psth['conditions'], 'Neuron', 'Time', ],
                       coords=[range(arr.shape[0]),
                               range(arr.shape[1]),
                               np.linspace(all_psth['window'][0],
                                           all_psth['window'][1], arr.shape[2]),
                               ],
                       name=all_psth['event'],
                       )
    if not (brain_group is None):
        xar = xar.assign_coords(Brain_area=('Neuron', brain_group))  # .sel(Brain_area='CA1')
        xar = xar.assign_coords(Brain_area_color=('Neuron', brain_group_color))

    return xar


### Functions to get our data to spykes library

# @title Functions to get our data to spykes library


def concat_spiketimes(sps):
    # print(sps['ss'].shape,sps['ss_passive'].shape)# Neuron X trial
    sp = []
    neu = []
    trial_starts = []
    trial_start = 0
    # Iterate over trials:
    for i_trial, spt in enumerate(sps.T):
        sp.append(np.hstack(spt) + trial_start)  # add i_trial*trial duration
        # Iterate over neurons
        neu.append(
            np.hstack([np.tile(i_neu, len(spnt)) for i_neu, spnt in enumerate(spt)])
        )
        trial_starts.append(trial_start)
        trial_start += i_trial * 2.5
    sp = np.hstack(sp)
    neu = np.hstack(neu)
    return sp, neu, np.array(trial_starts)


def convert_raster_to_spiketimes(dat, fs=100):
    """
    # Convert binned raster to spiketimes
    fs=100. # Sampling rate
    """

    # Make spiketime list
    trial_dur = dat['spks'].shape[2] / fs
    # Repeat t_trial n_trial times
    trial_starts = np.repeat(trial_dur, dat['spks'].shape[1])
    # cumsum
    trial_starts = np.cumsum(trial_starts)
    # starts with zero
    trial_starts -= trial_dur

    sp = [[np.where(r_trial)[0] / fs + trial_starts[i_trial]
           for i_trial, r_trial in enumerate(r_neuron)]
          for r_neuron in dat['spks']]  # list of spiketimes

    neu = np.hstack([np.repeat(i_neu, np.hstack(sp_n).shape[0]) for i_neu, sp_n in enumerate(sp)])
    sp = np.hstack([np.hstack(sp_n) for sp_n in sp])
    return sp, neu, trial_starts


def convert_events_to_dataframe(dat, trial_starts,
                                event_names=('gocue', 'response_time', 'feedback_time'),
                                condition_names=('response', 'contrast_right', 'contrast_left', 'feedback_type')):
    """
    Make events dataframe
    
    """
    event_names = np.array(event_names)
    events_df = []
    # Loop over events
    for event_name in event_names[event_names != ['trial_start']]:
        # Make events
        df = pd.DataFrame({'time_' + event_name: (dat[event_name].squeeze().copy() + trial_starts)})
        # Add conditions
        for condition_name in condition_names:
            df[condition_name] = dat[condition_name].squeeze()
        df['trial_start'] = trial_starts
        events_df.append(df)
    # concatentate
    events_df = pd.concat(events_df).reset_index()
    events_df = events_df.rename({'index': 'i_trial'}, axis=1)
    events_df = events_df.rename({'time_feedback_time': 'time_feedback', 'time_response_time': 'time_response'}, axis=1)
    # Prevent errors from negative quantities:
    events_df['response'] = events_df['response'] + 1
    events_df['feedback_type'] = events_df['feedback_type'] + 1
    return events_df


def hv_render_png(fig):
    import holoviews as hv
    # renderer = hv.renderer('bokeh').instance(mode='server')
    from IPython.display import display_png
    renderer = hv.renderer('bokeh')
    png, info = renderer(fig, fmt='png')
    display_png(png, raw=True, webdriver='/home/m/anaconda3/envs/dj/lib/python3.7/site-packages/bokeh/io/webdriver.py')
    # bokeh.io.webdriver


def spykes_get_brain_regions(brain_area, i_session=0):
    regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain", "basal ganglia", "cortical subplate", "other"]

    # col_list = ['blue','green','red','cyan','magenta','yellow','black']
    col_list = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    sorted_brain_areas = [['VISrl', 'VISl', 'VISpm', 'VISp', 'VISa', 'VISam'],
                          ['POL', 'MG', 'PT', 'LP', 'LGd', 'SPF', 'LD', 'PO', 'MD', 'VAL', 'VPL', 'VPM', 'TH', 'RT',
                           'CL', 'LH'],
                          ['CA', 'CA2', 'CA3', 'POST', 'CA1', 'SUB', 'DG'],
                          ['DP', 'COA', 'ILA', 'ORBm', 'RSP', 'OLF', 'PL', 'ACA', 'MOs', 'MOp', 'SSs', 'SSp', 'ORB',
                           'PIR', 'AUD', 'TT'],
                          ['SCsg', 'NB', 'APN', 'PAG', 'SCs', 'SCig', 'MRN', 'IC', 'SCm', 'MB', 'RN', 'ZI'],
                          ['LSc', 'LS', 'LSr', 'MS', 'ACB', 'GPe', 'OT', 'CP', 'SI', 'SNr'],
                          ['BMA', 'EP', 'EPd', 'MEA', 'BLA', 'root']]

    all_brain_areas = []
    all_area = []
    all_colors = []
    for sublist in range(len(sorted_brain_areas)):
        for item in range(len(sorted_brain_areas[sublist])):
            all_brain_areas.append(sorted_brain_areas[sublist][item])
            all_area.append(regions[sublist])
            all_colors.append(col_list[sublist])

    all_area = np.array(all_area)
    all_brain_areas = np.array(all_brain_areas)
    all_colors = np.array(all_colors)
    brain_regions = pd.DataFrame({'brain_area': brain_area,
                                  'i_neuron': range(len(brain_area)),
                                  'i_session': i_session})
    for brain_area in brain_regions['brain_area'].unique():
        brain_regions.loc[brain_regions['brain_area'] == brain_area,
                          'brain_group'] = all_area[all_brain_areas == brain_area][0]

        brain_regions.loc[brain_regions['brain_area'] == brain_area,
                          'brain_group_color'] = all_colors[all_brain_areas == brain_area][0]

    return brain_regions.set_index('i_neuron')


def spykes_add_brain_regions(spykes_df, brain_area):
    brain_regions = spykes_get_brain_regions(brain_area)

    spykes_df = pd.concat([brain_regions, spykes_df], axis=1, levels='i_neuron').dropna(0, 'any')
    print(f'This dataset has {np.unique(brain_area)}, meaning {spykes_df["brain_group"].unique()}')
    return spykes_df


def spykes_get_times(s_ts, s_id, thr_n_spikes=25):
    """
    Use spykes library
    NB: Don't laugh, I wrote this a long time ago!
    :param s_ts:
    :param s_id:
    :param debug:
    :return:
    """

    from spykes.plot import neurovis
    s_id = s_id.astype('int')
    spykes_list = [neurovis.NeuroVis(s_ts[s_id == iu], name=str(iu)) for iu in np.unique(s_id) if
                   len(s_ts[s_id == iu]) > thr_n_spikes]
    spykes_df = pd.DataFrame({'spykes': spykes_list, 'i_neuron': [int(neuron.name) for neuron in spykes_list]})
    return spykes_df.set_index(['i_neuron'])


# use sns to Plot by condition
import seaborn as sns


def heatmap(df, **kwargs):
    """
    # This needs a pivoted dataframe already
    df.index[g.dendrogram_row.reordered_ind]
    Drawbacks:
    1. sns.heatmap has no faceting
    """
    import matplotlib.colors as colors
    from scipy.stats import zscore

    sns.heatmap(zscore(df, 0), robust=True,
                center=True,
                # norm=colors.SymLogNorm(0.03),
                **kwargs);


def plot_psth_add_event_lines(events_df, binsize=10, ax=None):
    """
    binsize=psth['binsize']
    :param events_df:
    :param psth:
    :return:
    """
    # Make vert lines
    for event_name, col in zip(['time_gocue', 'time_response'], ['w','y']):
        #events_df.columns[events_df.columns.str.contains('time')]:
        event_times_psth_index = (events_df[event_name] - events_df['trial_start']) * 1000 / binsize
        if ax is None:
            ax = plt.gca()
        ax.axvline(event_times_psth_index.mean() - 20, color=col, linewidth=1) # +/- 20 bins
        ax.axvline(event_times_psth_index.mean() + 20, color=col, linewidth=1) # +/- 20 bins
        # ax.axvline(event_times_psth_index.mean() - event_times_psth_index.std(), color='w', linewidth=.8)
        # ax.axvline(event_times_psth_index.mean() + event_times_psth_index.std(), color='y', linewidth=.8)


def cluster_trial_start(spks, brain_regions, events_df=None, binsize=10, row_colors_string='brain_group_color'):
    import seaborn as sns
    import matplotlib.pylab as plt
    import matplotlib.ticker as ticker

    if spks.ndim > 2:  # needs averaging
        spks = spks.mean(1)
    active_neurons = (spks.sum(1) > 6).nonzero()[0]
    spks = spks[active_neurons]
    brain_regions = brain_regions.loc[active_neurons]
    # Cluster
    g = sns.clustermap(spks,
                       method='weighted',
                       z_score=0,  # Row zscore
                       figsize=[7, 15],
                       col_cluster=False,
                       # standard_scale=0,
                       metric="cosine",
                       center=True,
                       row_colors=brain_regions.reset_index()[row_colors_string].values,
                       xticklabels=20,
                       cbar_kws={'label':'Firing rate (z-score)'},
                       robust=True)

    # Remove colorbar
    # g.cax.set_visible(False)
    # x axis to seconds
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x/100,2)))
    g.ax_heatmap.xaxis.set_major_formatter(ticks)
    g.ax_heatmap.set_xlabel('Time (s)')
    # if not (events_df is None):
    #     plot_psth_add_event_lines(events_df, binsize=binsize, ax=g.ax_heatmap)
    ax=g.ax_heatmap
    ax.axvline(30, color='w', linewidth=1)  # +/- 20 bins
    ax.axvline(70, color='w', linewidth=1)  # +/- 20 bins

    ax.axvline(110, color='y', linewidth=1)  # +/- 20 bins
    ax.axvline(150, color='y', linewidth=1)  # +/- 20 bins

    # Legend
    plt.figure()
    brain_groups = []
    for i, (col, xar) in enumerate(brain_regions.reset_index().groupby('brain_group_color')):
        plt.plot(i, 0, col + 'o', markersize=10)
        brain_groups.append(xar['brain_group'].iloc[0])
    _ = plt.legend(brain_groups)

    # plt.savefig('/home/m/legend.svg')
    plt.show()

    return g.dendrogram_row.reordered_ind


def cluster_trial_start_by_condition(df, condition_name):
    """

    :param df: df.loc[g.dendrogram_row.reordered_ind]
    :param condition_name:
    :return:
    """

    # Plot conditions separately in subplots
    plt.figure(figsize=[10, 10])
    all_conditions = df.reset_index()[condition_name].unique()
    for i_condition, condition in enumerate(all_conditions):
        plt.subplot(1, len(all_conditions), i_condition + 1)
        heatmap(df.xs(condition, level=condition_name), cbar=False)
        plt.title(f"{condition_name} = {condition}")
    plt.show()


def df_zscore(df, y='Power near', axis=('Subject',)):
    """
    standardize (default)
    zscore transform column of dataframe 'y' by "axis"
    :param df:
    :param y: string denoting column to transfor
    :param axis: can be list
    :return:
    """

    # def standardize(x):
    #     return (x - x.min()) / (x.max() - x.min())

    # zscore
    from scipy.stats import zscore
    df[y] = df.groupby(axis)[y].transform(lambda x: zscore(x, 0))
    # df[y] = df.groupby(axis)[y].transform(lambda x: standardize(x))
    return df


def cluster(xar, plotose=True, row_colors_string='Brain_area_color', thr_n_spikes=15):
    value_name = xar.name
    condition_name = xar.dims[0]
    df = xar.copy(deep=True).to_dataframe().reset_index().set_index('Neuron')
    df = df[df.groupby('Neuron')[value_name].sum() > thr_n_spikes].reset_index()

    # zscore
    df = df_zscore(df, value_name, 'Neuron')

    df = df.pivot(index=['Neuron', condition_name, 'Brain_area_color', 'Brain_area'],
                  columns='Time',
                  values=value_name)

    # Cluster conditions together, plot them (also together)
    try:
        g = sns.clustermap(df,
                           method='weighted',
                           # z_score=0,
                           figsize=[5, 10],
                           col_cluster=False,
                           # standard_scale=0,
                           metric="cosine",
                           center=True,
                           row_colors=df.reset_index()[row_colors_string].values,
                           robust=True)

        # Legend

        import matplotlib.pylab as plt
        plt.figure()
        brain_groups = []
        for i, (col, xar) in enumerate(df.reset_index().groupby('Brain_area_color')):
            plt.plot(i, 0, col + 'o', markersize=10)
            brain_groups.append(xar['Brain_area'].iloc[0])
        _ = plt.legend(brain_groups)
        plt.show()
    except Exception as e:
        print(e)
        # import pdb;pdb.set_trace()
        return None, None, None

    # reorder neurons according to Clustering
    index = df.index[g.dendrogram_row.reordered_ind]
    df = df.loc[index]

    plt.show()
    if plotose:
        # Plot conditions separately in subplots
        plt.figure(figsize=[10, 10])
        all_conditions = df.reset_index()[condition_name].unique()
        for i_condition, condition in enumerate(all_conditions):
            plt.subplot(1, len(all_conditions), i_condition + 1)
            heatmap(df.loc[index].xs(condition, level=condition_name), cbar=False)
            plt.title(f"{condition_name} = {condition}")
        plt.suptitle(f"Centered at {value_name}")
        plt.show()

    # Back to normal dataframe
    df_tidy = df.copy()
    df = df.reset_index().melt(id_vars=['Neuron', condition_name, 'Brain_area_color', 'Brain_area'],
                               value_name=value_name)
    # df=df.set_index(list(set(df.columns)-{value_name}))

    # Categorical neuron prevents rearrangement
    df['Neuron'] = df['Neuron'].astype('str')
    df['Time'] = df['Time'].astype('str')

    return df, df_tidy, index
