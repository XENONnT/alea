import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import poisson, norm
from scipy.interpolate import interp1d
import multihist as mh
from multihist import Histdd

from copy import deepcopy


def interleave_bincv(binc, bine):
    rc = np.zeros(2 * len(binc))
    re = np.zeros(2 * len(binc))
    rc[0::2] = binc
    rc[1::2] = binc
    re[0::2] = bine[0:-1]
    re[1::2] = bine[1::]
    return rc, re


class pdf_plotter():
    def __init__(self, ll, analysis_space_plot=None):
        """Generate histograms for all sources

        Args:
            ll (ll object): binference ll object
            analysis_space_plot (list of tuples, optional): if not None
                this replaces the bin edges defined in the ll config file
                for plotting / defining the histogras.
                Example: analysis_space_plot =
                    [("cs1", np.linspace(0, 100, 101)),
                     ("logcs2", np.linspace(1.7, 3.9, 125 + 1))]
                Defaults to None.
        """
        self.ll = deepcopy(ll)
        bins = []  # bin edges
        bincs = []  # bin centers of each component
        direction_names = []
        dtype = []
        data_length = 1  # number of bins in nD histogram
        self.data_lengths = []  # list of number of bins of each component

        if analysis_space_plot is not None:
            analysis_space = analysis_space_plot
        else:
            analysis_space = self.ll.base_model.config['analysis_space']
        for direction in analysis_space:
            bins.append(direction[1])
            binc = 0.5 * (direction[1][1::] + direction[1][0:-1])
            bincs.append(binc)
            #  binc_dict[direction[0]] = binc
            dtype.append((direction[0], float))
            data_length *= len(direction[1]) - 1
            self.data_lengths.append(len(direction[1]) - 1)
            direction_names.append(direction[0])
        dtype.append(('source', int))

        data_binc = np.zeros(data_length, dtype=dtype)
        for i, l in enumerate(product(*bincs)):
            for n, v in zip(direction_names, l):
                data_binc[n][i] = v

        self.ll.set_data(data_binc)

        self.source_histograms = []
        for i in range(len(self.ll.base_model.sources)):
            self.source_histograms.append(Histdd(bins=bins))

    def get_pdf(self,
                i_source=None,
                expected_events=False,
                slice_args={},
                **kwargs):
        """get a histogram of the PDF or of the expected events for the
        entire analysis space or projected onto one axis.

        Args:
            i_source (int, optional): index of a source, the histogram
                of which will be returned. If None is passed, a histogram
                with all sources is returned. Defaults to None.
            expected_events (bool, optional): If True, histogram of expected
                events is returned, otherwise a histogram of the PDF.
                Defaults to False.
            slice_args (dict, optional): Slicing arguments for plotting.
                Defaults to {}.
            kwargs are rate_multipliers and shape_parameter_settings
                that are passed to the call of the full ll output, as
                well as n_mc, which is used as a rate factor if passed.

        Raises:
            ValueError: if slice_axis is given in slice_args but not
                collapse_slices.

        Returns:
            multhist.Hisdd: histogram of PDF / expected events
        """
        n_mc = kwargs.get('n_mc', None)
        if n_mc is not None:
            del kwargs['n_mc']
        ret = self.ll(full_output=True, **kwargs)  # result, mus, ps
        if type(ret) == float:
            print("ERROR, generator kwarg outside range?")
            print(kwargs)
        #print("ret,kwargs",ret,kwargs)
        _, mus, ps_array = ret
        for i in range(len(self.ll.base_model.sources)):
            self.source_histograms[i].histogram = ps_array[i].reshape(
                self.data_lengths)
            #source_histograms[i]*=source_histograms[i].bin_volumes()
        self.mus = mus
        self.last_kwargs = kwargs

        if i_source is not None:
            ret_histogram = deepcopy(self.source_histograms[i_source])
            ret_histogram.histogram *= ret_histogram.bin_volumes()
            if expected_events:
                if n_mc is not None:
                    rate_factor = n_mc
                    print(
                        'Multiply histogram with rate factor(n_mc): {}'.format(
                            rate_factor))
                else:
                    rate_factor = mus[i_source]
                    print('Multiply histogram with rate factor(mu): {}'.format(
                        rate_factor))
                ret_histogram *= rate_factor

        else:
            ret_histogram = self.source_histograms[0].similar_blank_hist()
            for i in range(len(mus)):
                ret_histogram.histogram += mus[i] * self.source_histograms[
                    i].histogram / np.sum(mus)
            ret_histogram.histogram *= ret_histogram.bin_volumes()
            if expected_events:
                if n_mc is not None:
                    rate_factor = n_mc
                    print('Multiply histogram with rate factor (n_mc): {}'.
                          format(rate_factor))
                else:
                    rate_factor = np.sum(mus)
                    print('Multiply histogram with rate factor (sum(mus)): {}'.
                          format(rate_factor))

                ret_histogram *= rate_factor
        ret_histogram.axis_names = [
            n for n, _ in self.ll.base_model.config['analysis_space']
        ]

        if type(slice_args) is dict:
            slice_args = [slice_args]

        for sa in slice_args:
            # Supply slicing args for plotting:
            slice_axis = sa.get("slice_axis", None)
            sum_axis = sa.get(
                "sum_axis", False
            )  #decide if you wish to sum the histogram into lower dimensions or

            collapse_axis = sa.get('collapse_axis', None)
            collapse_slices = sa.get('collapse_slices', None)

            if (slice_axis is not None):
                #use total, not differential probability for slicing (makes sums easier):

                bes = ret_histogram.bin_edges[slice_axis]
                slice_axis_limits = sa.get("slice_axis_limits",
                                           [bes[0], bes[-1]])
                print("slicing axis", slice_axis)
                if sum_axis:
                    ret_histogram = ret_histogram.slicesum(
                        axis=slice_axis,
                        start=slice_axis_limits[0],
                        stop=slice_axis_limits[1])
                else:
                    ret_histogram = ret_histogram.slice(
                        axis=slice_axis,
                        start=slice_axis_limits[0],
                        stop=slice_axis_limits[1])

                if collapse_axis is not None:
                    if collapse_slices is None:
                        raise ValueError(
                            "To collapse you must supply collapse_slices")
                    ret_histogram = ret_histogram.collapse_bins(
                        collapse_slices, axis=collapse_axis)

        if not expected_events:
            ret_histogram.histogram /= ret_histogram.bin_volumes()

        return ret_histogram

    def plot_pdf(self,
                 i_source=None,
                 expected_events=False,
                 slice_args={},
                 plot_kwargs={},
                 data=None,
                 scatter_kwargs={},
                 **kwargs):
        h = self.get_pdf(i_source=i_source,
                         expected_events=expected_events,
                         slice_args=slice_args,
                         **kwargs)
        h.plot(**plot_kwargs)

        if data is not None:
            if 'source' in data.dtype.names:
                for i, source in enumerate(self.ll.base_model.sources):
                    scatter_kwargs_default = {'lw': 0.5, 'edgecolor': 'gray'}
                    scatter_kwargs_default['color'] = source.config.get(
                        'color', 'gray')
                    scatter_kwargs_default['label'] = source.config.get(
                        'label', '')
                    scatter_kwargs_default.update(scatter_kwargs)
                    axis_names = h.axis_names
                    data_source = data[data['source'] == i]
                    plt.scatter(data_source[axis_names[0]],
                                data_source[axis_names[1]],
                                **scatter_kwargs_default)
            else:
                scatter_kwargs_default = {'lw': 0.5, 'edgecolor': 'gray'}
                scatter_kwargs_default.update(scatter_kwargs)
                axis_names = h.axis_names
                plt.scatter(data[axis_names[0]], data[axis_names[1]],
                            **scatter_kwargs_default)

    def get_expectation_value(self, i_source=None, slice_args={}, **kwargs):
        if i_source is not None:
            return self.get_pdf(i_source=i_source,
                                expected_events=True,
                                slice_args=slice_args,
                                **kwargs).n
        mus = []
        for i, _ in enumerate(self.source_histograms):
            mus.append(
                self.get_pdf(i_source=i,
                             expected_events=True,
                             slice_args=slice_args,
                             **kwargs).n)
        return np.array(mus)

    def get_projected_plottable(self,
                                proj_axis=0,
                                i_source=0,
                                per_axis=False,
                                binning=None,
                                slice_args={},
                                cl=0.68,
                                positive_radius=True,
                                **kwargs):
        h = deepcopy(
            self.get_pdf(i_source=i_source,
                         slice_args=slice_args,
                         expected_events=True,
                         **kwargs))
        hh = h.project(axis=proj_axis)
        if positive_radius:
            hh.histogram[1] += hh.histogram[0]
            hh.histogram[0] = 0.
        bes = hh.bin_edges
        ns = np.concatenate([np.zeros(1), hh.cumulative_histogram])
        if binning is None:
            bins = hh.bin_edges
        elif type(binning) is np.ndarray:
            bins = binning
        elif np.isreal(binning):
            cumulative_interpolation = interp1d(ns, bes)
            nbins = np.ceil(ns[-1] / binning)
            bins = cumulative_interpolation(np.linspace(0, ns[-1], nbins))

        ns_interpolator = interp1d(bes,
                                   ns,
                                   bounds_error=False,
                                   fill_value=(0, ns[-1]))

        n_ret = ns_interpolator(bins)
        n_ret = n_ret[1::] - n_ret[0:-1]
        h_ret = mh.Hist1d(bins=bins)
        h_ret.histogram = n_ret
        if per_axis:
            h_ret /= h_ret.bin_volumes()
        if cl is not None:
            hd, hu = poisson(n_ret).interval(cl)
            hu = np.maximum(n_ret, hu)
        else:
            hd = hu = np.zeros(len(n_ret))
        if per_axis:
            hd = hd / h_ret.bin_volumes()
            hu = hu / h_ret.bin_volumes()
        hd, be = interleave_bincv(hd, bins)
        hu, be = interleave_bincv(hu, bins)
        plot_kwargs = {'lw': 3, 'color': 'black'}
        if i_source is not None:
            plot_kwargs['color'] = self.ll.base_model.sources[
                i_source].config.get("color", 'gray')
            plot_kwargs['label'] = self.ll.base_model.sources[
                i_source].config.get("label", 'label')
            plot_kwargs['lw'] = 4

        return h_ret, be, hd, hu, plot_kwargs

    def plot_projection(
        self,
        proj_axis=2,
        binning=None,
        per_axis=False,
        slice_args={},
        res={},
        data=None,
        cl=0.68,
        plot_kwarg_dict={
            'wimp': {
                'linestyle': '--'
            },
            'total': {
                'label': 'Total Model'
            }
        },
        ylim=None,
        xlim=None,
        scatter_kwargs={},
        plottables=None,
        logy=True,
        reference_period=None,
        positive_radius=False,
        plot_legend=True,
        plot_order=['cnns', 'radiogenic', 'ac', 'wall', 'er', 'total', 'wimp'],
    ):
        plottables, legend = plot_projection(
            pdf_plotter=self,
            proj_axis=proj_axis,
            binning=binning,
            per_axis=per_axis,
            slice_args=slice_args,
            res=res,
            data=data,
            cl=cl,
            plot_kwarg_dict=plot_kwarg_dict,
            ylim=ylim,
            xlim=xlim,
            scatter_kwargs=scatter_kwargs,
            plottables=plottables,
            logy=logy,
            reference_period=reference_period,
            plot_order=plot_order,
            plot_legend=plot_legend,
            positive_radius=positive_radius,
        )
        return plottables, legend

    def plot_projection_residual(
        self,
        proj_axis=2,
        binning=None,
        per_axis=False,
        slice_args={},
        res={},
        data=None,
        cl=0.68,
        plot_kwarg_dict={
            'wimp': {
                'linestyle': '--'
            },
            'total': {
                'label': 'Total Model'
            }
        },
        ylim=None,
        xlim=None,
        scatter_kwargs={},
        plottables=None,
        logy=True,
        reference_period=None,
        plot_legend=True,
        positive_radius=False,
        plot_order=['cnns', 'radiogenic', 'ac', 'wall', 'er', 'total', 'wimp'],
    ):
        plottables, legend, ps = plot_projection_residual(
            pdf_plotter=self,
            proj_axis=proj_axis,
            binning=binning,
            per_axis=per_axis,
            slice_args=slice_args,
            res=res,
            data=data,
            cl=cl,
            plot_kwarg_dict=plot_kwarg_dict,
            ylim=ylim,
            xlim=xlim,
            scatter_kwargs=scatter_kwargs,
            plottables=plottables,
            logy=logy,
            reference_period=reference_period,
            plot_order=plot_order,
            plot_legend=plot_legend,
            positive_radius=positive_radius,
        )
        return plottables, legend, ps

    def plot_projected_difference(
        self,
        proj_axis=0,
        i_source=0,
        binning=None,
        fractional=False,
        absolute=False,
        slice_args={},
        per_axis=False,
        res={},
        resmod={},
        plot_kwargs={},
    ):
        plot_projected_difference(
            self,
            proj_axis=proj_axis,
            i_source=i_source,
            binning=binning,
            fractional=fractional,
            absolute=absolute,
            slice_args=slice_args,
            per_axis=per_axis,
            res=res,
            resmod=resmod,
            plot_kwargs=plot_kwargs,
        )


def plot_projected_difference(plotter,
                              proj_axis=0,
                              i_source=0,
                              binning=None,
                              fractional=False,
                              absolute=False,
                              slice_args={},
                              per_axis=False,
                              res={},
                              resmod={},
                              plot_kwargs={}):

    h, _, _, _, default_plot_kwargs = plotter.get_projected_plottable(
        proj_axis=proj_axis,
        i_source=i_source,
        per_axis=per_axis,
        binning=binning,
        slice_args=slice_args,
        **res)
    print(h, len(h))
    res0 = deepcopy(res)
    res0.update(resmod)
    h0, _, _, _, _ = plotter.get_projected_plottable(proj_axis=proj_axis,
                                                     i_source=i_source,
                                                     per_axis=per_axis,
                                                     binning=binning,
                                                     slice_args=slice_args,
                                                     **res0)
    if not fractional:
        h = h - h0
    else:
        h = (h - h0) / h
    if absolute:
        h.histogram = abs(h.histogram)
    default_plot_kwargs.update(plot_kwargs)
    h.plot(**default_plot_kwargs)


def plot_projection(
    pdf_plotter=None,
    proj_axis=2,
    binning=None,
    per_axis=False,
    slice_args={},
    res={},
    data=None,
    cl=0.68,
    plot_kwarg_dict={
        'wimp': {
            'linestyle': '--'
        },
        'total': {
            'label': 'Total Model'
        }
    },
    ylim=None,
    xlim=None,
    scatter_kwargs={},
    plottables=None,
    logy=True,
    reference_period=None,
    plot_legend=True,
    positive_radius=True,
    plot_order=['cnns', 'radiogenic', 'ac', 'wall', 'er', 'total', 'wimp'],
):
    #  from sr_ll_definition import human_name_dict
    if plottables is None:
        plottables = {}
        plottables['total'] = pdf_plotter.get_projected_plottable(
            proj_axis=proj_axis,
            i_source=None,
            per_axis=per_axis,
            binning=binning,
            slice_args=slice_args,
            positive_radius=positive_radius,
            **res)
        bins = plottables['total'][0].bin_edges
        for i, sn in enumerate(pdf_plotter.ll.source_name_list):
            plottables[sn] = pdf_plotter.get_projected_plottable(
                proj_axis=proj_axis,
                i_source=i,
                per_axis=per_axis,
                binning=bins,
                slice_args=slice_args,
                positive_radius=positive_radius,
                **res)
    else:
        bins = plottables['total'][0].bin_edges
    for k in plot_order:
        (h_res, be, ed, eu, kwarg) = plottables[k]
        if reference_period is not None:
            h_res = h_res / reference_period
        kwarg.update(plot_kwarg_dict.get(k, {}))
        h_res.plot(**kwarg)
    h_res, be, ed, eu, kwarg = plottables['total']
    if reference_period is not None:
        ed = ed / reference_period
        eu = eu / reference_period
    if cl is not None:
        plt.fill_between(be, ed, eu, alpha=0.3, color=kwarg['color'])
    bins = h_res.bin_edges
    if data is not None:
        binv, _ = np.histogram(data, bins=bins)
        if positive_radius:
            binv, _ = np.histogram(np.abs(data), bins=bins)
        binc = 0.5 * (bins[0:-1] + bins[1::])
        if per_axis:
            binv = binv / h_res.bin_volumes()
        if reference_period is not None:
            binv = binv / reference_period
        scatter_kwargs_default = {'color': 'magenta', 'marker': 'o', 's': 25}
        scatter_kwargs_default.update(scatter_kwargs)
        plt.scatter(binc, binv, zorder=100, **scatter_kwargs_default)
    if logy:
        plt.yscale('log')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if pdf_plotter is not None:
        axis_name = pdf_plotter.ll.base_model.config['analysis_space'][
            proj_axis][0]
    else:
        anames = {0: 'cs1', 1: 'cs2_bottom', 2: 'r'}
        axis_name = anames[proj_axis]
    #  plt.xlabel(human_name_dict.get(axis_name, axis_name))
    if axis_name == 'r':
        xlim = plt.gca().get_xlim()
        rticks = [10, 20, 30, 35, 40, 42, 44]
        if bins[0] < 0:
            rticks = [
                np.sqrt(abs(bins[0])), 0, 20, 30, 35, 40,
                np.sqrt(abs(bins[-1]))
            ]
        r2ticks = [r * abs(r) for r in rticks]
        rtickns = ["{:.1f}".format(r) for r in rticks]
        plt.gca().set_xticks(r2ticks)
        plt.gca().set_xticklabels(rtickns)
        plt.xlim(xlim)
        plt.xlabel("$r$[$\mathrm{cm}$]")
    period_add = ""
    if reference_period is not None:
        period_add = "/$d$"
    if per_axis:
        pass
        #  plt.ylabel(
        #      'events/{:s}'.format(human_name_dict.get(axis_name, axis_name)) +
        #      period_add)
    else:
        plt.ylabel('events' + period_add)

    # order legend reverse from how we plotted it:
    objs, handles = plt.gca().get_legend_handles_labels()
    objs = objs[::-1]
    handles = handles[::-1]

    if plot_legend:
        legend = plt.legend(objs,
                            handles,
                            loc=2,
                            bbox_to_anchor=(1, 1),
                            borderaxespad=0)
    else:
        legend = None

    return plottables, legend


def plot_projection_residual(
    pdf_plotter=None,
    proj_axis=2,
    binning=None,
    per_axis=False,
    slice_args={},
    res={},
    data=None,
    cl=0.68,
    plot_kwarg_dict={
        'wimp': {
            'linestyle': '--'
        },
        'total': {
            'label': 'Total Model'
        }
    },
    ylim=None,
    xlim=None,
    scatter_kwargs={},
    plottables=None,
    logy=True,
    reference_period=None,
    plot_legend=True,
    positive_radius=False,
    plot_order=['cnns', 'radiogenic', 'ac', 'wall', 'er', 'total', 'wimp'],
):
    fig1 = plt.figure()
    frame1 = fig1.add_axes((.1, .3, .8, .6))
    plottables, legend = plot_projection(
        pdf_plotter=pdf_plotter,
        proj_axis=proj_axis,
        binning=binning,
        per_axis=per_axis,
        slice_args=slice_args,
        res=res,
        data=data,
        cl=cl,
        plot_kwarg_dict=plot_kwarg_dict,
        ylim=ylim,
        xlim=xlim,
        scatter_kwargs=scatter_kwargs,
        plottables=plottables,
        logy=logy,
        reference_period=reference_period,
        plot_order=plot_order,
        plot_legend=plot_legend,
        positive_radius=positive_radius,
    )
    xlabel = plt.gca().get_xlabel()
    xlim = plt.gca().get_xlim()
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    frame2 = fig1.add_axes((.1, .1, .8, .2))
    plt.ylabel('p-value')
    plt.xlabel('')
    hist_res, be, ed, eu, kwarg = plottables['total']

    mus = hist_res.histogram
    if per_axis:
        mus = mus * hist_res.bin_volumes()
    bins = hist_res.bin_edges
    ns, _ = np.histogram(data, bins=bins)
    if positive_radius:
        ns, _ = np.histogram(np.abs(data), bins=bins)

    ps = poisson(mus).sf(ns - 0.1)
    ps = -1 * norm().ppf(ps)
    binc = 0.5 * (bins[0:-1] + bins[1::])
    plt.scatter(binc, ps, color='k', s=scatter_kwargs.get("s", 20))
    # plt.xlabel('radius')
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    if proj_axis == 2:
        rticks = [10, 20, 30, 35, 40, 42, 44]
        r2ticks = [r * abs(r) for r in rticks]
        if bins[0] < 0:
            print("switch to twoside binning")
            rticks = [
                np.sign(bins[0]) * np.sqrt(abs(bins[0])), 0, 20, 30, 35, 40
            ]
        r2ticks = [r * abs(r) for r in rticks]
        rtickns = ["{:.1f}".format(r) for r in rticks]
        plt.gca().set_xticks(r2ticks)
        plt.gca().set_xticklabels(rtickns)
        plt.xlim(xlim)
        plt.xlabel("$r$[$\mathrm{cm}$]")
    # plt.yticks([0,0.5,1])
    # plt.ylim([-0.3,1.3])
    plt.yticks([-2, 0, 2])
    plt.ylim([-3, 3])
    plt.ylabel('$\sigma$')
    plt.xlabel(xlabel)
    return plottables, legend, ps
