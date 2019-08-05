"""
hardcoded plot for triple combination figure
"""


import csv
from copy import deepcopy

import click
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
from scipy.integrate import simps
from scipy.optimize import minimize
import toml


# PALETTE_3 = [
#     (109/255, 181/255, 255/255),
#     (177/255, 88/255, 0/255),
#     (6/255, 70/255, 81/255)
# ]
PALETTE_3 = [
    (0/255, 0/255, 0/255),
    (200/255, 180/255, 55/255),
    (0/255, 114/255, 178/255),
]


# PALETTE_4 = [
#     (0/255, 12/255, 12/255),
#     (66/255, 66/255, 0/255),
#     (109/255, 181/255, 255/255),
#     (200/255, 100/255, 0/255),
# ]
PALETTE_4 = [
    (0/255, 0/255, 0/255),
    (200/255, 180/255, 55/255),
    (204/255, 121/255, 167/255),
    (0/255, 114/255, 178/255),
]




@click.group()
def main():
    pass




def get_conc(drug, time):
    """
    time axis for datapoints
    drug pharmacokinetics info
    """
    t_reduced = time % (drug['tau'])
    conc = drug['S']*drug['F']*drug['D']*drug['ka'] / (drug['Vd']*(drug['ka'] - drug['ke'])) * \
           (np.exp(-drug['ke']*t_reduced)/(1 - np.exp(-drug['ke']*drug['tau'])) - \
            np.exp(-drug['ka']*t_reduced)/(1 - np.exp(-drug['ka']*drug['tau'])))
    return np.roll(conc, int(drug['offset']/(drug['tau']*drug['repeats'])*len(time)))


def get_effect(drug1, drug2, method, dose1=None, dose2=None, ic1=1, ic2=1, m=1, resolution=1000):
    """
    Get the area under curve of the growth rate curve defined by a certain
    drugs pharmacokinetic profile
    """
    assert drug1['tau']*drug1['repeats'] == drug2['tau']*drug2['repeats']
    if dose1:
        drug1['D'] = dose1
    if dose2:
        drug2['D'] = dose2
    time = np.linspace(0, drug1['tau']*drug1['repeats'], resolution)
    conc1 = get_conc(drug1, time)
    conc2 = get_conc(drug2, time)
    if method == 'e':
        dve = 1/(1 + (conc1/ic1 + conc2/ic2)**m)
    elif method == 'n':
        dve = 1/(1 + (conc1/ic1 + conc2/ic2 + conc2*conc1/(ic1*ic2))**m)
    else:
        print("invalid method in get_effect")
        exit(0)
    return simps(dve, time)


def get_effect_single(drug, dose=None, ic=1, resolution=1000, m=1):
    """
    Get the area under curve of the growth rate curve defined by a certain
    drugs pharmacokinetic profile.
    Single drug has no interaction parameters
    """
    time = np.linspace(0, drug['tau']*drug['repeats'], resolution)
    if dose:
        drug['D'] = dose
    conc = get_conc(drug, time)
    dve = 1/(1 + (conc/ic)**m)
    return simps(dve, time)


def normalize_effect_single(drug, effect_target, m=1, xtol=1e-9):
    """
    Set the drug dose such that the effect calculated by get_effect_single
    equals effect_target
    In principle, two drugs normalized like this will be equally effective
    """
    res = minimize(lambda x: abs(effect_target - get_effect_single(drug, x[0], m=m)),
                   0.1, method='nelder-mead', options={'xtol': xtol, 'disp': False})
    drug['D'] = res.x[0]


def normalize_effect(drug1, drug2, method, effect_target, ratio, ic1=1, ic2=1, m=1, xtol=1e-9):
    """
    Set the drug dose such that the effect calculated by get_effect
    equals effect_target with a given drug ratio
    """
    res = minimize(lambda x: abs(effect_target - get_effect(drug1, drug2, method,
                                                            x[0]*ratio, x[0]*(1 - ratio), m=m,
                                                            ic1=ic1, ic2=ic2)),
                   0.1, method='nelder-mead', options={'xtol': xtol, 'disp': False})
    drug1['D'] = res.x[0]*ratio
    drug2['D'] = res.x[0]*(1 - ratio)


@main.command()
@click.option('--save', type=click.Path(), default=None)
@click.option('-t', '--optimization-target', type=str)
def as_combos(save, optimization_target):
    """
    click construct
    """

    assert (optimization_target in ['xauc', 'min'])

    # opt_targe = 'xauc'
    # opt_targe = 'min'
    opt_targe = optimization_target

    if save is not None:
        pdf_out = PdfPages(save)


    drug_bo = toml.load('intermediate/bosutinib.toml')
    drug_as = toml.load('intermediate/asciminib.toml')
    drug_ax = toml.load('intermediate/axitinib.toml')

    print(drug_bo['name'], drug_bo['D'])
    print(drug_as['name'], drug_as['D'])
    print(drug_ax['name'], drug_ax['D'])

    print(drug_bo['name'], get_effect_single(drug_bo))
    print(drug_as['name'], get_effect_single(drug_as))
    print(drug_ax['name'], get_effect_single(drug_ax))

    print(drug_bo['T315I'])
    print(drug_as['T315I'])
    print(drug_ax['T315I'])

    normalize_effect(drug_bo, drug_as, 'n', 0.1, 0.5)
    normalize_effect_single(drug_ax, 0.95)

    print(drug_bo['name'], get_effect_single(drug_bo))
    print(drug_as['name'], get_effect_single(drug_as))
    for m in [0.5, 1, 2]:
        normalize_effect(drug_bo, drug_as, 'n', 0.1, 0.5, m=m)
        print('m = ', m, ':')
        print('\t', drug_bo['name'], drug_bo['D'])
        print('\t', drug_as['name'], drug_as['D'])
        print('\t', get_effect(drug_bo, drug_as, 'n', m=m))
    print(drug_ax['name'], get_effect_single(drug_ax))

    print(drug_bo['name'], drug_bo['D'])
    print(drug_as['name'], drug_as['D'])
    print(drug_ax['name'], drug_ax['D'])

    drug_bo['offset'] = 0

    offsetspace = np.linspace(0, 1, 13)[:-1]
    # offsetspace = [0]
    time = np.linspace(0, 1, 1000)

    best_offsets = [{}, {}, {}]#{'as': 0.0, 'ax': 0.0}
    best_effect = [999, 999, 999]


    asbo_traces = []
    asboax_traces = []


    for as_offset in offsetspace:
        drug_as['offset'] = as_offset
        for ax_offset in offsetspace:
            drug_ax['offset'] = ax_offset

            # print(as_offset, ax_offset)

            for k, m in enumerate([0.5, 1, 2]):
                normalize_effect(drug_bo, drug_as, 'n', 0.1, 0.5, m=m)
                normalize_effect_single(drug_ax, 0.95, m=m)
                conc_bo = get_conc(drug_bo, time)
                conc_as = get_conc(drug_as, time)
                conc_ax = get_conc(drug_ax, time)
                r = 1/(1 + (conc_bo/drug_bo['T315I'] + \
                            conc_ax/drug_ax['T315I'] + \
                            conc_as/drug_as['T315I'] + \
                            conc_bo*conc_as/drug_bo['T315I']/drug_as['T315I'] + \
                            conc_ax*conc_as/drug_ax['T315I']/drug_as['T315I'])**m) / \
                            (1/(1 + (conc_bo/drug_bo['native'] + \
                                     conc_ax/drug_ax['native'] + \
                                     conc_as/drug_as['native'] + \
                                     conc_bo*conc_as/drug_bo['native']/drug_as['native'] + \
                                     conc_ax*conc_as/drug_ax['native']/drug_as['native'])**m))

                if opt_targe == 'xauc':
                    auc = simps(r, x=time)
                elif opt_targe == 'min':
                    auc = min(r)

                if auc < best_effect[k]:
                    best_offsets[k]['as'] = as_offset
                    best_offsets[k]['ax'] = ax_offset
                    best_effect[k] = auc
                    print(best_offsets, m)


    print('best:', best_offsets)
    asboax_best_offsets = best_offsets[:]
    for i in range(3):
        asboax_best_offsets[i]['bo'] = 1 - asboax_best_offsets[i]['as']
        if asboax_best_offsets[i]['bo'] == 1:
            asboax_best_offsets[i]['bo'] = 0
        asboax_best_offsets[i]['ax'] -= asboax_best_offsets[i]['as']
        asboax_best_offsets[i]['as'] = 0.0
    print('best shifted:', best_offsets)

    markers = []

    for k, m, color in zip(range(3), [0.5, 1, 2],
                        [(109/255, 181/255, 255/255),
                         (177/255, 88/255, 0/255),
                         (6/255, 70/255, 81/255)]):


        markers.append(mlines.Line2D([], [], color=color, linestyle='-', linewidth=1.0))

        drug_bo['offset'] = best_offsets[k]['bo']
        drug_as['offset'] = best_offsets[k]['as']
        drug_ax['offset'] = best_offsets[k]['ax']

        normalize_effect(drug_bo, drug_as, 'n', 0.1, 0.5, m=m)
        normalize_effect_single(drug_ax, 0.95, m=m)
        conc_bo = get_conc(drug_bo, time)
        conc_as = get_conc(drug_as, time)
        conc_ax = get_conc(drug_ax, time)
        print('concs', drug_bo['D'], drug_as['D'], drug_ax['D'])
        print('concs', drug_bo['D'], drug_as['D'], drug_ax['D'])
        print('effect boas', get_effect(drug_bo, drug_as, 'n', m=m))

        r = 1/(1 + (conc_bo/drug_bo['T315I'] + \
                    conc_ax/drug_ax['T315I'] + \
                    conc_as/drug_as['T315I'] + \
                    conc_bo*conc_as/drug_bo['T315I']/drug_as['T315I'] + \
                    conc_ax*conc_as/drug_ax['T315I']/drug_as['T315I'])**m) / \
                    (1/(1 + (conc_bo/drug_bo['native'] + \
                             conc_ax/drug_ax['native'] + \
                             conc_as/drug_as['native'] + \
                             conc_bo*conc_as/drug_bo['native']/drug_as['native'] + \
                             conc_ax*conc_as/drug_ax['native']/drug_as['native'])**m))
        p = 1/(1 + (conc_bo/drug_bo['T315I'] + \
                    conc_as/drug_as['T315I'] + \
                    conc_bo*conc_as/drug_bo['T315I']/drug_as['T315I'])**m) / \
                    (1/(1 + (conc_bo/drug_bo['native'] + \
                             conc_as/drug_as['native'] + \
                             conc_bo*conc_as/drug_bo['native']/drug_as['native'])**m))

        asboax_traces.append(r[:])
        asbo_traces.append(p[:])



    drug_as = toml.load('intermediate/asciminib.toml')
    drug_ax = toml.load('intermediate/axitinib.toml')

    print(drug_as['name'], drug_as['D'])
    print(drug_ax['name'], drug_ax['D'])

    print(drug_as['name'], get_effect_single(drug_as))
    print(drug_ax['name'], get_effect_single(drug_ax))

    print(drug_as['T315I'])
    print(drug_ax['T315I'])

    for m in [0.5, 1, 2]:
        normalize_effect_single(drug_as, 0.1, m=m)
        normalize_effect_single(drug_ax, 0.95, m=m)
        print('m = ', m, ':')
        print('\t', drug_as['name'], drug_as['D'])
        print('\t', drug_ax['name'], drug_ax['D'])
        print('\t', get_effect(drug_as, drug_as, 'n', m=m))

    print(drug_as['name'], get_effect_single(drug_as))
    print(drug_ax['name'], get_effect_single(drug_ax))

    print(drug_as['name'], drug_as['D'])
    print(drug_ax['name'], drug_ax['D'])

    drug_as['offset'] = 0

    offsetspace = np.linspace(0, 1, 13)[:-1]
    # offsetspace = [0]
    time = np.linspace(0, 1, 1000)

    best_offsets = [{}, {}, {}]#{'as': 0.0, 'ax': 0.0}
    best_effect = [999, 999, 999]


    for ax_offset in offsetspace:
        drug_ax['offset'] = ax_offset

            # print(as_offset, ax_offset)

        for k, m in enumerate([0.5, 1, 2]):
            normalize_effect_single(drug_as, 0.1, m=m)
            normalize_effect_single(drug_ax, 0.95, m=m)
            conc_as = get_conc(drug_as, time)
            conc_ax = get_conc(drug_ax, time)
            r = 1/(1 + (conc_ax/drug_ax['T315I'] + \
                        conc_as/drug_as['T315I'] + \
                        conc_ax*conc_as/drug_ax['T315I']/drug_as['T315I'])**m) / \
                        (1/(1 + (conc_ax/drug_ax['native'] + \
                                 conc_as/drug_as['native'] + \
                                 conc_ax*conc_as/drug_ax['native']/drug_as['native'])**m))

            if opt_targe == 'xauc':
                auc = simps(r, x=time)
            elif opt_targe == 'min':
                auc = min(r)

            if auc < best_effect[k]:
                best_offsets[k]['ax'] = ax_offset
                best_effect[k] = auc
                print(best_offsets, m)


    print(best_offsets)
    asax_best_offsets = best_offsets[:]
    for i in range(3):
        asax_best_offsets[i]['as'] = 0.0

    markers = []

    as_traces = []
    asax_traces = []

    for k, m, color in zip(range(3), [0.5, 1, 2],
                        [(109/255, 181/255, 255/255),
                         (177/255, 88/255, 0/255),
                         (6/255, 70/255, 81/255)]):


        markers.append(mlines.Line2D([], [], color=color, linestyle='-', linewidth=1.0))

        drug_as['offset'] = 0
        drug_ax['offset'] = best_offsets[k]['ax']

        normalize_effect_single(drug_as, 0.1, m=m)
        normalize_effect_single(drug_ax, 0.95, m=m)
        conc_as = get_conc(drug_as, time)
        conc_ax = get_conc(drug_ax, time)

        r = 1/(1 + (conc_ax/drug_ax['T315I'] + \
                    conc_as/drug_as['T315I'] + \
                    conc_ax*conc_as/drug_ax['T315I']/drug_as['T315I'])**m) / \
                    (1/(1 + (conc_ax/drug_ax['native'] + \
                             conc_as/drug_as['native'] + \
                             conc_ax*conc_as/drug_ax['native']/drug_as['native'])**m))

        p = 1/(1 + (conc_as/drug_as['T315I'])**m) / \
                    (1/(1 + (conc_as/drug_as['native'])**m))
        asax_traces.append(r[:])
        as_traces.append(p[:])


    print(asboax_best_offsets)
    print(asax_best_offsets)

    for k, m in enumerate([0.5, 1, 2]):
        fig, axs = plt.subplots()
        fig.set_size_inches(5.5, 4)

        axs.plot(time, asax_traces[k], linewidth=1.0, color=PALETTE_4[0], label='Asciminib + Axitinib')
        axs.plot(time, as_traces[k], linewidth=1.0, linestyle='--', color=PALETTE_4[0], label='Asciminib')
        axs.plot(time, asboax_traces[k], linewidth=1.0, color=PALETTE_4[2], label='Asciminib + Bosutinib + Axitinib')
        axs.plot(time, asbo_traces[k], linewidth=1.0, linestyle='--', color=PALETTE_4[2], label='Asciminib + Bosutinib')

        axs.axvline(asboax_best_offsets[k]['bo'], color='grey', linewidth=1.0, linestyle=':')
        axs.axvline(asboax_best_offsets[k]['as'], color='grey', linewidth=1.0, linestyle=':')
        axs.axvline(asboax_best_offsets[k]['ax'], color='grey', linewidth=1.0, linestyle=':')
        axs.axvline(asax_best_offsets[k]['as'], color='grey', linewidth=1.0, linestyle=':')
        axs.axvline(asax_best_offsets[k]['ax'], color='grey', linewidth=1.0, linestyle=':')

        occupied = {}

        yoff = 0
        xoff = asboax_best_offsets[k]['bo']
        if xoff in occupied:
            yoff += 0.035*occupied[xoff]
            occupied[xoff] += 1
        else:
            occupied[xoff] = 1
        axs.text(xoff, 1.03 + yoff, 'BO', transform=axs.transAxes, size=7,
                 color=PALETTE_4[2],
                 verticalalignment='center', horizontalalignment='center')
        yoff = 0
        xoff = asboax_best_offsets[k]['as']
        if xoff in occupied:
            yoff += 0.035*occupied[xoff]
            occupied[xoff] += 1
        else:
            occupied[xoff] = 1
        axs.text(xoff, 1.03 + yoff, 'AS', transform=axs.transAxes, size=7,
                 color=PALETTE_4[2],
                 verticalalignment='center', horizontalalignment='center')
        yoff = 0
        xoff = asboax_best_offsets[k]['ax']
        if xoff in occupied:
            yoff += 0.035*occupied[xoff]
            occupied[xoff] += 1
        else:
            occupied[xoff] = 1
        axs.text(xoff, 1.03 + yoff, 'AX', transform=axs.transAxes, size=7,
                 color=PALETTE_4[2],
                 verticalalignment='center', horizontalalignment='center')
        yoff = 0
        xoff = asax_best_offsets[k]['as']
        if xoff in occupied:
            yoff += 0.035*occupied[xoff]
            occupied[xoff] += 1
        else:
            occupied[xoff] = 1
        axs.text(xoff, 1.03 + yoff, 'AS', transform=axs.transAxes, size=7,
                 color=PALETTE_4[0],
                 verticalalignment='center', horizontalalignment='center')
        yoff = 0
        xoff = asax_best_offsets[k]['ax']
        if xoff in occupied:
            yoff += 0.035*occupied[xoff]
            occupied[xoff] += 1
        else:
            occupied[xoff] = 1
        axs.text(xoff, 1.03 + yoff, 'AX', transform=axs.transAxes, size=7,
                 color=PALETTE_4[0],
                 verticalalignment='center', horizontalalignment='center')


        axs.set_xlim(0, 1)
        axs.set_ylim(0.3, 10**np.ceil(np.log10(axs.get_ylim()[1])))
        axs.set_yscale('log')
        axs.set_xlabel('Time [h]')
        axs.set_ylabel(r'$\chi f_v$')

        wt_area = plt.Polygon([(0, 0.3), (1, 0.3), (1, 1), (0, 1)], color='lightgrey')
        axs.add_patch(wt_area)

        axs.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axs.set_xticklabels([int(24*x) for x in axs.get_xticks()])

        axs.text(0.5, 0.05, 'WT grows faster', color='w',
                 verticalalignment='center', horizontalalignment='center',
                 transform=axs.transAxes)
        axs.text(0.5, 0.95, 'T315I grows faster', color='lightgrey',
                 verticalalignment='center', horizontalalignment='center',
                 transform=axs.transAxes)

        full_lines1 = mlines.Line2D([], [], color=PALETTE_4[0], linestyle='-', linewidth=1.0)
        dashed_lines1 = mlines.Line2D([], [], color=PALETTE_4[0], linestyle='--', linewidth=1.0)
        full_lines2 = mlines.Line2D([], [], color=PALETTE_4[2], linestyle='-', linewidth=1.0)
        dashed_lines2 = mlines.Line2D([], [], color=PALETTE_4[2], linestyle='--', linewidth=1.0)
        blank = mlines.Line2D([], [], color='w', linestyle='-', linewidth=0.0, alpha=0.0)

        pos11 = axs.get_position()
        print(pos11)
        # symbols = [full_lines1, dashed_lines1, full_lines2, dashed_lines2]
        symbols = [full_lines1, dashed_lines1, full_lines2, dashed_lines2, blank, blank, blank, blank]
        # labels = [
        #     'Asciminib + axitinib',
        #     'Asciminib',
        #     'Asciminib + bosutinib + axitinib',
        #     'Asciminib + bosutinib'
        # ]
        # labels = [
        #     'AS + AX',
        #     'AS',
        #     'AS + BO + AX',
        #     'AS + BO'
        # ]
        labels = [
            'AS + AX',
            'AS',
            'AS + BO + AX',
            'AS + BO',
            '',
            'AS - Asciminib',
            'AX - Axitinib',
            'BO - Bosutinib'
        ]



        box = axs.get_position()
        axs.set_position([box.x0, box.y0, box.width * 0.727, box.height])
        axs.legend(symbols, labels, loc=(1.05, 0.3), ncol=1, frameon=False)

        if save is not None:
            pdf_out.savefig()
        else:
            plt.show()


    if save is not None:
        pdf_out.close()


if __name__ == '__main__':
    main()
