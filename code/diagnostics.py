"""
Diagnostic plots for telling if there might be a useful way of adding low dose drug
to another drug treatment for preventing some particular mutation
"""


import click
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.integrate import simps
from scipy.optimize import minimize
import toml


# Custom color palettes used in some plots
PALETTE_3 = [
    (0/255, 0/255, 0/255),
    (200/255, 180/255, 55/255),
    (0/255, 114/255, 178/255),
]

PALETTE_4 = [
    (0/255, 0/255, 0/255),
    (200/255, 180/255, 55/255),
    (204/255, 121/255, 167/255),
    (0/255, 114/255, 178/255),
]


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


@click.group()
def main():
    """
    main construct for click to function in self-contained file
    """
    pass


@main.command()
@click.argument('adrug', type=click.Path())
@click.argument('bdrug', type=click.Path())
@click.argument('mutant', type=str)
@click.argument('dataset', type=int)
@click.option('-o', '--outfile', type=click.Path(), default=None)
def make(adrug, bdrug, mutant, dataset, outfile):
    """
    create diagnostic plots for given drug parameter files
    """

    # parsing of core variables
    drug_a = toml.load(adrug)
    drug_b = toml.load(bdrug)

    # make sure that drug dosing schedules line up
    assert(drug_a['tau']*drug_a['repeats'] == drug_b['tau']*drug_b['repeats'])

    cell1 = [drug_a['native'][dataset], drug_b['native'][dataset]]
    cell2 = [drug_a[mutant][dataset], drug_b[mutant][dataset]]

    adose = np.arange(drug_a['min_dose'], drug_a['max_dose'],
                      drug_a['max_dose'] / 128)
    bdose = np.arange(drug_b['min_dose'], drug_b['max_dose'],
                      drug_b['max_dose'] / 128)

    extent = [drug_b['min_dose'], drug_b['max_dose'],
              drug_a['min_dose'], drug_a['max_dose']]
    aspect = (drug_b['max_dose'] - drug_b['min_dose']) / \
             (drug_a['max_dose'] - drug_a['min_dose'])

    # shift offsets such that drug_a has offset 0 and drug b has all the offset
    period = drug_a['tau']*drug_a['repeats']
    ba_offset = drug_b['offset'] - drug_a['offset']
    if ba_offset < 0:
        ba_offset += period
    drug_a['offset'] = 0
    drug_b['offset'] = ba_offset


    if outfile is not None:
        # first time, construct the pdf
        pdf_out = PdfPages(outfile + '.pdf')


    # heatmap with dose trail and velocity-time charts

    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(12, 8)


    T_RESOLUTION = 240
    T_SKIP = 30
    t = np.arange(0, drug_a['tau']*drug_a['repeats'], drug_a['tau']*drug_a['repeats']/T_RESOLUTION)

    def get_conc(d):
        tt = t%d['tau']
        c = d['S']*d['F']*d['D']*d['ka'] / (d['Vd']*(d['ka'] - d['ke'])) * \
            (np.exp(-d['ke']*tt)/(1 - np.exp(-d['ke']*d['tau'])) - \
             np.exp(-d['ka']*tt)/(1 - np.exp(-d['ka']*d['tau'])))
        return np.roll(c, int(d['offset']/(d['tau']*d['repeats'])*T_RESOLUTION))

    conc_a = get_conc(drug_a)
    conc_b = get_conc(drug_b)

    adose_c = np.arange(0, max(conc_a)*1.05, max(conc_a)*1.05 / 128)
    bdose_c = np.arange(0, max(conc_b)*1.05, max(conc_b)*1.05 / 128)

    extent_c = [0, max(conc_b)*1.05, 0, max(conc_a)*1.05]
    aspect_c = max(conc_b) / max(conc_a)

    Xc, Yc = np.meshgrid(bdose_c, adose_c)
    # calculate difference between wildtype and mutant growth rates
    # for each position in a dose_a vs dose_b plot
    dvce_c = 1/(1 + Yc/cell2[0] + Xc/cell2[1]) - \
             1/(1 + Yc/cell1[0] + Xc/cell1[1])
    dvcn_c = 1/(1 + Yc/cell2[0] + Xc/cell2[1] + Xc*Yc/cell2[0]/cell2[1]) - \
             1/(1 + Yc/cell1[0] + Xc/cell1[1] + Xc*Yc/cell1[0]/cell1[1])

    # vmax_c = max([abs(dvce).max(), abs(dvcn).max()])
    # vmin_c = -max([abs(dvce).max(), abs(dvcn).max()])
    vmax_c = 1
    vmin_c = -1

    im00 = axs[0][0].imshow(dvce_c, interpolation='bicubic', cmap=cm.BrBG,
                            origin='lower', extent=extent_c, aspect=aspect_c,
                            vmax=vmax_c, vmin=vmin_c)
    axs[0][0].contour(dvce_c, [0], colors='grey', origin='lower', extent=extent_c)

    axs[0][0].set_title('Exclusive interaction')
    axs[0][0].set_xlabel(drug_b['name'] + r' dose [WT IC$_{50}$ multiples]')
    axs[0][0].set_ylabel(drug_a['name'] + r' dose [WT IC$_{50}$ multiples]')

    im = axs[1][0].imshow(dvcn_c, interpolation='bicubic', cmap=cm.BrBG,
                          origin='lower', extent=extent_c, aspect=aspect_c,
                          vmax=vmax_c, vmin=vmin_c)
    cnt = axs[1][0].contour(dvcn_c, [0], colors='grey', origin='lower', extent=extent_c)

    axs[1][0].set_title('Nonexclusive interaction')
    axs[1][0].set_xlabel(drug_b['name'] + ' dose [WT IC$_{50}$ multiples]')
    axs[1][0].set_ylabel(drug_a['name'] + ' dose [WT IC$_{50}$ multiples]')

    cb = fig.colorbar(im, ax=axs[0][0], fraction=0.046, pad=0.04)
    cb.add_lines(cnt)
    cb = fig.colorbar(im, ax=axs[1][0], fraction=0.046, pad=0.04)
    cb.add_lines(cnt)

    axs[0][0].plot(np.append(conc_b, conc_b[0]), np.append(conc_a, conc_a[0]), linewidth=1.0, color='black')
    axs[1][0].plot(np.append(conc_b, conc_b[0]), np.append(conc_a, conc_a[0]), linewidth=1.0, color='black')
    axs[0][0].scatter(conc_b[0], conc_a[0], marker='o', color='black', facecolors='none')
    axs[1][0].scatter(conc_b[0], conc_a[0], marker='o', color='black', facecolors='none')
    for i in range(T_RESOLUTION//T_SKIP):
        ca1 = conc_a[i*T_SKIP]
        ca2 = conc_a[i*T_SKIP + 1]
        cb1 = conc_b[i*T_SKIP]
        cb2 = conc_b[i*T_SKIP + 1]

        xi, yi = (int(round(n*128)) for n in (cb2 / max(bdose_c), ca2 / max(adose_c)))
        value = im00.get_array()[yi, xi]
        color = im00.cmap(im00.norm(value))
        axs[0][0].annotate("", xy=(cb2, ca2), xytext=(cb1, ca1),
                           arrowprops=dict(width=0.0, headwidth=5.0, headlength=8.0,
                                           facecolor=color))
        xi, yi = (int(round(n*128)) for n in (cb2 / max(bdose_c), ca2 / max(adose_c)))
        value = im.get_array()[yi, xi]
        color = im.cmap(im.norm(value))
        axs[1][0].annotate("", xy=(cb2, ca2), xytext=(cb1, ca1),
                           arrowprops=dict(width=0.0, headwidth=5.0, headlength=8.0,
                                           facecolor=color))

    dvel = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1]) - \
           1/(1 + conc_a/cell1[0] + conc_b/cell1[1])
    dven = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1] + conc_b*conc_a/cell2[0]/cell2[1]) - \
           1/(1 + conc_a/cell1[0] + conc_b/cell1[1] + conc_b*conc_a/cell1[0]/cell1[1])
    pl = 1/(1 + conc_a/cell2[0]) - 1/(1 + conc_a/cell1[0])

    rvel = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1]) / \
           (1/(1 + conc_a/cell1[0] + conc_b/cell1[1]))
    rven = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1] + conc_b*conc_a/cell2[0]/cell2[1]) / \
           (1/(1 + conc_a/cell1[0] + conc_b/cell1[1] + conc_b*conc_a/cell1[0]/cell1[1]))
    rpl = 1/(1 + (conc_a/cell2[0])) / (1/(1 + (conc_a/cell1[0])))

    print(conc_a, conc_b)

    print(1/(1 + conc_a[0]/cell2[0] + conc_b[0]/cell2[1]) - \
           1/(1 + conc_a[0]/cell1[0] + conc_b[0]/cell1[1]))

    print(1/(1 + conc_b[0]/cell2[0] + conc_a[0]/cell2[1]) - \
           1/(1 + conc_b[0]/cell1[0] + conc_a[0]/cell1[1]))

    axs[0][1].plot(t*24, dvel, label=drug_a['name'] + ' + ' + drug_b['name'], color='k')
    axs[0][1].plot(t*24, pl, label=drug_a['name'], color='grey')
    axs[1][1].plot(t*24, dven, label=drug_a['name'] + ' + ' + drug_b['name'], color='k')
    axs[1][1].plot(t*24, pl, label=drug_a['name'], color='grey')

    axs[0][2].plot(t*24, rvel, label=drug_a['name'] + ' + ' + drug_b['name'], color='k')
    axs[0][2].plot(t*24, rpl, label=drug_a['name'], color='grey')
    axs[1][2].plot(t*24, rven, label=drug_a['name'] + ' + ' + drug_b['name'], color='k')
    axs[1][2].plot(t*24, rpl, label=drug_a['name'], color='grey')

    for i in [0, 1]:
        axs[0][1 + i].set_xticks(np.arange(0, 24*drug_a['tau']*drug_a['repeats'] + 1, 2)[1:])
        axs[1][1 + i].set_xticks(np.arange(0, 24*drug_a['tau']*drug_a['repeats'] + 1, 2)[1:])
        axs[0][1 + i].spines['top'].set_color('none')
        axs[0][1 + i].spines['right'].set_color('none')
        axs[1][1 + i].spines['top'].set_color('none')
        axs[1][1 + i].spines['right'].set_color('none')

        axs[0][1 + i].set_xlim(0, 24*drug_a['tau']*drug_a['repeats'])
        axs[1][1 + i].set_xlim(0, 24*drug_a['tau']*drug_a['repeats'])
        axs[0][1 + i].set_xlabel('Time [h]')
        axs[1][1 + i].set_xlabel('Time [h]')

    axs[0][1].spines['bottom'].set_position('center')
    axs[1][1].spines['bottom'].set_position('center')

    axs[0][1].set_ylim(-1, 1)
    axs[1][1].set_ylim(-1, 1)

    axs[0][1].set_ylabel(mutant + '-Native growth rate difference along dose trace')
    axs[1][1].set_ylabel(mutant + '-Native growth rate difference along dose trace')
    axs[0][1].legend()
    axs[1][1].legend()

    axs[0][2].set_yscale('log')
    axs[1][2].set_yscale('log')

    plt.tight_layout()

    if outfile is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # velocity difference charts at various hill coefficients

    fig, axs = plt.subplots(ncols=2, nrows=2)
    fig.set_size_inches(8, 8)

    assert(drug_a['tau']*drug_a['repeats'] == drug_b['tau']*drug_b['repeats'])

    T_RESOLUTION = 240
    t = np.arange(0, drug_a['tau']*drug_a['repeats'], drug_a['tau']*drug_a['repeats']/T_RESOLUTION)

    def get_conc(d):
        tt = t%d['tau']
        c = d['S']*d['F']*d['D']*d['ka'] / (d['Vd']*(d['ka'] - d['ke'])) * \
            (np.exp(-d['ke']*tt)/(1 - np.exp(-d['ke']*d['tau'])) - \
             np.exp(-d['ka']*tt)/(1 - np.exp(-d['ka']*d['tau'])))
        return np.roll(c, int(d['offset']/(d['tau']*d['repeats'])*T_RESOLUTION))

    conc_a = get_conc(drug_a)
    conc_b = get_conc(drug_b)

    hill_list = [0.5, 1, 2, 4]
    colors_main = ['k']*len(hill_list)
    styles_main = [':', '-', '--', '-.']

    for m, color, style in zip(hill_list, colors_main, styles_main):
        dvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) - \
               1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m)
        dven = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1] + conc_b*conc_a/cell2[0]/cell2[1])**m) - \
               1/(1 + (conc_a/cell1[0] + conc_b/cell1[1] + conc_b*conc_a/cell1[0]/cell1[1])**m)
        # rvel = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1]) / \
        #        (1/(1 + conc_a/cell1[0] + conc_b/cell1[1]))
        # rven = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1] + conc_b*conc_a/cell2[0]/cell2[1]) / \
        #        (1/(1 + conc_a/cell1[0] + conc_b/cell1[1] + conc_b*conc_a/cell1[0]/cell1[1]))
        rvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) / \
               (1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m))
        rven = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1] + conc_b*conc_a/cell2[0]/cell2[1])**m) / \
               (1/(1 + (conc_a/cell1[0] + conc_b/cell1[1] + conc_b*conc_a/cell1[0]/cell1[1])**m))

        axs[0][0].plot(t*24, dvel, label=np.round(m, 2), color=color, linestyle=style)
        axs[1][0].plot(t*24, dven, label=np.round(m, 2), color=color, linestyle=style)
        axs[0][1].plot(t*24, rvel, label=np.round(m, 2), color=color, linestyle=style)
        axs[1][1].plot(t*24, rven, label=np.round(m, 2), color=color, linestyle=style)
    for i in [0, 1]:
        axs[0][0 + i].set_xticks(np.arange(0, 24*drug_a['tau']*drug_a['repeats'] + 1, 2)[1:])
        axs[1][0 + i].set_xticks(np.arange(0, 24*drug_a['tau']*drug_a['repeats'] + 1, 2)[1:])
        axs[0][0 + i].spines['top'].set_color('none')
        axs[0][0 + i].spines['right'].set_color('none')
        axs[1][0 + i].spines['top'].set_color('none')
        axs[1][0 + i].spines['right'].set_color('none')

        axs[0][0 + i].set_xlim(0, 24*drug_a['tau']*drug_a['repeats'])
        axs[1][0 + i].set_xlim(0, 24*drug_a['tau']*drug_a['repeats'])
        axs[0][0 + i].set_xlabel('Time [h]')
        axs[1][0 + i].set_xlabel('Time [h]')

    axs[0][0].spines['bottom'].set_position('center')
    axs[1][0].spines['bottom'].set_position('center')

    axs[0][0].set_ylim(-1, 1)
    axs[1][0].set_ylim(-1, 1)

    axs[0][1].set_yscale('log')
    axs[1][1].set_yscale('log')

    axs[0][0].set_ylabel(mutant + '-Native growth rate difference along dose trace')
    axs[1][0].set_ylabel(mutant + '-Native growth rate difference along dose trace')
    axs[0][1].set_ylabel(mutant + '/Native growth rate ratio along dose trace')
    axs[1][1].set_ylabel(mutant + '/Native growth rate ratio along dose trace')
    axs[0][0].legend()

    plt.tight_layout()

    if outfile is not None:
        pdf_out.savefig()
    else:
        plt.show()


    # # done with plots, close plot pdf
    if outfile is not None:
        pdf_out.close()


@main.command()
@click.argument('adrug', type=click.Path())
@click.argument('bdrug', type=click.Path())
@click.argument('mutant', type=str)
@click.argument('dataset', type=int)
@click.option('-m', '--interaction-mode', type=str)
@click.option('-t', '--optimization-target', type=str)
def optimize_offset(adrug, bdrug, mutant, dataset, interaction_mode, optimization_target):
    """
    find the offset for drug b that minimizes the AUC or minimum of the f_v(t) curve
    -t auc | min
    """
    # parsing of core variables

    drug_a = toml.load(adrug)
    drug_b = toml.load(bdrug)

    cell1 = [drug_a['native'][dataset], drug_b['native'][dataset]]
    cell2 = [drug_a[mutant][dataset], drug_b[mutant][dataset]]

    adose = np.arange(drug_a['min_dose'], drug_a['max_dose'],
                      drug_a['max_dose'] / 128)
    bdose = np.arange(drug_b['min_dose'], drug_b['max_dose'],
                      drug_b['max_dose'] / 128)


    extent = [drug_b['min_dose'], drug_b['max_dose'],
              drug_a['min_dose'], drug_a['max_dose']]
    aspect = (drug_b['max_dose'] - drug_b['min_dose']) / \
             (drug_a['max_dose'] - drug_a['min_dose'])

    assert(drug_a['tau']*drug_a['repeats'] == drug_b['tau']*drug_b['repeats'])

    T_RESOLUTION = 240
    T_SKIP = 30
    t = np.arange(0, drug_a['tau']*drug_a['repeats'], drug_a['tau']*drug_a['repeats']/T_RESOLUTION)

    OFFSET_RESOLUTION = T_RESOLUTION
    min_offset = 0.0
    max_offset = drug_b['tau']*drug_b['repeats']
    offsets = np.arange(min_offset, max_offset, (max_offset - min_offset)/OFFSET_RESOLUTION)

    best_offset_e = [0.0, 1e20]
    best_offset_n = [0.0, 1e20]

    def get_conc(d, offset=None):
        tt = t%d['tau']
        c = d['S']*d['F']*d['D']*d['ka'] / (d['Vd']*(d['ka'] - d['ke'])) * \
            (np.exp(-d['ke']*tt)/(1 - np.exp(-d['ke']*d['tau'])) - \
             np.exp(-d['ka']*tt)/(1 - np.exp(-d['ka']*d['tau'])))
        if offset is None:
            return np.roll(c, int(d['offset']/(d['tau']*d['repeats'])*T_RESOLUTION))
        else:
            return np.roll(c, int(offset/(d['tau']*d['repeats'])*T_RESOLUTION))


    for offset in offsets:
        conc_a = get_conc(drug_a)
        conc_b = get_conc(drug_b, offset)

        dvel = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1]) - \
               1/(1 + conc_a/cell1[0] + conc_b/cell1[1])
        dven = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1] + conc_b*conc_a/cell2[0]/cell2[1]) - \
               1/(1 + conc_a/cell1[0] + conc_b/cell1[1] + conc_b*conc_a/cell1[0]/cell1[1])
        pl = 1/(1 + conc_a/cell2[0]) - 1/(1 + conc_a/cell1[0])

        rvel = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1]) / \
               (1/(1 + conc_a/cell1[0] + conc_b/cell1[1]))
        rven = 1/(1 + conc_a/cell2[0] + conc_b/cell2[1] + conc_b*conc_a/cell2[0]/cell2[1]) / \
               (1/(1 + conc_a/cell1[0] + conc_b/cell1[1] + conc_b*conc_a/cell1[0]/cell1[1]))

        if optimization_target == 'd-auc':
            target_e = simps(dvel, x=t)
            target_n = simps(dven, x=t)
        elif optimization_target == 'min':
            target_e = min(dvel)
            target_n = min(dven)
        elif optimization_target == 'xauc':
            target_e = simps(rvel, x=t)
            target_n = simps(rven, x=t)
        else:
            print('Unknown optimization target (use auc | min):', optimization_target)


        print(offset, target_e, target_n, sep='\t')

        if target_e < best_offset_e[1]:
            best_offset_e = [offset, target_e]
        if target_n < best_offset_n[1]:
            best_offset_n = [offset, target_n]

    print(best_offset_e)
    print(best_offset_n)

    if interaction_mode == 'e':
        drug_b['offset'] = best_offset_e[0]
    elif interaction_mode == 'n':
        drug_b['offset'] = best_offset_n[0]
    else:
        print('Unknown interaction mode (use e or n):', interaction_mode)

    with open(bdrug, 'w') as out_toml:
        toml.dump(drug_b, out_toml)


@main.command()
@click.option('-a', '--adrugs', type=click.Path(), multiple=True)
@click.option('-b', '--bdrug', type=click.Path())
@click.option('--mutant', type=str)
@click.option('--dataset', type=int)
@click.option('-m', '--interaction-mode', type=str)
@click.option('-t', '--optimization-target', type=str)
@click.option('--save', type=click.Path(), default=None)
def alltrace(adrugs, bdrug, mutant, dataset, interaction_mode, optimization_target, save):
    """
    interactions between many drugs (a) and one other (b)
    at m = 0.5, 1, 2
    """

    assert optimization_target in ['min', 'xauc']
    opt_targe = optimization_target
    # opt_targe = 'xauc'
    # opt_targe = 'min'


    if save is not None:
        pdf_out = PdfPages(save)


    # version with plasma concentration based doses

    # bdrug = 'intermediate/axitinib.toml'
    # dataset = 0
    # mutant = 'T315I'

    for k, m in enumerate([0.5, 1, 2]):

        fig, axs = plt.subplots()
        fig.set_size_inches(5.5, 4)

        occupied = {}

        names = []

        # for i, adrug in enumerate(['intermediate/imatinib.toml', 'intermediate/nilotinib.toml',
                                   # 'intermediate/dasatinib.toml', 'intermediate/bosutinib.toml']):
        for i, adrug in enumerate(adrugs):

            drug_a = toml.load(adrug)
            drug_b = toml.load(bdrug)

            print(drug_a['name'], drug_a['D'], get_effect_single(drug_a))
            print(drug_b['name'], drug_b['D'], get_effect_single(drug_b))

            names.append(drug_a['name'])

            print(drug_a[mutant])
            print(drug_b[mutant])
            print(drug_a['native'])
            print(drug_b['native'])

            cell1 = [drug_a['native'][dataset], drug_b['native'][dataset]]
            cell2 = [drug_a[mutant][dataset], drug_b[mutant][dataset]]

            print(cell1, cell2)

            # shift offsets such that drug_a has offset 0 and drug b has all the offset
            period = drug_a['tau']*drug_a['repeats']
            ba_offset = drug_b['offset'] - drug_a['offset']
            if ba_offset < 0:
                ba_offset += period
            drug_a['offset'] = 0
            drug_b['offset'] = ba_offset

            assert(drug_a['tau']*drug_a['repeats'] == drug_b['tau']*drug_b['repeats'])

            T_RESOLUTION = 240
            t = np.arange(0, drug_a['tau']*drug_a['repeats'], drug_a['tau']*drug_a['repeats']/T_RESOLUTION)

            conc_a = get_conc(drug_a, t)
            conc_b = get_conc(drug_b, t)

            offsetspace = np.linspace(0, 1, 49)[:-1]
            best_offsets = [0.0, 0.0, 0.0]#{'as': 0.0, 'ax': 0.0}
            best_effect = [999, 999, 999]


            for b_offset in offsetspace:
                drug_b['offset'] = b_offset
                conc_a = get_conc(drug_a, t)
                conc_b = get_conc(drug_b, t)
                # vfte_m = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m)
                # vfte_w = 1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m)

                # dvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) - \
                #        1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m)
                # pl = 1/(1 + (conc_a/cell2[0])**m) - 1/(1 + (conc_a/cell1[0])**m)

                rvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) / \
                       (1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m))
                rpl = 1/(1 + (conc_a/cell2[0])**m) / (1/(1 + (conc_a/cell1[0])**m))

                if (opt_targe == 'xauc'):
                    auc = simps(rvel, x=t)
                elif (opt_targe == 'min'):
                    auc = min(rvel)

                if auc < best_effect[k]:
                    best_offsets[k] = b_offset
                    best_effect[k] = auc

            print(best_offsets)

            drug_b['offset'] = best_offsets[k]
            conc_a = get_conc(drug_a, t)
            conc_b = get_conc(drug_b, t)

            vfte_m = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m)
            vfte_w = 1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m)

            dvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) - \
                    1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m)
            pl = 1/(1 + (conc_a/cell2[0])**m) - 1/(1 + (conc_a/cell1[0])**m)

            rvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) / \
                    (1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m))
            rpl = 1/(1 + (conc_a/cell2[0])**m) / (1/(1 + (conc_a/cell1[0])**m))

            axs.plot(t, rvel, linewidth=1.0, color=PALETTE_4[i], label=drug_a['name'])
            axs.plot(t, rpl, linewidth=1.0, linestyle='--', color=PALETTE_4[i])

            axs.axvline(drug_b['offset'], color='grey', linewidth=1.0, linestyle=':')

            yoff = 0
            xoff = best_offsets[k]
            if xoff in occupied:
                yoff += 0.04*occupied[xoff]
                occupied[xoff] += 1
            else:
                occupied[xoff] = 1
            axs.text(xoff, 1.03 + yoff, drug_b['name'][:2].upper(), transform=axs.transAxes, size=8,
                     color=PALETTE_4[i],
                     verticalalignment='center', horizontalalignment='center')

            axs.set_xlim(0, drug_a['tau']*drug_a['repeats'])
            axs.set_yscale('log')
            axs.set_xlabel('Time [h]')
            axs.set_ylabel(r'$\chi f_v$')

            wt_area = plt.Polygon([(0, 0.3), (1, 0.3), (1, 1), (0, 1)], color='lightgrey')
            axs.add_patch(wt_area)

            axs.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            print(axs.get_xticks())
            axs.set_xticklabels([int(24*x) for x in axs.get_xticks()])

        axs.set_ylim(0.3, 10**np.ceil(np.log10(axs.get_ylim()[1])))
        print(axs.get_ylim())
        axs.text(0.5, 0.05, 'WT grows faster', color='w',
                 verticalalignment='center', horizontalalignment='center',
                 transform=axs.transAxes)
        axs.text(0.5, 0.95, mutant + ' grows faster', color='lightgrey',
                 verticalalignment='center', horizontalalignment='center',
                 transform=axs.transAxes)

        full_lines = mlines.Line2D([], [], color='k', linestyle='-', linewidth=1.0)
        dashed_lines = mlines.Line2D([], [], color='k', linestyle='--', linewidth=1.0)
        # markers = [mlines.Line2D([], [], color=color, linestyle='-', linewidth=1.0) for color in PALETTE_4]
        markers = [mlines.Line2D([], [], color=color, linestyle='none', marker='.', markersize=15.0, linewidth=1.0) for color in PALETTE_4[:len(adrugs)]]

        pos11 = axs.get_position()
        print(pos11)
        symbols = markers + [full_lines, dashed_lines]
        # labels = ['Imatinib', 'Nilotinib', 'Dasatinib', 'Bosutinib'] + ['Combination\nwith Axitinib', 'Monodrug']
        labels = names + ['Combination\nwith ' + drug_b['name'], 'Monodrug']
        print(symbols, labels)
        box = axs.get_position()
        axs.set_position([box.x0, box.y0, box.width * 0.727, box.height])
        axs.legend(symbols, labels, loc=(1.05, 0.3), ncol=1, frameon=False)

        if save is not None:
            pdf_out.savefig()
        else:
            plt.show()




    # version with separately normalized doses

    # bdrug = 'intermediate/axitinib.toml'
    # dataset = 0
    # mutant = 'T315I'

    for k, m in enumerate([0.5, 1, 2]):

        fig, axs = plt.subplots()
        fig.set_size_inches(5.5, 4)

        names = []

        occupied = {}

        # for i, adrug in enumerate(['intermediate/imatinib.toml', 'intermediate/nilotinib.toml',
        #                            'intermediate/dasatinib.toml', 'intermediate/bosutinib.toml']):
        #                            # 'intermediate/dasatinib.toml', 'intermediate/bosutinib.toml']):
        for i, adrug in enumerate(adrugs):

            drug_a = toml.load(adrug)
            drug_b = toml.load(bdrug)

            print(drug_a['name'], drug_a['D'], get_effect_single(drug_a))
            print(drug_b['name'], drug_b['D'], get_effect_single(drug_b))

            names.append(drug_a['name'])

            print(drug_a['T315I'])
            print(drug_b['T315I'])
            print(drug_a['native'])
            print(drug_b['native'])

            normalize_effect_single(drug_a, 0.1, m=m)
            normalize_effect_single(drug_b, 0.95, m=m)
            print(drug_a['name'], m, drug_a['D'], get_effect_single(drug_a, m=m))
            print(drug_b['name'], m, drug_b['D'], get_effect_single(drug_b, m=m))

            cell1 = [drug_a['native'][dataset], drug_b['native'][dataset]]
            cell2 = [drug_a[mutant][dataset], drug_b[mutant][dataset]]

            print(cell1, cell2)

            # shift offsets such that drug_a has offset 0 and drug b has all the offset
            period = drug_a['tau']*drug_a['repeats']
            ba_offset = drug_b['offset'] - drug_a['offset']
            if ba_offset < 0:
                ba_offset += period
            drug_a['offset'] = 0
            drug_b['offset'] = ba_offset

            assert(drug_a['tau']*drug_a['repeats'] == drug_b['tau']*drug_b['repeats'])

            T_RESOLUTION = 240
            t = np.arange(0, drug_a['tau']*drug_a['repeats'], drug_a['tau']*drug_a['repeats']/T_RESOLUTION)

            conc_a = get_conc(drug_a, t)
            conc_b = get_conc(drug_b, t)

            offsetspace = np.linspace(0, 1, 13)[:-1]
            best_offsets = [0.0, 0.0, 0.0]#{'as': 0.0, 'ax': 0.0}
            best_effect = [999, 999, 999]


            for b_offset in offsetspace:
                drug_b['offset'] = b_offset
                normalize_effect_single(drug_a, 0.1, m=m)
                normalize_effect_single(drug_b, 0.95, m=m)
                conc_a = get_conc(drug_a, t)
                conc_b = get_conc(drug_b, t)
                # vfte_m = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m)
                # vfte_w = 1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m)

                # dvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) - \
                #        1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m)
                # pl = 1/(1 + (conc_a/cell2[0])**m) - 1/(1 + (conc_a/cell1[0])**m)

                rvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) / \
                       (1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m))
                rpl = 1/(1 + (conc_a/cell2[0])**m) / (1/(1 + (conc_a/cell1[0])**m))
                # print(drug_a, cell2, cell1)

                if (opt_targe == 'xauc'):
                    auc = simps(rvel, x=t)
                elif (opt_targe == 'min'):
                    auc = min(rvel)

                if auc < best_effect[k]:
                    best_offsets[k] = b_offset
                    best_effect[k] = auc

            print(best_offsets)

            drug_b['offset'] = best_offsets[k]

            normalize_effect_single(drug_a, 0.1, m=m)
            normalize_effect_single(drug_b, 0.95, m=m)

            conc_a = get_conc(drug_a, t)
            conc_b = get_conc(drug_b, t)

            vfte_m = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m)
            vfte_w = 1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m)

            dvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) - \
                    1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m)
            pl = 1/(1 + (conc_a/cell2[0])**m) - 1/(1 + (conc_a/cell1[0])**m)

            rvel = 1/(1 + (conc_a/cell2[0] + conc_b/cell2[1])**m) / \
                    (1/(1 + (conc_a/cell1[0] + conc_b/cell1[1])**m))
            rpl = 1/(1 + (conc_a/cell2[0])**m) / (1/(1 + (conc_a/cell1[0])**m))

            axs.plot(t, rvel, linewidth=1.0, color=PALETTE_4[i], label=drug_a['name'])
            axs.plot(t, rpl, linewidth=1.0, linestyle='--', color=PALETTE_4[i])
            axs.axvline(drug_b['offset'], color='grey', linewidth=1.0, linestyle=':')

            yoff = 0
            xoff = best_offsets[k]
            if xoff in occupied:
                yoff += 0.04*occupied[xoff]
                occupied[xoff] += 1
            else:
                occupied[xoff] = 1
            axs.text(xoff, 1.03 + yoff, 'AX', transform=axs.transAxes, size=8,
                     color=PALETTE_4[i],
                     verticalalignment='center', horizontalalignment='center')

            axs.set_xlim(0, drug_a['tau']*drug_a['repeats'])
            axs.set_yscale('log')
            axs.set_xlabel('Time [h]')
            axs.set_ylabel(r'$\chi f_v$')

            wt_area = plt.Polygon([(0, 0.3), (1, 0.3), (1, 1), (0, 1)], color='lightgrey')
            axs.add_patch(wt_area)

            axs.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            print(axs.get_xticks())
            axs.set_xticklabels([int(24*x) for x in axs.get_xticks()])

        axs.set_ylim(0.3, 10**np.ceil(np.log10(axs.get_ylim()[1])))
        axs.text(0.5, 0.05, 'WT grows faster', color='w',
                 verticalalignment='center', horizontalalignment='center',
                 transform=axs.transAxes)
        axs.text(0.5, 0.95, 'T315I grows faster', color='lightgrey',
                 verticalalignment='center', horizontalalignment='center',
                 transform=axs.transAxes)

        full_lines = mlines.Line2D([], [], color='k', linestyle='-', linewidth=1.0)
        dashed_lines = mlines.Line2D([], [], color='k', linestyle='--', linewidth=1.0)
        # markers = [mlines.Line2D([], [], color=color, linestyle='-', linewidth=1.0) for color in PALETTE_4]
        # markers = [mlines.Line2D([], [], color=color, linestyle='none', marker='.', markersize=15.0, linewidth=1.0) for color in PALETTE_4]
        markers = [mlines.Line2D([], [], color=color, linestyle='none', marker='.', markersize=15.0, linewidth=1.0) for color in PALETTE_4[:len(adrugs)]]

        pos11 = axs.get_position()
        print(pos11)
        symbols = markers + [full_lines, dashed_lines]
        labels = names + ['Combination\nwith ' + drug_b['name'], 'Monodrug']
        # labels = ['Imatinib', 'Nilotinib', 'Dasatinib', 'Bosutinib'] + ['Combination\nwith Axitinib', 'Monodrug']
        print(symbols, labels)
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
