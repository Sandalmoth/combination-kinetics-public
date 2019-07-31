"""
Diagnostic plots for examining how big of a dose reduction that becomes
possible from running two drugs in parallel.
"""


import csv
from copy import deepcopy

import click
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
from scipy.integrate import simps
from scipy.optimize import minimize
import toml


# there is some code repetition below, as the data-generating 'make' and table producing 'bigtable'
# were written separately, and function interfaces differed slightly


@click.group()
def main():
    """
    main construct for click to function in self-contained file
    """
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


def get_effect_single_const(drug, dose=None, ic=1, m=1):
    """
    Get effect (i.e. f_v) for a single drug without pharmacokinecits
    I.e. assuming constant concentration
    It takes into account the possibility of less than daily dosing
    """
    if dose:
        drug['D'] = dose
    conc = drug['D']
    return 1/(1 + (conc/ic)**m)


def get_effect_const(drug1, drug2, method, dose1=None, dose2=None, ic1=1, ic2=1, m=1):
    """
    Get the effect (i.e. auc) given a pair of drugs
    """
    assert drug1['tau']*drug1['repeats'] == drug2['tau']*drug2['repeats']
    if dose1:
        drug1['D'] = dose1
    if dose2:
        drug2['D'] = dose2
    conc1 = drug1['D']
    conc2 = drug2['D']
    if method == 'e':
        dve = 1/(1 + (conc1/ic1 + conc2/ic2)**m)
    elif method == 'n':
        dve = 1/(1 + (conc1/ic1 + conc2/ic2 + conc2*conc1/(ic1*ic2))**m)
    else:
        print("invalid method in get_effect")
        exit(0)
    return dve


def normalize_effect_single_const(drug, effect_target, m=1, xtol=1e-9):
    """
    Set the drug dose such that the effect calculated by get_effect_single
    equals effect_target
    In principle, two drugs normalized like this will be equally effective
    """
    res = minimize(lambda x: abs(effect_target - get_effect_single_const(drug, x[0], m=m)),
                   0.1, method='nelder-mead', options={'xtol': xtol, 'disp': False})
    drug['D'] = res.x[0]


def normalize_effect_dual_const(drug1, drug2, method, effect_target, ratio, ic1=1, ic2=1, m=1, xtol=1e-9):
    """
    Set the drug dose such that the effect calculated by get_effect
    equals effect_target with a given drug ratio
    """
    res = minimize(lambda x: abs(effect_target - get_effect_const(drug1, drug2, method,
                                                            x[0]*ratio, x[0]*(1 - ratio), m=m,
                                                            ic1=ic1, ic2=ic2)),
                   0.1, method='nelder-mead', options={'xtol': xtol, 'disp': False})
    drug1['D'] = res.x[0]*ratio
    drug2['D'] = res.x[0]*(1 - ratio)


def get_effect2(drug1, drug2, method, dose1=None, dose2=None, ic1=1, ic2=1, m=1, resolution=1000):
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



def get_effect(drug1, drug2, method, ic1=1, ic2=1, resolution=1000, m=1):
    """
    Get the area under curve of the growth rate curve defined by a certain
    drugs pharmacokinetic profile
    """
    assert drug1['tau']*drug1['repeats'] == drug2['tau']*drug2['repeats']
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



def normalize_effect(drug, effect_target, xtol=1e-9, m=1):
    """
    Set the drug dose such that the effect calculated by get_effect_single
    equals effect_target
    In principle, two drugs normalized like this will be equally effective
    """
    res = minimize(lambda x: abs(effect_target - get_effect_single(drug, x[0], m=m)),
                   1.0, method='nelder-mead', options={'xtol': xtol, 'disp': False})
    drug['D'] = res.x[0]


def normalize_effect_dual(drug1, drug2, method, effect_target, ratio, ic1=1, ic2=1, m=1, xtol=1e-9):
    """
    Set the drug dose such that the effect calculated by get_effect
    equals effect_target with a given drug ratio
    """
    res = minimize(lambda x: abs(effect_target - get_effect2(drug1, drug2, method,
                                                            x[0]*ratio, x[0]*(1 - ratio), m=m,
                                                            ic1=ic1, ic2=ic2)),
                   0.1, method='nelder-mead', options={'xtol': xtol, 'disp': False})
    drug1['D'] = res.x[0]*ratio
    drug2['D'] = res.x[0]*(1 - ratio)



def optimize_reduction(drug1, drug2, method, effect_target, ratio_resolution, phase_resolution, m=1):
    """
    find how much drug doses can be lowered at a range of drug ratios
    """

    period = drug1['tau']*drug1['repeats']

    ratio = np.linspace(0, 1, ratio_resolution)

    drug1['offset'] = 0

    dsums = np.empty((ratio_resolution, phase_resolution))

    for i, dose1 in enumerate(ratio):
        dose2 = 1 - dose1

        # Since fractional absorption isn't used anyway
        # use it to hold dose ratio reduction
        drug1['F'] = dose1
        drug2['F'] = dose2

        for k, phase in enumerate(np.arange(0, 1, 1/phase_resolution)):

            drug2['offset'] = phase*period

            def _effect(var):
                drug1['S'] = var[0]
                drug2['S'] = var[0]
                return abs(effect_target - get_effect(drug1, drug2, method, m=m))

            res = minimize(_effect, 0.5, bounds=[(0, 10)],
                           method='L-BFGS-B', options={'disp': None})

            # salt fraction isn't used either
            # use it to store reduction
            drug1['S'] = res.x[0]
            drug2['S'] = res.x[0]

            print(drug1['S'], drug1['F'], drug2['S'], drug2['F'])

            dsums[i][k] = drug1['S']*drug1['F'] + drug2['S']*drug2['F']

    return ratio, dsums



@main.command()
@click.argument('adrug', type=click.Path())
@click.argument('bdrug', type=click.Path())
@click.argument('dataset', type=int)
@click.option('-m', 'mutants', type=str, multiple=True)
@click.option('-o', '--outfile', type=click.Path(), default=None)
@click.option('-c', '--csvfile', type=click.Path(), default=None)
def make(adrug, bdrug, dataset, mutants, outfile, csvfile):
    """
    create diagnostic plots for given drug parameter files
    adrug/bdrug - toml files of drug parameters
    dataset - integer indicating which IC50 dataset from the toml files to use
    mutants - list of mutants to use in comparison table
    outfile - if present save all plots as pdf to this file
    """

    effect_target = 0.1

    # optimal ratio and phase plots

    if outfile is not None:
        # first time, construct the multipage pdf
        pdf_out = PdfPages(outfile)

    # for each value of the hill coefficent m:
    #   find the dose that creates a given effect target
    #   optimized for adminstration offset and drug ratio
    for m in [0.5, 1, 2]:

        drug_a = toml.load(adrug)
        drug_b = toml.load(bdrug)

        normalize_effect(drug_a, effect_target, m=m)
        normalize_effect(drug_b, effect_target, m=m)

        fig, axs = plt.subplots(ncols=3)
        fig.set_size_inches(8.5, 2.4)

        ratio_resolution = 21
        phase_resolution = 24

        phase = np.linspace(0, 1, phase_resolution + 1)
        ratio, dsum = optimize_reduction(drug_a, drug_b, 'e', effect_target,
                                         ratio_resolution, phase_resolution, m=m)
        color = (2/3, 2/3, 2/3)
        axs[0].plot(ratio, np.min(dsum, axis=1), color=color)
        axs[0].plot(ratio, dsum[:, 0], color=color, linestyle='--')
        axs[1].scatter(ratio, np.argmin(dsum, axis=1)/dsum.shape[1],
                      color=color, marker='o', label='Exclusive')
        phase_effect = dsum[np.where(dsum == dsum.min())[0][0], :]
        axs[2].plot(phase, np.append(phase_effect, phase_effect[0]), color=color)

        ratio, dsum = optimize_reduction(drug_a, drug_b, 'n', effect_target,
                                         ratio_resolution, phase_resolution, m=m)
        color = (0, 0, 0)

        axs[0].plot(ratio, np.min(dsum, axis=1), color=color)
        axs[0].plot(ratio, dsum[:, 0], color=color, linestyle='--')
        axs[1].scatter(ratio, np.argmin(dsum, axis=1)/dsum.shape[1],
                      marker='.', color=color, label='Nonexclusive')
        phase_effect = dsum[np.where(dsum == dsum.min())[0][0], :]
        axs[2].plot(phase, np.append(phase_effect, phase_effect[0]), color=color)

        print(adrug, bdrug, 'mimimum m =', m, np.min(np.min(dsum, axis=1)))

        axs[0].set_ylabel('Dose reduction\nfactor')
        axs[0].set_xlabel('Drug ratio')
        axs[0].set_xticks([0, 0.5, 1])
        axs[0].set_xticklabels(['100% ' + drug_a['name'], '', '100% ' + drug_b['name']])

        axs[1].set_ylabel('Best ' + drug_b['name'] + '\nadministration delay')
        axs[1].set_xlabel('Drug ratio')
        axs[1].set_xticks([0, 0.5, 1])
        axs[1].set_xticklabels(['100% ' + drug_a['name'], '', '100% ' + drug_b['name']])
        axs[1].set_ylim(-.05, 1.05)
        axs[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[1].set_yticklabels([int(24*x) for x in axs[1].get_yticks()])

        axs[2].set_xlabel(drug_b['name'] + ' administration delay')
        axs[2].set_ylabel('Dose reduction\nfactor of best ratio')
        axs[2].set_xlim(-.05, 1.05)
        axs[2].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[2].set_xticklabels([int(24*x) for x in axs[2].get_xticks()])

        plt.tight_layout()

        if outfile is not None:
            pdf_out.savefig()
            plt.savefig(outfile[:-3] + str(m) + '.svg', format='svg', dpi=4000)
        else:
            plt.show()

    # Mutation effect at reduction table
    drug_a = toml.load(adrug)
    drug_b = toml.load(bdrug)

    normalize_effect(drug_a, effect_target)
    normalize_effect(drug_b, effect_target)

    ic_table = np.empty((len(mutants), 2))

    ratios_best = [-1, -1]

    for method_num, method in enumerate(['e', 'n']):

        ratio, dsum = optimize_reduction(drug_a, drug_b, 'n', effect_target,
                                         ratio_resolution, phase_resolution)

        best_ratio = ratio[np.where(dsum == dsum.min())[0][0]]
        print(best_ratio)

        ratios_best[method_num] = best_ratio

        for mutant_num, mutant in enumerate(mutants):
            print(mutant)
            ic_relative_a = drug_a[mutant][dataset] / drug_a['native'][dataset]
            ic_relative_b = drug_b[mutant][dataset] / drug_b['native'][dataset]
            print(ic_relative_a, ic_relative_b)
            if method == 'n':
                # solve
                # 1 = cx/a + c(1-x)/b + c²x(1-x)/(a*b)
                # for c to find the ic50 given this drug ratio
                # c = (±√(a²(x - 1)² - 6ab(x² - x) + b²x²) - ax + a + bx)/(2(x² - x))
                # redefinitions to reduce verbosity
                a = ic_relative_a
                b = ic_relative_b
                x = best_ratio

                z = x**2 - x
                s = np.sqrt(a**2 *(x - 1)**2 - 6*a*b*z + b**2 *x**2)

                ic_effective_1 = (s - a*x + a + b*x) / (2*z)
                ic_effective_2 = (-s - a*x + a + b*x) / (2*z)
                if ic_effective_1 < 0:
                    if ic_effective_2 < 0:
                        print('Impossible effective IC50 values for combination')
                        print(ic_effective_1, ic_effective_2)
                        exit(0)
                    else:
                        ic_relative = ic_effective_2
                else:
                    if ic_effective_2 > 0:
                        print('Cannot discriminate correct and incorrect combination IC50 values')
                        print(ic_effective_1, ic_effective_2)
                        exit(0)
                    ic_relative = ic_effective_1
                print(ic_relative)

                # we also need to know how effective the combination is against the wildtype
                # 1 = cx + c(1-x) + c²x(1-x)
                # c = (1 ± √(-4x² + 4x + 1)) / 2(x² - x)
                #   = (±√((ax - a - bx)² - 4ab(x² - x)) - ax + a + bx) / (2(x² - x))
                #   = (±√(1 - 4(x² - x)) - x + 1 + x) / (2(x² - x))
                #   = (±√(1 - 4x² + 4x) + 1) / (2(x² - x))
                z = x**2 - x
                ic_wt_1 = (1 - np.sqrt(-4*x**2 + 4*x + 1)) / (2*z)
                ic_wt_2 = (1 + np.sqrt(-4*x**2 + 4*x + 1)) / (2*z)
                print('icwt', ic_wt_1, ic_wt_2)
                if ic_wt_1 < 0:
                    if ic_wt_2 < 0:
                        print('Impossible wt IC50 values for combination')
                        print(ic_wt_1, ic_wt_2)
                        exit(0)
                    else:
                        ic_wt = ic_wt_2
                else:
                    if ic_wt_2 > 0:
                        print('Cannot discriminate correct and incorrect combination IC50 values')
                        print(ic_wt_1, ic_wt_2)
                        exit(0)
                    ic_wt = ic_wt_1
                print(ic_wt)

            elif method == 'e':
                # solve
                # 1 = cx/a + c(1-x)/b
                # c = 1 / (x/a + (1-x)/b)
                a = ic_relative_a
                b = ic_relative_b
                x = best_ratio

                ic_relative = 1 / (x/a + (1 - x)/b)
                print(ic_relative)

                # also find
                # 1 = cx + c(1-x)
                # c = 1
                # that was easy
                ic_wt = 1
                print(ic_wt)

            ic_table[mutant_num][method_num] = ic_relative/ic_wt

    print(ic_table)
    print(ic_table.shape)

    # save calculated ic50s in a table for later use
    if csvfile is not None:
        fieldnames = ['drug_a', 'drug_b', 'ratio', 'mutant', 'method', 'ic50']
        with open(csvfile, 'a') as csv_handle:
            wtr = csv.DictWriter(csv_handle, fieldnames=fieldnames)
            for mutant, line in zip(mutants, ic_table):
                for i, method in enumerate(['exclusive', 'nonexclusive']):
                    ld = {
                        'drug_a': drug_a['name'],
                        'drug_b': drug_b['name'],
                        'ratio': ratios_best[i],
                        'mutant': mutant,
                        'method': method,
                        'ic50': line[i]
                    }
                    wtr.writerow(ld)

    fig, axs = plt.subplots()
    fig.set_size_inches(3, len(mutants)/2)
    axs.imshow(ic_table, interpolation='nearest', vmin=0,
               extent=(0.05, 5.95, 0.05, len(mutants) - 0.05))
               # extent=(0.05, 5.95, 0.05, len(mutants) - 0.05), cmap=cc.m_isolum)
    axs.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs.set_xlim(0, 6)
    axs.set_ylim(0, len(mutants))
    axs.set_xticks([1.5, 4.5])
    axs.set_xticklabels(['Exclusive', 'Nonexclusive'])
    axs.set_yticks(-np.arange(0, len(mutants)) + len(mutants) - 0.5)
    axs.set_yticklabels(mutants)
    for j in range(2):
        for i, ic50 in enumerate(ic_table[:, j]):
            axs.text(1.5 + 3*j, len(mutants)-i-0.5, str(np.round(ic50, 2)),
                     horizontalalignment='center', verticalalignment='center', color='k')

    for __, spine in axs.spines.items():
        spine.set_visible(False)
    axs.set_xticks([0, 3, 6], minor=True)
    axs.set_yticks(np.arange(len(mutants)+1), minor=True)
    axs.grid(which="minor", color="w", linestyle='-', linewidth=4)
    axs.tick_params(which="major", bottom=False, top=False, left=False)
    axs.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    if outfile is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # done with plots, close the pdf
    if outfile is not None:
        pdf_out.close()






@main.command()
@click.option('-t', '--tablefile', type=click.Path())
@click.option('-o', '--outfile', type=click.Path(), default=None)
# @click.option('-c', '--csvfile', type=click.Path(), default=None)
def bigtable(tablefile, outfile):
    """
    print a single table for all drugs using csv from make (above)
    modes - e exclusive only
            n nonexclusive only
    """

    print(tablefile)
    print(outfile)

    csvfile = tablefile
    # csvfile = 'intermediate/ic50.csv'
    mode = 'n'

    opt_target = 0.1

    # first get some bulk data from the file
    n_mutants = 0
    n_combos = 0

    names_adrug = []
    names_bdrug = []

    combo_names = []
    mutant_names = []
    combo_nicenames = []

    combo_ratios = []

    with open(csvfile, 'r') as csv_handle:

        combos = []
        nicenames = []
        mutants = []
        ratios = []

        fieldnames = ['drug_a', 'drug_b', 'ratio', 'mutant', 'method', 'ic50']
        # rdr = csv.DictReader(csv_handle, fieldnames=fieldnames)
        rdr = csv.DictReader(csv_handle)
        for line in rdr:
            if (mode == 'e' and line['method'] == 'nonexclusive') or \
               (mode == 'n' and line['method'] == 'exclusive'):
                continue
            mutants.append(line['mutant'])
            combos.append(line['drug_a'] + line['drug_b'])
            names_adrug.append(line['drug_a'])
            names_bdrug.append(line['drug_b'])
            print(line['ratio'])
            nicenames.append(line['drug_a'][:2].upper() + '-' + line['drug_b'][:2].upper() + \
                             r", x = " + str(np.round(float(line['ratio']), 2)))
            ratios.append(np.round(float(line['ratio']), 2))

        n_combos = len(set(combos))
        n_mutants = len(set(mutants))
        # combo_names = combos[::n_mutants*2]
        combo_names = sorted(list(set(combos)))
        combo_nicenames = [nicenames[combos.index(x)] for x in combo_names]
        combo_ratios = [ratios[combos.index(x)] for x in combo_names]
        mutant_names = mutants[:n_mutants]
        print(combo_names)
        print(mutant_names)
        # mutant_names = sorted(list(set(mutants)))

    print(n_combos, n_mutants)

    ic_table = np.empty((n_mutants, n_combos))

    # now parse to get data we need
    with open(csvfile, 'r') as csv_handle:
        fieldnames = ['drug_a', 'drug_b', 'ratio', 'mutant', 'method', 'ic50']
        # rdr = csv.DictReader(csv_handle, fieldnames=fieldnames)
        rdr = csv.DictReader(csv_handle)
        for line in rdr:
            if (mode == 'e' and line['method'] == 'nonexclusive') or \
               (mode == 'n' and line['method'] == 'exclusive'):
                continue
            print(line['mutant'])
            print(ic_table[mutant_names.index(line['mutant'])])
            print(combo_names.index(line['drug_a'] + line['drug_b']))
            print(combo_names)
            print(line['drug_a'] + line['drug_b'])
            ic_table[mutant_names.index(line['mutant'])] \
                    [combo_names.index(line['drug_a'] + line['drug_b'])] = line['ic50']


    print(combo_nicenames)
    print(combo_ratios)
    print(ic_table)
    print(mutant_names)
    print(combo_names)
    names_adrug = sorted(list(set(names_adrug)))
    names_bdrug = sorted(list(set(names_bdrug)))
    print(names_adrug, names_bdrug)

    if len(names_adrug) > 1:
        names_adrug, names_bdrug = names_bdrug, names_adrug

    adrug_filename = 'intermediate/' + names_adrug[0].lower() + '.toml'
    print(adrug_filename)

    bdrug_filenames = ['intermediate/' + x.lower() + '.toml' for x in names_bdrug]
    print(bdrug_filenames)




    shape_table = deepcopy(ic_table)

    # calculate the non-linear effect for each position in the table
    # basically for each drug combination:
    #   normalize effect of each drug separately to given effect target
    #   calculate effectiveness against mutant at that drug
    #   normalize effect of combination to given effect target
    #   calculate effectiveness against mutant of combination
    #   if effect of combination is worse (f_v is higher) than for either drugs alone
    #     save in table and mark with red box in figure
    # adrug = 'intermediate/asciminib.toml'
    adrug = adrug_filename
    # for i, bdrug in enumerate(['intermediate/bosutinib.toml', 'intermediate/dasatinib.toml', 'intermediate/imatinib.toml', 'intermediate/nilotinib.toml', 'intermediate/ponatinib.toml']):
    for i, bdrug in enumerate(bdrug_filenames):
        drug_a = toml.load(adrug)
        drug_b = toml.load(bdrug)

        print(drug_a['name'], drug_b['name'])

        m_nonlin = 2

        # normalize_effect_single(drug_a, opt_target, m=m_nonlin)
        # normalize_effect_single(drug_b, opt_target, m=m_nonlin)
        normalize_effect_single_const(drug_a, opt_target, m=m_nonlin)
        normalize_effect_single_const(drug_b, opt_target, m=m_nonlin)

        print(get_effect_single(drug_a, m=m_nonlin), get_effect_single(drug_b, m=m_nonlin))
        print(get_effect_single_const(drug_a, m=m_nonlin), get_effect_single_const(drug_a, m=m_nonlin))

        single_effects = []

        print('\tmutant\ta eff.\tb eff.\ta ic50\tbic50\tsimil\tscore')
        for mutant in mutant_names:
            # e1 = get_effect_single(drug_a, ic=drug_a[mutant][0], m=m_nonlin)
            # e2 = get_effect_single(drug_b, ic=drug_b[mutant][0], m=m_nonlin)
            e1 = get_effect_single_const(drug_a, ic=drug_a[mutant][0], m=m_nonlin)
            e2 = get_effect_single_const(drug_b, ic=drug_b[mutant][0], m=m_nonlin)
            print('s', mutant, round(e1, 2), round(e2, 2), round(drug_a[mutant][0], 2), round(drug_b[mutant][0], 2), round(max(drug_a[mutant][0], drug_b[mutant][0])/min(drug_a[mutant][0], drug_b[mutant][0]), 2), round((drug_a[mutant][0] + drug_b[mutant][0])/max(drug_a[mutant][0], drug_b[mutant][0])*min(drug_a[mutant][0], drug_b[mutant][0]), 2), sep='\t')
            single_effects.append(max(e1, e2))

        # normalize_effect_dual(drug_a, drug_b, mode, opt_target, 1 - combo_ratios[i], m=m_nonlin)
        normalize_effect_dual_const(drug_a, drug_b, mode, opt_target, 1 - combo_ratios[i], m=m_nonlin)

        print(get_effect2(drug_a, drug_b, mode, m=m_nonlin))
        print(get_effect_const(drug_a, drug_b, mode, m=m_nonlin))

        dual_effects = []

        # e1 = get_effect2(drug_a, drug_b, mode, m=m_nonlin)
        e1 = get_effect_const(drug_a, drug_b, mode, m=m_nonlin)
        for mutant in mutant_names:
            # e2 = get_effect2(drug_a, drug_b, mode, ic1=drug_a[mutant][0], ic2=drug_b[mutant][0], m=m_nonlin)
            e2 = get_effect_const(drug_a, drug_b, mode, ic1=drug_a[mutant][0], ic2=drug_b[mutant][0], m=m_nonlin)
            # print('d', mutant, e1, e2, drug_a[mutant], drug_b[mutant])
            print('d', mutant, round(e1, 2), round(e2, 2), round(drug_a[mutant][0], 2), round(drug_b[mutant][0], 2), round(max(drug_a[mutant][0], drug_b[mutant][0])/min(drug_a[mutant][0], drug_b[mutant][0]), 2), round((drug_a[mutant][0] + drug_b[mutant][0])/max(drug_a[mutant][0], drug_b[mutant][0])*min(drug_a[mutant][0], drug_b[mutant][0]), 2), sep='\t')
            dual_effects.append(e2)

        for mutant, es, ed in zip(mutant_names, single_effects, dual_effects):
            print(mutant, es, ed)

        for j in range(len(mutant_names)):
            shape_table[j][i] = 1.0 if single_effects[j] > dual_effects[j] else dual_effects[j]/single_effects[j]

    print(ic_table)
    print(shape_table)
    print((shape_table > 1) * 1.0)

    if outfile is not None:
        # first time, construct the multipage pdf
        pdf_out = PdfPages(outfile)

    fig, axs = plt.subplots()
    fig.set_size_inches(n_combos*1.5, n_mutants/2)
    im = axs.imshow(-ic_table, interpolation='nearest', vmax=0,
               extent=(0.05, n_combos*3 - 0.05, 0.05, n_mutants - 0.05), cmap=cm.pink)
    axs.tick_params(top=False, bottom=False, left=False, right=False,
                    labeltop=True, labelbottom=False)
    axs.set_xlim(0, n_combos*3)
    axs.set_ylim(0, len(mutant_names))
    axs.set_xticks(np.arange(n_combos)*3 + 1.5)
    combo_nicenames = [x.replace(', ', '\n') for x in combo_nicenames]
    print(combo_nicenames)
    axs.set_xticklabels(combo_nicenames)
    axs.set_yticks(-np.arange(0, len(mutant_names)) + len(mutant_names) - 0.5)
    axs.set_yticklabels(mutant_names)
    for j in range(n_combos):
        for i, (ic50, shape) in enumerate(zip(ic_table[:, j], shape_table[:, j])):
            axs.text(1.5 + 3*j, n_mutants-i-0.5, str(np.round(ic50, 2)),
                     horizontalalignment='center', verticalalignment='center',
                     color='k' if ic50 < 0.5*np.max(ic_table) else 'w')
            if shape != 1.0:
                rect = patches.Rectangle((0.1 + 3*j, n_mutants-i-0.1), 2.8, -0.8,
                                         linewidth=3, edgecolor='r', facecolor='none')
                axs.add_patch(rect)

    for __, spine in axs.spines.items():
        spine.set_visible(False)
    axs.set_xticks(np.arange(n_combos+1)*3, minor=True)
    axs.set_yticks(np.arange(n_mutants+1), minor=True)
    axs.grid(which="minor", color="w", linestyle='-', linewidth=4)
    axs.tick_params(which="minor", bottom=False, left=False)

    # plt.tight_layout()
    cb = fig.colorbar(im, ax=axs, orientation='horizontal',
                      fraction=0.042, pad=0.04)
    # cb.ax.get_xaxis().labelpad = 15
    cb.set_label(r'$\widetilde{\mathrm{IC}}_{50}^{\mathrm{eff}}$', labelpad=-35, x=-0.06)
    cb.ax.invert_xaxis()
    cbticks = [0, 5, 10, 15, 20, 25]
    cb.set_ticks([-x for x in cbticks])
    cb.set_ticklabels(cbticks)

    rect = patches.Rectangle((-1.5, -3.2), 1.5, 0.7,
                             linewidth=2, edgecolor='r', facecolor='none', clip_on=False)
    axs.add_patch(rect)
    axs.text(6.5, -2.85, "More sensitive than indicated due to dose-effect curve shape",
             horizontalalignment='center', verticalalignment='center', clip_on=False)

    if outfile is not None:
        pdf_out.savefig()
    else:
        plt.show()

    # done with plots, close the pdf
    if outfile is not None:
        pdf_out.close()




if __name__ == '__main__':
    main()
