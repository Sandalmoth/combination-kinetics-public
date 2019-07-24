"""
preprocessing adding data from ic50 compilation to drug toml files
"""


import csv

import click
import toml


@click.group()
def main():
    """
    click construct
    """
    pass


@main.command()
@click.option('-d', '--drugfile', type=click.Path())
@click.option('-c', '--csvfile', type=click.Path())
@click.option('-o', '--outfile', type=click.Path())
@click.option('-m', 'mutants', type=str, multiple=True)
def add_data(drugfile, csvfile, outfile, mutants):
    """
    add information about mutation ic50s from csv csvfile to toml drugfile
    saves it as outfile
    """

    ic_data = {}
    with open(csvfile, 'r') as in_csv:
        rdr = csv.DictReader(in_csv)
        for row in rdr:
            if row['drug'] not in ic_data:
                ic_data[row['drug']] = {}
            ic_data[row['drug']][row['mutation']] = (row['ic50-prefix'], row['ic50'])

    # print(ic_data)

    drug_data = toml.load(drugfile)
    drug = drug_data['name'].lower()
    print('Adding data for', drug)
    print('Available mutations are', ic_data[drug].keys())
    for mutant in mutants:
        if mutant not in ic_data[drug]:
            print('WARNING: Data for', mutant, 'not available. Skipping')
            continue
        if ic_data[drug][mutant][0] != '':
            print(drug, mutant, 'has prefix')
        drug_data[mutant] = [float(ic_data[drug][mutant][1])]

    drug_data['native'] = [1.0]

    with open(outfile, 'w') as out_toml:
        toml.dump(drug_data, out_toml)



if __name__ == "__main__":
    main()
