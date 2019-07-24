configfile:
    "config.yml"


# add data from the ic50 csv to tomlfiles with drug pharmacokinetics parameters
rule add_ic50_data:
    input:
        drugs = "data/raw_internal/{drug}.toml",
        ic50s = "data/raw_internal/ic50-compilation.csv"
    output:
        "intermediate/{drug}.toml"
    params:
        mutants = config["ic50_mutants"],
    run:
        mstr = ' '
        for m in params.mutants:
            mstr += ' -m' + m
        shell(" \
            python3 code/add-compilation.py add_data \
                -d {input.drugs} \
                -c {input.ic50s} \
                {mstr} \
                -o {output} \
        ")
        