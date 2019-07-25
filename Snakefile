configfile:
    "config.yml"


rule all:
    input:
        "intermediate/imatinib-axitinib-e-min.pdf",
        "intermediate/nilotinib-axitinib-e-min.pdf",
        "intermediate/dasatinib-axitinib-e-min.pdf",
        "intermediate/bosutinib-axitinib-e-min.pdf",
        "intermediate/imatinib-axitinib-e-xauc.pdf",
        "intermediate/nilotinib-axitinib-e-xauc.pdf",
        "intermediate/dasatinib-axitinib-e-xauc.pdf",
        "intermediate/bosutinib-axitinib-e-xauc.pdf",
        "intermediate/alltrace-e-min.pdf",
        "intermediate/alltrace-e-xauc.pdf",


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


rule generate_figureset:
    input:
        "intermediate/{drug_a}.toml",
        "intermediate/{drug_b}.toml"
    output:
        "intermediate/{drug_a}-{drug_b}-{interaction}-{mode}.pdf"
    params:
        mutant = config["trace_mutant"],
        dataset = config["axitinib_dataset"]
    shell:
        """
        python3 code/diagnostics.py optimize_offset \
            {input} \
            {params.mutant} \
            {params.dataset} \
            -m {wildcards.interaction} \
            -t {wildcards.mode}
        python3 code/diagnostics.py make \
            {input} \
            {params.mutant} \
            {params.dataset} \
            -o intermediate/{wildcards.drug_a}-{wildcards.drug_b}-{wildcards.interaction}-{wildcards.mode}
        """

        
rule generate_alltrace:
        # if you've run generate_figureset for these drugs before, the requisite inputs are already present
    output:
        "intermediate/alltrace-{interaction}-{mode}.pdf"
    params:
        mutant = config["trace_mutant"],
        dataset = config["axitinib_dataset"],
        adrugs = config["alltrace_druga"],
        bdrug = config["alltrace_drugb"]
    run:
        mstr = ' '
        for m in params.adrugs:
            mstr += ' -a' + m
        shell(" \
            python3 code/diagnostics.py alltrace \
                {mstr} \
                -b {params.bdrug} \
                --mutant {params.mutant} \
                --dataset {params.dataset} \
                -m {wildcards.interaction} \
                -t {wildcards.mode} \
                --save {output} \
        ")