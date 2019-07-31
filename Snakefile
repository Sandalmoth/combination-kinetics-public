configfile:
    "config.yml"


rule all:
    input:
        "results/figures/imatinib-axitinib-e-min.pdf",
        "results/figures/nilotinib-axitinib-e-min.pdf",
        "results/figures/dasatinib-axitinib-e-min.pdf",
        "results/figures/bosutinib-axitinib-e-min.pdf",
        "results/figures/imatinib-axitinib-e-xauc.pdf",
        "results/figures/nilotinib-axitinib-e-xauc.pdf",
        "results/figures/dasatinib-axitinib-e-xauc.pdf",
        "results/figures/bosutinib-axitinib-e-xauc.pdf",
        "results/figures/alltrace-e-min.pdf",
        "results/figures/alltrace-e-xauc.pdf",
        "results/figures/imatinib-asciminib-ic50.pdf",
        "results/figures/nilotinib-asciminib-ic50.pdf",
        "results/figures/dasatinib-asciminib-ic50.pdf",
        "results/figures/bosutinib-asciminib-ic50.pdf",
        "results/figures/ponatinib-asciminib-ic50.pdf",
        "results/figures/ic50-combos.pdf",
        "results/figures/as-combos-min.pdf",
        "results/figures/as-combos-xauc.pdf"


rule make_empty_table:
    params:
        table_path = config["ic50_table"]
    output:
        config["ic50_table"]
    shell:
        """
        echo 'drug_a,drug_b,ratio,mutant,method,ic50' > {params.table_path}
        """


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
        "results/figures/{drug_a}-{drug_b}-{interaction}-{mode}.pdf"
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
            -o results/figures/{wildcards.drug_a}-{wildcards.drug_b}-{wildcards.interaction}-{wildcards.mode}
        """

        
rule generate_alltrace:
        # if you've run generate_figureset for these drugs before, the requisite inputs are already present
    output:
        "results/figures/alltrace-{interaction}-{mode}.pdf"
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


rule generate_ic50:
    input:
        "intermediate/{drug_a}.toml",
        "intermediate/{drug_b}.toml",
        config["ic50_table"]
    output:
        "results/figures/{drug_a}-{drug_b}-ic50.pdf"
    params:
        mutants = config["ic50_mutants"],
        dataset = config["asciminib_dataset"],
        table_path = config["ic50_table"]
    run:
        mstr = ' '
        for m in params.mutants:
            mstr += ' -m' + m
        istr = ' '.join(input[:2])
        print(istr)
        shell(" \
            python3 code/reduction.py make \
                {istr} \
                {params.dataset} \
                {mstr} \
                -o {output} \
                -c {params.table_path} \
        ")


rule big_table:
    output:
        "results/figures/ic50-combos.pdf"
    params:
        table_path = config["ic50_table"]
    shell:
        """
        python3 code/reduction.py bigtable \
            -t {params.table_path} \
            -o {output}
        """


rule triple:
    input:
        "intermediate/bosutinib.toml",
        "intermediate/asciminib.toml",
        "intermediate/axitinib.toml"
    output:
        "results/figures/as-combos-{target}.pdf"
    shell:
        """
        python3 code/triple.py as_combos \
            --save {output} \
            -t {wildcards.target}
        """