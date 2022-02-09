rule all:
    input:
        "{translate}.pair_epistasis.csv"

rule process_epistasis:
    input:
        "{translate}.translated.tsv",
        "{translate}.node_clades.txt",
        "{translate}.sample_paths.txt"
    output:
        "{translate}.pair_epistasis.csv",
        "{translate}.anchored_conservation.csv"
    shell:
        "python3 process_epistasis.py -t {input[0]} -c {input[1]} -p {input[2]} -pa {output[0]} -aa {output[1]}"

rule write_translation:
    input:
        "NC_045512v2.fa",
        "ncbiGenes.gtf",
        "{translate}.masked.pb"
    output:
        "{translate}.translated.tsv"
    shell:
        "matUtils summary -i {input[2]} -t {output} -g {input[1]} -f {input[0]}"

rule write_clades:
    input:
        "{translate}.masked.pb"
    output:
        "{translate}.node_clades.txt"
    shell:
        "matUtils summary -i {input} -C {output}"

rule write_paths:
    input:
        "{translate}.masked.pb"
    output:
        "{translate}.sample_paths.txt"
    shell:
        "matUtils extract -i {input} -S {output}"

