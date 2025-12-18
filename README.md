# GWAS-based Selector

**GWAS-based Selector** is a Python tool that implements a **population-aware,
multiclass GWAS-driven feature selection pipeline** for **cgMLST allelic data**.

It wraps **Pyseer** ([Pyseer repository](https://github.com/mgalardini/pyseer)) to:
- handle **categorical cgMLST alleles**
- run **one-vs-rest (OVR) GWAS** for multiclass phenotypes
- scale to thousands of loci via **block-wise parallel execution**
- return **biologically meaningful loci** suitable for downstream machine learning or epidemiological analysis

## Overview and Concept

### Why GWAS-based Feature Selection?

Genome-Wide Association Studies (GWAS) aim to identify statistical associations
between genetic variants and a phenotype.

In microbial genomics, GWAS is used to investigate:
- host specificity
- virulence determinants
- antimicrobial resistance
- ecological or epidemiological classes

However, **bacterial populations are highly structured**: closely related strains tend to share both genotype and phenotype.
This induces strong confounding due to **population structure**.

The GWAS-based selector addresses this by:
- explicitly correcting for population structure
- selecting loci based on **statistical association, not correlation**
- reducing dimensionality while preserving biological signal

### How Pyseer is used here

**Pyseer** is a microbial GWAS engine that tests each genetic variant using
a model of the form:

```text
phenotype ~ variant + population_structure
```

Population structure is modeled using:
- a pairwise genomic distance matrix
- MDS components derived from that matrix
- linear or linear mixed models

Pyseer outputs association statistics (p-values) for each variant. GWAS-based Selector builds on Pyseer by:
- converting cgMLST alleles into binary presence/absence variants
- running Pyseer in parallel blocks
- applying structural and statistical filters
- aggregating results at the locus level
- supporting multiclass phenotypes via OVR strategy

## Inputs

The tool requires three mandatory inputs, all aligned by sample ids:

### cgMLST allelic profiles

Allelic profile matrix used to derive GWAS variants.

- **Rows**: samples (isolates)
- **Columns**: loci
- **Values**: categorical allele identifiers

| ceppoID | locusA | locusB | locusC | locusD |
|---------|--------|--------|--------|--------|
| S1 | 3 | 1 | 12 | 4 |
| S2 | 0 | 1 | 5 | 4 |
| S3 | 3 | 4 | 12 | 2 |
| S4 | 1 | 1 | 5 | 2 |

### Phenotypes

Sample-level phenotype labels used for One-vs-Rest GWAS.

- **Rows**: samples
- **Column**: categorical phenotype

| ceppoID | phenotype |
|---------|-----------|
| S1 | poultry |
| S2 | ruminant |
| S3 | poultry |
| S4 | wild_birds |

### Distance matrix

Pairwise genomic distance matrix used to model population structure.

- **Square matrix**
- **Same samples in rows and columns**
- **Diagonal = 0**

|     | S1 | S2 | S3 | S4 |
|-----|----|----|----|----|
| S1 | 0 | 12 | 3 | 9 |
| S2 | 12 | 0 | 14 | 8 |
| S3 | 3 | 14 | 0 | 10 |
| S4 | 9 | 8 | 10 | 0 |

## Process
## Outputs

