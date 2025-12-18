# GWAS-based Selector

**GWAS-based Selector** is a Python tool that implements a **population-aware,
multiclass GWAS-driven feature selection pipeline** for **cgMLST allelic data**.

It wraps **Pyseer** and extends it to:
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

---

### What Is Pyseer and how It is used here

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
## Process
## Outputs

