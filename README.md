# GWAS-based Selector (v.1.0.0-alpha)

**GWAS-based Selector** is a Python tool that implements a **population-aware,
multiclass GWAS-driven feature selection pipeline** for **cgMLST allelic data**.

It wraps **Pyseer** ([Pyseer repository](https://github.com/mgalardini/pyseer)) to:
- handle **categorical cgMLST alleles**
- run **one-vs-rest (OVR) GWAS** for multiclass phenotypes
- scale to thousands of loci via **block-wise parallel execution**
- return **biologically meaningful loci** suitable for downstream machine learning or epidemiological analysis

---

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

---

## Inputs

The tool requires three mandatory inputs, all aligned by sample ids:

### 1. cgMLST allelic profiles

Allelic profile matrix used to derive GWAS variants.

- **Rows**: samples (isolates)
- **Columns**: loci
- **Values**: categorical allele identifiers

| sampleID | locusA | locusB | locusC | locusD |
|---------|--------|--------|--------|--------|
| S1 | 3 | 1 | 12 | 4 |
| S2 | 0 | 1 | 5 | 4 |
| S3 | 3 | 4 | 12 | 2 |
| S4 | 1 | 1 | 5 | 2 |

### 2. Phenotypes

Sample-level phenotype labels used for One-vs-Rest GWAS.

- **Rows**: samples
- **Column**: categorical phenotype

| sampleID | phenotype |
|---------|-----------|
| S1 | p1 |
| S2 | p2 |
| S3 | p1 |
| S4 | p3 |

### 3. Distance matrix

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

---

## Process

GWAS-based Selector implements a **population-aware, multiclass GWAS pipeline**
on cgMLST allelic data. The process is designed to be **statistically robust**, **scalable**, and
**biologically interpretable**. The overview is shown in `GWAS-based-selector-arch.png` and in `GWAS-based-selector-wf.pdf` files.

### High-level workflow

1. Read and align **cgMLST** and **phenotypes** by sampleID.
2. Read the **distance matrix**, symmetrize it, and reorder it to match
   the cgMLST sample order
3. Enumerate cgMLST-derived variants and convert them to string identifiers  

   **Example:**

   ```text
   locus L1, alleles {0, 1, 3}
   → variants: L1_0, L1_1, L1_3, ...
   ```

4. Apply an optional Minor Allele Frequency (MAF) filter. Remove variants with very low or very high frequency (global, structure-based filter)
5. Compute a pattern-based Bonferroni threshold
   - count unique presence/absence patterns
   - compute threshold as:

     ```text
     p_threshold = α / (n_unique patterns)
     ```

6. For each phenotype class (One-vs-Rest):
   - build a binary phenotype vector
   - y_bin = 1{phenotype == class}
   - split all variants into blocks of size block_size
   - for each block (parallel execution):
         - apply pattern-threshold filtering (pre-Pyseer)
         - remove variants that are:
                  - monomorphic
                  - too rare in cases or controls
                  (phenotype-dependent structural filter)
         - generate an RTAB presence/absence matrix
           (samples × filtered variants)
         - run Pyseer using:
                  - RTAB block
                  - binary phenotype
                  - aligned distance matrix
                  - Multidimensional Scaling (MDS) components
         - extract (variant, p-value) pairs
         - merge results across all blocks
         - compute q-values using Benjamini–Hochberg FDR
         - apply a double statistical filter:
                  - p-value ≤ Bonferroni threshold
                  - q-value ≤ FDR threshold
         - map variants back to loci
         - aggregate variants at the locus level
           (minimum q-value per locus, q_min)
         - optional Top-K selection
         - retain the K loci with the highest significance
         (highest −log10(q_min))

7. Aggregate selected loci across classes
8. Save outputs

---

## Output

GWAS-based Selector generates multiple outputs, designed for both
**human interpretation** and **programmatic reuse**.

### 1. Filtered cgMLST Dataset

**File:** `filtered_subset_cgMLST.csv`  
*(generated only when `--multi union` is used)*

- **Rows:** samples
- **Columns:** loci selected by GWAS
- **Values:** original cgMLST alleles

This dataset represents the **GWAS-driven feature-selected cgMLST matrix**
and can be directly used for:
- machine learning models
- downstream statistical analyses

### 2. GWAS Results Summary

**File:** `results.txt`

Human-readable summary of the analysis, including:
- input statistics (number of samples, loci, and classes)
- parameters used for GWAS and filtering
- per-class results
- selected variants and loci with associated p-values and q-values

**Example:**

```text
## Class 'p1'
significant_variants_after_threshold: 12
final_selected_loci: 5

variant        pval        qval
locusA_3     2.3e-07     1.8e-06
locusC_12    5.0e-04     1.0e-02
...          ...         ...
```

### 3. Execution Log

**File:** `report.log`

Complete execution trace of the pipeline, including:
- timestamps
- block-level progress
- memory usage
- Pyseer commands and runtime information
- warnings and error messages

This file ensures **full reproducibility and debuggability**.

### 4. Serialized Feature Object

**File:** `selected_features.obj`

Binary (pickle) object for **programmatic reuse**.

It stores:
- all parameters used in the run
- per-class selected variants and loci
- aggregated multiclass selections

---

## Installation

This section describes how to install Conda, create a dedicated environment, and
install all required dependencies

### 1. Install Conda

If Conda is not already installed, download and install **Miniconda**
(the recommended lightweight distribution):

- **Miniconda (official download page):**  
  https://docs.conda.io/en/latest/miniconda.html

Follow the instructions for your operating system (Linux, macOS, or Windows/WSL).

### 2. Create environment

After installation, create your conda environment (e.g. gwas_selector_env):

```bash
conda create -n gwas_selector_env python=3.11 -y
conda activate gwas_selector_env
```

### 3. Install dependencies

Download the repository. Then:

```bash
cd <repo_name>
```

From the repository:

```bash
pip install -r requirements.txt
```

Example of usage pattern from cli:

```bash
python3 GWAS-based-selector.py \
  GWAS_INPUT/cgmlst.csv \
  GWAS_INPUT/phenotypes.csv \
  GWAS_INPUT/distance_matrix.tsv \
  --out GWAS_OUTPUT/run_01 \
  --log GWAS_OUTPUT/run_01/report.log \
  --pyseer pyseer \
  --id-col sample_id \
  --phen-col phenotype_col \
  --block-size 1000 \
  --workers 3 \
  --cpu 3 \
  --mds 10 \
  --maf 0.001 \
  --use-patterns-bonferroni \
  --alpha-patterns 0.001 \
  --min-case-count 5 \
  --min-control-count 5 \
  --fdr 0.001 \
  --k 50 \
  --multi union
```

To deactivate the environment (after the tool usage):

```bash
conda deactivate
```






