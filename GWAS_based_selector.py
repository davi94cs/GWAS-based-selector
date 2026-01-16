"""

Pyseer GWAS wrapper for cgMLST data (categorical) with population correction.

Input required:
- subset_cgMLST.csv      (columns: sampleID, Locus1, ..., LocusN; categorical/allelic values)
- subset_phenotypes.csv  (columns: sampleID, categorical target (poultry, ruminants, wild_birds))
- distance.tsv           (SQUARE MATRIX samples x samples; diagonal=0)

Output generated:
- filtered_subset_cgMLST.csv   (only selected loci)
- report.log                   (full execution log)
- results.txt                  (p-value, q-value, selected variants and loci, summary)
- selected_features.obj        (pickle: selected_variants, selected_loci, params)

"""

import os  # Standard library for interacting with the operating system: paths, environment variables, file/directory ops, process info
import re  # Regular expressions: searching, matching, and replacing text with patterns
import sys  # Access to Python runtime details: command-line args (sys.argv), standard I/O streams, exit codes, interpreter info
import time  # Time utilities: sleep, timestamps, simple timing/benchmarking
import math  # Fast, low-level math functions/constants (e.g., sqrt, log, pi)
import pickle  # Serialize/deserialize Python objects to/from byte streams (for caching or saving models/configs)
import logging  # Configurable logging framework (levels, handlers, formatting) for structured runtime messages
import tempfile  # Create secure temporary files/directories that auto-clean when closed
import subprocess  # Spawn and manage external processes/commands; capture their output/return codes
from typing import List, Tuple, Dict, Optional  # Type hints for readability and static analysis: annotate list/tuple/dict types
import numpy as np  # Core numerical array library: vectorized computation, linear algebra, random numbers
import pandas as pd  # Data analysis library: DataFrame/Series for tabular data, CSV/TSV I/O, grouping, joins
from concurrent.futures import ProcessPoolExecutor, as_completed  # High-level multiprocessing: run functions in parallel across processes; as_completed iterates results as workers finish
import argparse  # Command-line argument parsing: define flags/options, auto-generate --help
from collections import Counter  # Count hashable objects: tally occurrences in lists/iterables, useful for frequency analysis

try:  # Tries to import psutil (CPU/RAM usage monitoring). Sets _HAS_PSUTIL flag so the rest of the program can optionally use it if installed. For installation (post environment activation): conda install -c conda-forge psutil 
    import psutil  
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

### CLI ###

def parse_args():

    """
    Parse command-line arguments for the cgMLST multiclass OVR GWAS selector.
    """

    ap = argparse.ArgumentParser(description="Pyseer-based multiclass (OVR) GWAS on cgMLST with square distance matrix.")
    # Positional
    ap.add_argument("cgmlst_csv", help="subset_cgMLST.csv (sampleID + loci categorical)")
    ap.add_argument("phen_csv", help="subset_phenotypes.csv (sampleID, phenotype category)")
    ap.add_argument("distance_tsv", help="distance.tsv (square matrix sampleID x sampleID)")
    # Pyseer & blocks
    ap.add_argument("--pyseer", default="pyseer", help="Pyseer executable")
    ap.add_argument("--block-size", type=int, default=2000, help="Variants per RTAB block")
    ap.add_argument("--workers", type=int, default=3, help="Parallel pyseer blocks")
    ap.add_argument("--cpu", type=int, default=3, help="CPU cores per pyseer block")
    # MDS = number of dimensions (mapped to --max-dimensions)
    ap.add_argument("--mds", type=int, default=15, help="Number of MDS dimensions for pyseer")
    # Filtering
    ap.add_argument("--maf", type=float, default=None, help="Minimum allele frequency (MAF) filter")
    ap.add_argument("--min-case-count", type=int, default=5, help="Minimum number of case samples carrying a variant for it to be kept in a block "
                                                                   "(pattern threshold).")
    ap.add_argument("--min-control-count", type=int, default=5, help="Minimum number of control samples carrying a variant for it to be kept in a block "
                                                                     "(pattern threshold).")
    ap.add_argument("--alpha-patterns", type=float, default=None, help=("Alpha for Bonferroni threshold based on the number of unique variant patterns "
                                                                        "(analogous to pyseer count_patterns.py --alpha). "
                                                                        "If not set, no pattern-based Bonferroni threshold is applied."))
    ap.add_argument("--use-patterns-bonferroni", action="store_true", help=("If set, use a Bonferroni threshold based on the number of unique variant patterns "
                                                                            "to filter pyseer results (similar to count_patterns.py)."))
    ap.add_argument("--fdr", type=float, default=0.001, help="BH-FDR threshold")
    ap.add_argument("--k", type=int, default=None, help="Max number of loci to keep per class after FDR (top-K by best q-value per locus)."
                                                        "If not set, keep all significant loci.")
    # Multiclass mode
    ap.add_argument("--multi", choices=["union", "per_class"], default="union", help=("Union: filter cgMLST on union of selected loci across classes."
                                                                                      "Per_class: no global filtering."))
    # Logging and output
    ap.add_argument("--log", default="report.log", help="Log file path")
    ap.add_argument("--out", default="selector_out", help="Output folder for selector results")
    # Coherence with column names
    ap.add_argument("--id-col", default="id", help="Name of the sample ID column in both cgMLST and phenotype files")
    ap.add_argument("--phen-col", default="target_IZSAM", help="Name of the phenotype column in the phenotype file")
    return ap.parse_args()

### LOGGER ---> LOGGING HELPERS ###

def mem_usage_str() -> str:

    """
    Return a formatted string describing the current process memory usage.

    - If `psutil` is not installed, returns a placeholder string.
    - Retrieves the current process using its PID.
    - Reads the process's resident set size (RSS), i.e., physical memory usage.
    - Collects all child processes recursively.
    - Computes the total RSS memory used by all child processes.
    - Defines a helper `fmt()` function to convert bytes into megabytes.
    - Returns a formatted string showing:
        * rss: memory used by the main process
        * children_rss: total memory used by all child processes
        * n_children: number of child processes
    """

    if not _HAS_PSUTIL:
        return "mem=? (psutil not installed)"
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss
    children = proc.children(recursive=True)
    rss_child = sum((c.memory_info().rss for c in children), start=0)
    def fmt(b): return f"{b/1024/1024:.1f}MB"
    return f"rss={fmt(rss)} children_rss={fmt(rss_child)} n_children={len(children)}"

def setup_logger(log_path: str, level_console=logging.INFO, level_file=logging.DEBUG) -> logging.Logger:

    """
    Configure and return a logger that writes messages to both a file and the console.

    - Creates or retrieves a logger named "GWASSelector".
    - Sets the logger level to DEBUG (the most verbose).
    - Defines a log message format with timestamp, level, and message.
    - Creates a FileHandler that writes logs to `log_path` in UTF-8 (overwrite mode).
    - Sets the file handler level and applies the defined format.
    - Creates a StreamHandler that writes logs to stdout (console).
    - Sets the console handler level and applies the same format.
    - Ensures handlers are not duplicated if the function is called multiple times.
    - Adds both file and console handlers to the logger.
    - Returns the fully configured logger instance.
    """

    logger = logging.getLogger("GWAS_selector")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level_file); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_console); ch.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

### DATA SYNC ---> DISTANCE MATRIX COHERENCE ###

def read_square_distance_matrix(path: str, logger: logging.Logger) -> pd.DataFrame:

    """
    Read and validate a square distance matrix from a file, symmetrize it, and zero the diagonal.

    - Logs the start of the loading process.
    - Attempts to read the file as a tab-separated table; if that fails, retries with comma separation.
    - Ensures both row and column labels are strings.
    - Converts all cell values to numeric (invalid entries become NaN).
    - Checks that the matrix is square (same number of rows and columns).
    - Checks that row and column labels correspond to the same set of sample IDs.
    - Reorders the columns to match the row order exactly.
    - Sets the diagonal values to zero.
    - Symmetrizes the matrix by averaging it with its transpose.
    - Logs summary information: number of samples, diagonal zeroed, and symmetry confirmation.
    - Returns the processed symmetric square distance matrix as a pandas DataFrame.
    """
    
    logger.info(f"[DIST] reading square distance matrix: {path}")
    try:
        D = pd.read_csv(path, sep="\t", header=0, index_col=0)
    except Exception:
        D = pd.read_csv(path, sep=",", header=0, index_col=0)
    D.index = D.index.astype(str)
    D.columns = D.columns.astype(str)
    D = D.apply(pd.to_numeric, errors="coerce")
    if D.shape[0] != D.shape[1]:
        raise ValueError(f"[DIST] not square matrix: {D.shape}") # 
    if set(D.index) != set(D.columns):
        raise ValueError("[DIST] rows and columns do not match as sample sets") 
    # Controllo NaN: meglio fallire qui che passare una matrice sporca a pyseer
    if D.isna().values.any():
        n_nan = int(D.isna().sum().sum())
        logger.error(f"[DIST] distance matrix contains {n_nan} NaN entries")
        raise ValueError("[DIST] distance matrix contains NaN values")
    # ordina colonne come le righe, azzera diag, simmetrizza
    D = D.loc[D.index, D.index]
    np.fill_diagonal(D.values, 0.0)
    D = (D + D.T) / 2.0
    logger.info(f"[DIST] loaded N={D.shape[0]} | diag=0 | sym OK")
    return D

def subset_order_distance(D: pd.DataFrame, ids: pd.Index, logger: logging.Logger) -> pd.DataFrame:

    """
    Subset and reorder a distance matrix to match a given list of sample IDs.
    
    - Normalizes the given sample IDs (`ids`) to strings, to ensure consistent
    comparison with the distance matrix index (which is also converted to str
    in `read_square_distance_matrix`).

    - Checks that all requested IDs exist in the distance matrix `D`.  
    If some IDs are missing, logs an error with an example of missing entries
    and raises a ValueError.

    - Extracts a square submatrix of `D` restricted to those IDs, preserving
    exactly the order in which they appear in `ids`.

    - Sets the diagonal of the resulting matrix to zero to guarantee consistency
    (even if the source matrix already has zero diagonal).

    - Returns the subsetted and properly ordered distance matrix as a pandas
    DataFrame ready for pyseer `--distances`.
    """

    # Normalizza gli ids a stringa per essere coerenti con D.index (che è str)
    ids_str = pd.Index([str(i) for i in ids])
    missing = [i for i in ids_str if i not in D.index]
    if missing:
        logger.error(f"[DIST] {len(missing)} missing samples in distance.tsv (es. {missing[:5]})")
        raise ValueError("distance matrix does not cover all samples.")
    D_sub = D.loc[ids_str, ids_str].copy()
    np.fill_diagonal(D_sub.values, 0.0)
    logger.info(f"[DIST] subset distance matrix to N={D_sub.shape[0]} samples")
    return D_sub

### VARIANTS GENERATOR ---> VARIANTS (ALLELES) UTILITIES ###

def maf_filter_variants_per_locus(cg: pd.DataFrame, maf: Optional[float], logger=None) -> List[str]:
    """
    Efficient MAF filter for allele-based variants (locus_allele) computed per locus.

    Parameters:
    - cg as cgMLST dataframe indexed by sample ID, with loci as columns.
    - maf as MAF threshold in (0, 0.5]. If None or <= 0, no filtering is performed.
    - logger (optional)

    Returns:
    - List of variants "locus_allele" that pass maf <= freq <= (1 - maf).
    """

    if maf is None or maf <= 0.0:
        # No MAF filtering: enumerate all observed alleles
        all_vars: List[str] = []
        for locus in cg.columns:
            # dropna to avoid "nan" being treated as an allele
            alleles = cg[locus].dropna().astype(str).unique().tolist()
            all_vars.extend([f"{locus}_{a}" for a in alleles])
        if logger:
            logger.info(f"[MAF] disabled | variants={len(all_vars)}")
        return all_vars
    if maf > 0.5:
        raise ValueError("maf must be <= 0.5")
    n = len(cg)
    out: List[str] = []
    for locus in cg.columns:
        # Convert once per locus
        s = cg[locus].dropna().astype(str)
        if s.empty:
            continue
        counts = s.value_counts(dropna=True)
        freqs = counts / float(n)  
        # keep alleles with maf <= f <= 1-maf
        keep_alleles = freqs[(freqs >= maf) & (freqs <= (1.0 - maf))].index.tolist()
        out.extend([f"{locus}_{a}" for a in keep_alleles])
    if logger:
        logger.info(f"[MAF] maf_min={maf} | variants_kept={len(out)}")
    return out

def variant_to_locus(variant: str) -> str:

    """
    Extract the locus name from a variant identifier string.

    - Handles multiple possible separators:
        * "_" (underscore)
        * "=" (equals sign)
        * "|", ":", or "-" (generic delimiters)
    - Splits the input string at the first occurrence of these separators.
    - Returns the prefix (the locus name) as a string.
    - Used to map back from full variant names (e.g., "geneX_a1") to their base locus ("geneX").
    """

    if "_" in variant:
        return variant.split("_", 1)[0]
    if "=" in variant:
        return variant.split("=", 1)[0]
    return re.split(r"[\|\:\-]", variant)[0]

### GWAS MAKER ---> UTILS, RTAB CONSTRUCTION, PYSEER RUNNING (FOR EACH CLASS) ###

def chunk_list(items: List[str], block_size: int) -> List[List[str]]:

    """
    Split a list of items into consecutive chunks of a specified block size.

    - Takes a list `items` and an integer `block_size`.
    - Iteratively slices the list into sublists, each containing up to `block_size` elements.
    - The final chunk may be smaller if the total number of items is not divisible by `block_size`.
    - Returns a list of sublists, preserving the original order.
    """

    return [items[i:i + block_size] for i in range(0, len(items), block_size)]

def write_rtab_block_stream(cgmlst_df: pd.DataFrame, sample_col: str, block_variants: List[str], out_path: str, logger: logging.Logger) -> None:

    """
    Write an RTAB file for a block of variants in a faster way:
    - group variants by locus
    - convert each locus column to str once
    - compute patterns with numpy vectorization

    RTAB format produced:
      header: "variant<TAB>S1<TAB>S2..."
      each row: "locus_allele<TAB>0/1<TAB>0/1..."

    Parameters:
    
    - cgmlst_df : pd.DataFrame
        - Table with sample column and loci columns (alleles).
        - Often you pass cg.reset_index() so that sample_col exists as a column.
    - sample_col : str
        - Name of the sample column (e.g. "ceppoID").
    - block_variants : List[str]
        - Variants in "locus_allele" format.
    - out_path : str
        - Output path.
    """

    if sample_col not in cgmlst_df.columns:
        raise ValueError(f"Missing sample column '{sample_col}' in df")
    samples = cgmlst_df[sample_col].astype(str).tolist()
    # Group alleles by locus to avoid repeated per-variant column conversion
    locus_to_alleles: Dict[str, List[str]] = {}
    for v in block_variants:
        if "_" not in v:
            continue
        locus, allele = v.split("_", 1)
        locus_to_alleles.setdefault(locus, []).append(allele)
    # Write RTAB
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("variant\t" + "\t".join(samples) + "\n")
        for locus, alleles in locus_to_alleles.items():
            if locus not in cgmlst_df.columns:
                continue
            # Convert column once per locus
            col_vals = cgmlst_df[locus].astype(str).to_numpy()
            # For deterministic output, you may sort alleles
            # alleles = sorted(set(alleles))
            for allele in alleles:
                pattern = (col_vals == allele)  # boolean array
                # Convert boolean to 0/1 strings efficiently
                bits = np.where(pattern, "1", "0")
                f.write(f"{locus}_{allele}\t" + "\t".join(bits.tolist()) + "\n")
    if logger is not None:
        logger.info(f"[RTAB] wrote {out_path} | variants={len(block_variants)} samples={len(samples)}")

def run_pyseer_block(pyseer_bin: str, phen_path: str, distances_square_path: str, mds_components: int, rtab_path: str, out_path: str,
                     cpu_per_block: int, logger: logging.Logger) -> None:
    
    """
    Run Pyseer on a single RTAB block and write the association results to `out_path`.

    - Builds the Pyseer command using: --phenotypes, --pres, --distances,
      --max-dimensions, and --cpu.
    - Executes Pyseer via subprocess; stdout (association table) is streamed directly
      to the output file.
    - Logs the command, runtime, memory usage, and output size.
    - Raises an error if the Pyseer binary is missing or if the run exits with a
      non-zero return code.
    """

    cmd = [pyseer_bin, "--phenotypes", phen_path, "--pres", rtab_path, "--distances", distances_square_path, "--max-dimensions", str(mds_components), 
           "--cpu", str(cpu_per_block)]
    logger.info(f"[BLOCK] run pyseer on {os.path.basename(rtab_path)} → {os.path.basename(out_path)} "
                f"| cpu={cpu_per_block} | {mem_usage_str()}")
    logger.debug(f"[BLOCK] cmd: {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        with open(out_path, "w") as fout:
            res = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        logger.error(f"[BLOCK] pyseer binary not found: {pyseer_bin}")
        raise
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        logger.error(f"[BLOCK] pyseer FAILED in {dt:.2f}s\nSTDERR:\n{res.stderr}")
        raise RuntimeError("pyseer failed for block")
    else:
        try:
            size_mb = os.path.getsize(out_path) / 1024 / 1024
        except Exception:
            size_mb = float("nan")
        logger.info(f"[BLOCK] pyseer OK in {dt:.2f}s out_size={size_mb:.1f}MB | {mem_usage_str()}")

def filter_variants_by_pattern_threshold(cg_str: pd.DataFrame, block_variants: List[str], y_bin: pd.Series, min_case: int = 5, min_control: int = 5, logger=None) -> List[str]:

    """
    Structural pre-GWAS filtering of block variants using case/control pattern thresholds.

    This function implements a PLINK-like QC step at the per-variant level:
    for each allele-specific variant (locus_allele), it examines the
    presence/absence pattern across samples and removes patterns that cannot
    support a stable or meaningful association test.

    For each variant v (e.g. "locusA_3") we compute:
      - pattern: 0/1 presence across samples
      - case_count:  number of case samples (y_bin == 1) that carry the allele
      - control_count: number of control samples (y_bin == 0) that carry the allele

    A variant is discarded if:
      - it is monomorphic (pattern all 0 or all 1), or
      - case_count < min_case, or
      - control_count < min_control.

    In particular:
      - If pattern == y_bin (allele only in cases, never in controls),
        then control_count = 0 and the variant is removed.
      - If pattern is the exact opposite of y_bin (allele only in controls),
        then case_count = 0 and the variant is removed.
      - Only variants with mixed presence (some cases and some controls carrying
        the allele) and sufficient counts in both groups are retained.

    Biological rationale:
  
    A variant is biologically informative only if it shows variation within both
    cases and controls. If all cases carry the allele and no controls do (or
    vice versa), the signal is typically driven by shared ancestry / clonal
    expansion rather than a true phenotype effect. This is the same motivation
    behind standard GWAS QC in tools like PLINK, where monomorphic variants,
    variants with very low minor allele counts, or variants present only in one
    phenotype group are removed before association testing.

    Statistical rationale:
    
    Variants present exclusively in cases or exclusively in controls cause
    complete or quasi-complete separation in logistic/LMM models, leading to
    unstable or infinite effect estimates and unreliable p-values. Requiring
    minimum counts in both groups (min_case, min_control) ensures that the
    regression has enough within-group variance to estimate an effect, improving
    the stability and power of downstream Pyseer runs.

    """

    # Ensure alignment
    y_bin = y_bin.loc[cg_str.index]
    case_mask = (y_bin.values == 1)
    control_mask = (y_bin.values == 0)
    keep: List[str] = []
    n = len(cg_str)
    # Use string representation to compare alleles robustly
    #cg_str = cg.astype(str)
    cols = cg_str.columns
    for v in block_variants:
        if "_" not in v:
            continue
        locus, allele = v.split("_", 1)
        if locus not in cols:
            continue
        col = cg_str[locus].values
        pattern = (col == allele)  # presence/absence
        total_count = int(pattern.sum())
        if total_count == 0 or total_count == n:
            # monomorphic pattern: no information
            del col, pattern
            continue
        case_count = int((pattern & case_mask).sum())
        control_count = int((pattern & control_mask).sum())
        if case_count < min_case or control_count < min_control:
            # too rare in either cases or controls
            del col, pattern
            continue
        keep.append(v)
        del col, pattern
    if logger is not None:
        logger.info(f"[THRESH] block_variants={len(block_variants)} → "f"kept_after_pattern_threshold={len(keep)} "f"(min_case={min_case}, min_control={min_control})")
    del cg_str, cols, case_mask, control_mask
    return keep

def run_ovr_gwas_for_class(label: str, y_bin: pd.Series, cg: pd.DataFrame, distances_square_path: str, all_variants: List[str], pyseer_bin: str,
                            mds_components: int, block_size: int, workers: int, cpu_per_block: int, logger: logging.Logger, min_case_count: int,
                            min_control_count: int) -> pd.DataFrame:
    
    """
    Run a one-vs-rest GWAS for a single class using Pyseer.

    - Takes a binary phenotype vector `y_bin` for the target class (1 = class, 0 = rest).
    - Splits `all_variants` into blocks of size `block_size`.
    - For each block:
      - Applies a pattern-threshold filter. It is a structural filter that removes non-informative variants for the given class (phenotype dependent)
      - Builds an RTAB presence/absence matrix from `cg` for the filtered variants.
      - Writes a temporary phenotype file for this class.
      - Launches Pyseer on the block (via `run_pyseer_block`) in parallel
        using `workers` processes and `cpu_per_block` CPUs per Pyseer run.
    - Collects per-block Pyseer outputs, normalizes column names, and extracts
      (variant, pval) pairs.
    - Concatenates all blocks into a single DataFrame, drops NaN p-values,
      sorts by p-value, and returns the GWAS table for this class.
    """
    
    cg_str = cg.astype(str)
    # Log di base sulla classe
    n_pos = int(y_bin.sum())
    n_neg = len(y_bin) - n_pos
    logger.info(f"[OVR:{label}] start | positives={n_pos} negatives={n_neg}")
    # Suddivide le varianti in blocchi
    blocks = chunk_list(all_variants, block_size)
    n_blocks = len(blocks)
    logger.info(f"[OVR:{label}] n_blocks={n_blocks} (~{block_size} vars/block) "
                f"| pattern_threshold min_case={min_case_count} min_control={min_control_count}")
    with tempfile.TemporaryDirectory() as tmpdir:
        # File di fenotipo per questa classe (con header, come si aspetta pyseer)
        phen_path = os.path.join(tmpdir, f"phen_{label}.txt")
        phen_df = pd.DataFrame({"sample": y_bin.index.astype(str), "phenotype": y_bin.astype(int).values})
        phen_df.to_csv(phen_path, sep="\t", header=True, index=False)
        del phen_df
        out_paths: List[str] = []
        futures = []
        # Parallelizza sui blocchi: N workers, ciascuno si prende M core cpu -> N*M core totali cpu usati in parallelo. Quando un blocco (e quindi un worker) finisce, ne parte un altro e si prende 3 core
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for bi, block_variants in enumerate(blocks):
                # 1) pattern-threshold per blocco 
                filtered_block_variants = filter_variants_by_pattern_threshold(cg_str=cg_str, block_variants=block_variants, y_bin=y_bin, min_case=min_case_count,
                                                                                min_control=min_control_count, logger=logger)
                if not filtered_block_variants:
                    logger.info(f"[OVR:{label}] block {bi+1}/{n_blocks} skipped after pattern "
                                f"threshold (0 variants kept).")
                    del filtered_block_variants
                    continue
                # 2) RTAB per questo blocco (solo varianti filtrate)
                bdir = os.path.join(tmpdir, f"block_{bi:05d}")
                os.makedirs(bdir, exist_ok=True)
                rtab_path = os.path.join(bdir, "block.rtab")
                write_rtab_block_stream(cg.reset_index(),"ceppoID", filtered_block_variants, rtab_path, logger)
                del filtered_block_variants
                # 3) Output associazioni pyseer per questo blocco
                out_path = os.path.join(bdir, "out.tsv")
                out_paths.append(out_path)
                futures.append(
                    ex.submit(run_pyseer_block, pyseer_bin, phen_path, distances_square_path, mds_components, rtab_path, out_path, cpu_per_block, logger))
            del blocks
            n_submitted = len(futures)
            if n_submitted == 0:
                logger.warning(f"[OVR:{label}] No blocks submitted to pyseer "
                               f"after pattern threshold (no variants kept).")
                del futures
                raise ValueError(f"No variants passed pattern threshold for class '{label}'.")
            done = 0
            for fut in as_completed(futures):
                # Se un blocco fallisce, solleva qui (RuntimeError da run_pyseer_block)
                fut.result()
                done += 1
                logger.info(f"[OVR:{label}] block {done}/{n_submitted} completed | "
                            f"{mem_usage_str()}")
            del futures   
        # Merge dei risultati pyseer per la classe
        gwas_list: List[pd.DataFrame] = []
        for pth in out_paths:
            if not os.path.exists(pth):
                logger.error(f"[OVR:{label}] missing block output: {pth}")
                raise FileNotFoundError("Manca out.tsv di un blocco")
            df = pd.read_csv(pth, sep="\t")
            # Normalizza nomi colonne (case-insensitive)
            cols = {c.lower(): c for c in df.columns}
            # Colonna variante: tipicamente 'variant'
            vcol = cols.get("variant", cols.get("pattern"))
            # Colonna p-value: tipicamente 'lrt-pvalue'; fallback su 'pvalue'/'pval'
            pcol = None
            for key in ("lrt-pvalue", "pvalue", "pval"):
                if key in cols:
                    pcol = cols[key]
                    break
            if vcol is None or pcol is None:
                logger.error(f"[OVR:{label}] colonne variant/pval non trovate in {pth} "
                             f"(columns={list(df.columns)})")
                del df, cols
                raise ValueError("Colonne mancanti variant/pval (es. 'variant', 'lrt-pvalue').")
            gwas_list.append(df[[vcol, pcol]].rename(columns={vcol: "variant", pcol: "pval"}))
            del df, cols, vcol, pcol
        if not gwas_list:
            logger.error(f"[OVR:{label}] nessun risultato GWAS disponibile.")
            del out_paths
            raise ValueError("Nessun risultato GWAS da combinare.")
        gwas = (pd.concat(gwas_list, axis=0, ignore_index=True).dropna(subset=["pval"]).sort_values("pval", ascending=True))
        del gwas_list, out_paths
        return gwas

### STATISTICAL FILTER ---> BH_FDR, BONFERRONI ###

def bh_fdr(pvals: np.ndarray) -> np.ndarray:

    """
    Apply the Benjamini–Hochberg (BH) correction for multiple testing (FDR control).

    - Takes a 1D array of raw p-values (`pvals`).
    - Sorts the p-values in ascending order and records their indices.
    - Computes the BH-adjusted q-values:
          q_i = (p_i * m) / rank_i (rank_i = 1 se i-esimo p-value è il più piccolo...)
      where m = total number of tests (non-NaN entries).
    - Enforces monotonicity by applying a cumulative minimum from the end.
    - Reorders the q-values back to the original p-value order.
    - Returns an array of adjusted p-values (q-values), same shape as input.
    """
    pvals = np.asarray(pvals, dtype=float)
    if pvals.ndim != 1:
        raise ValueError("bh_fdr expects a 1D array of p-values.")
    m = len(pvals)
    if m == 0:
        return pvals.copy()
    # Ordina per p crescente
    order = np.argsort(pvals)
    ranked = pvals[order]
    # BH formula: q_i = p_i * m / rank_i
    denom = np.arange(1, m + 1, dtype=float)
    q = ranked * m / denom
    # Monotonicità: q_i non deve aumentare passando dai più significativi ai meno
    q = np.minimum.accumulate(q[::-1])[::-1]
    # Rimappa nell'ordine originale
    out = np.empty_like(q)
    out[order] = q
    return out

def bonferroni_threshold_from_patterns(cg: pd.DataFrame, all_variants: List[str], alpha: float, logger: logging.Logger) -> float:

    """
    Approximate pyseer count_patterns Bonferroni threshold (statistical filter):

    - Build a 0/1 pattern for each variant (locus_allele),
    - Count the number of unique patterns across all variants,
    - Return alpha / n_unique_patterns.

    This is conceptually equivalent to:
        python scripts/count_patterns.py --alpha <alpha> patterns.txt
    but done in-memory using cgMLST and the allele encoding.
    """

    cg_str = cg.astype(str)
    cols = cg_str.columns
    patterns_seen = set()
    n_variants_total = len(all_variants)
    logger.info(f"[BONF] Computing unique patterns for {n_variants_total} variants "
                f"with alpha={alpha} (this may take some time).")
    for i, v in enumerate(all_variants, start=1):
        if "_" not in v:
            continue
        locus, allele = v.split("_", 1)
        if locus not in cols:
            continue
        col = cg_str[locus].values
        pattern = (col == allele).astype(np.uint8)
        key = pattern.tobytes()
        patterns_seen.add(key)
        if (i % 5000) == 0:
            logger.info(f"[BONF] processed {i}/{n_variants_total} variants | "
                        f"current_unique_patterns={len(patterns_seen)}")
        # Free per-iteration large temporaries
        del col, pattern, key
    n_unique = len(patterns_seen)
    if n_unique == 0:
        raise ValueError("[BONF] No unique patterns found (n_unique=0).")
    p_thr = alpha / float(n_unique)
    logger.info(f"[BONF] alpha={alpha} n_unique_patterns={n_unique} → "
                f"Bonferroni p-threshold={p_thr:.3e}")
    # Free heavy structures no longer needed
    del cg_str, cols, patterns_seen
    return p_thr

### MAIN (FULL PROCESS + OUTPUT AGGREGATOR) ###

def main():

    """
    Run the full multiclass cgMLST OVR GWAS pipeline using Pyseer.

    Pipeline summary:
    - Read and align cgMLST + phenotypes (id_col → ceppoID, phen_col → phenotype).
    - Read and align square distance matrix.
    - Enumerate cgMLST-derived variants and apply a global filter (MAF).
    - Compute a pattern-based Bonferroni p-threshold (count_patterns like).
    - For each phenotype class:
        * build one-vs-rest labels
        * run OVR GWAS via pyseer (rtab blocks), including phenotype dependent filtering
        * compute q-values (Benjamini-Hochberg)
        * filter by Bonferroni p-threshold and FDR (double filter)
        * aggregate variants per locus and apply Top-K by smallest q-value.
    - Aggregate loci across classes according to --multi (e.g. union).
    - Save:
        * filtered_subset_cgMLST.csv (for union strategy),
        * selected_features.obj,
        * results.txt.
    """

    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    logger = setup_logger(args.log)
    logger.info(f"[START] cgmlst={args.cgmlst_csv} phen={args.phen_csv} dist={args.distance_tsv}")
    logger.info(f"[PARAMS] pyseer={args.pyseer} block_size={args.block_size} "
                f"workers={args.workers} cpu_per_block={args.cpu} mds_dims={args.mds} "
                f"fdr={args.fdr} maf={args.maf} multi={args.multi} | {mem_usage_str()}")
    t0 = time.perf_counter()
    # 1) Lettura e allineamento cgMLST + phenotypes
    cg = pd.read_csv(args.cgmlst_csv)
    ph = pd.read_csv(args.phen_csv)
    if args.id_col not in cg.columns:
        logger.error(f"cgMLST file missing ID column '{args.id_col}'. Columns: {list(cg.columns)}")
        sys.exit(2)
    if args.id_col not in ph.columns:
        logger.error(f"Phenotypes file missing ID column '{args.id_col}'. Columns: {list(ph.columns)}")
        sys.exit(2)
    if args.phen_col not in ph.columns:
        logger.error(f"Phenotypes file missing phenotype column '{args.phen_col}'. "
                     f"Columns: {list(ph.columns)}")
        sys.exit(2)
    # Rinomina in schema interno: ceppoID / phenotype
    if args.id_col != "ceppoID":
        cg = cg.rename(columns={args.id_col: "ceppoID"})
        ph = ph.rename(columns={args.id_col: "ceppoID"})
    if args.phen_col != "phenotype":
        ph = ph.rename(columns={args.phen_col: "phenotype"})
    ph = ph.set_index("ceppoID")
    y_cat = ph["phenotype"].astype(str)
    cg = cg.set_index("ceppoID")
    common = cg.index.intersection(y_cat.index)
    if len(common) == 0:
        logger.error("Nessun campione in comune tra cgMLST e phenotypes.")
        sys.exit(2)
    cg = cg.loc[common]
    y_cat = y_cat.loc[common]
    classes = sorted(y_cat.unique().tolist())
    logger.info(f"[IO] aligned samples={len(common)} loci={cg.shape[1]} "
                f"classes={classes} | {mem_usage_str()}")
    # 2) Matrice di distanza: lettura, subset/ordine per cg.index
    D_full = read_square_distance_matrix(args.distance_tsv, logger)
    D = subset_order_distance(D_full, cg.index, logger)
    del D_full
    # Bonferroni threshold (valore iniziale)
    bonf_p_thr: float | None = None
    with tempfile.TemporaryDirectory() as tmpdir:
        dist_aligned_path = os.path.join(tmpdir, "distance_aligned.tsv")
        D.to_csv(dist_aligned_path, sep="\t")
        logger.info(f"[DIST] aligned distance matrix written to {dist_aligned_path} | shape={D.shape}")
        del D
        # 3) Enumerazione varianti da cgMLST + filtro MAF globale
        all_variants = maf_filter_variants_per_locus(cg=cg, maf=args.maf, logger=logger)
        logger.info(f"[ENUM+MAF] total_variants_after_maf={len(all_variants)}")
        # 4b) Bonferroni threshold calculation
        if getattr(args, "use_patterns_bonferroni", False) and args.alpha_patterns is not None:
            bonf_p_thr = bonferroni_threshold_from_patterns(cg=cg, all_variants=all_variants, alpha=args.alpha_patterns, logger=logger)
        else:
            logger.info("[BONF] Pattern-based Bonferroni not used.")
        # 5) One-vs-Rest per ogni classe: pyseer + filtri post-GWAS
        per_class_results: Dict[str, pd.DataFrame] = {}
        per_class_selected_variants: Dict[str, List[str]] = {}
        per_class_selected_loci: Dict[str, List[str]] = {}
        per_class_stats: Dict[str, Dict[str, int]] = {}
        for cls in classes:
            y_bin = (y_cat == cls).astype(int)
            gwas_cls = run_ovr_gwas_for_class(label=cls, y_bin=y_bin, cg=cg, distances_square_path=dist_aligned_path, all_variants=all_variants, pyseer_bin=args.pyseer,
                                              mds_components=args.mds, block_size=args.block_size, workers=args.workers, cpu_per_block=args.cpu, logger=logger,
                                              min_case_count=args.min_case_count, min_control_count=args.min_control_count)
            del y_bin
            if gwas_cls.empty:
                logger.warning(f"[OVR:{cls}] No GWAS results (empty GWAS table).")
                per_class_results[cls] = gwas_cls
                per_class_selected_variants[cls] = []
                per_class_selected_loci[cls] = []
                per_class_stats[cls] = {"n_sig": 0, "n_loci_final": 0}
                continue
            # 5a) Benjamini–Hochberg: calcolo qval
            gwas_cls = gwas_cls.sort_values("pval", ascending=True)
            gwas_cls["qval"] = bh_fdr(gwas_cls["pval"].values)
            if not gwas_cls["pval"].between(0, 1).all():
                logger.error(f"[OVR:{cls}] p-values out of [0,1].")
                sys.exit(3)
            if not gwas_cls["qval"].between(0, 1).all():
                logger.error(f"[OVR:{cls}] q-values out of [0,1].")
                sys.exit(3)
            per_class_results[cls] = gwas_cls
            # 5b) Filtro di significatività: Bonferroni (pattern-based) + FDR (doppio filtro)
            if bonf_p_thr is not None:
                mask_sig = gwas_cls["pval"] <= bonf_p_thr
            else:
                mask_sig = np.ones(len(gwas_cls), dtype=bool)
            if args.fdr is not None:
                mask_sig &= (gwas_cls["qval"] <= args.fdr)
            df_sig = gwas_cls[mask_sig].copy()
            del mask_sig
            n_sig = len(df_sig)
            logger.info(
                f"[OVR:{cls}] significant_variants after Bonferroni/FDR: {n_sig} "
                f"(bonf_thr={bonf_p_thr:.3e} if not None)"
                if bonf_p_thr is not None
                else f"[OVR:{cls}] significant_variants after FDR only: {n_sig}")
            if n_sig == 0:
                per_class_selected_variants[cls] = []
                per_class_selected_loci[cls] = []
                per_class_stats[cls] = {"n_sig": 0, "n_loci_final": 0}
                continue
            df_sig["locus"] = df_sig["variant"].map(variant_to_locus)
            # 5c) Aggregazione per locus: q_min e score basato su q_min
            agg = df_sig.groupby("locus").agg(q_min=("qval", "min"))
            agg["score"] = -np.log10(agg["q_min"])
            agg_sorted = agg.sort_values("score", ascending=False)
            loci_sorted = list(agg_sorted.index)
            # 5d) Top-K loci (opzionale)
            if args.k is not None:
                top_loci = loci_sorted[: args.k]
            else:
                top_loci = loci_sorted
            df_final = df_sig[df_sig["locus"].isin(top_loci)].copy()
            sel_vars = df_final["variant"].tolist()
            sel_loci = sorted(set(top_loci))
            per_class_selected_variants[cls] = sel_vars
            per_class_selected_loci[cls] = sel_loci
            per_class_stats[cls] = {"n_sig": n_sig, "n_loci_final": len(sel_loci)}
            logger.info(f"[OVR:{cls}] FINAL loci={len(sel_loci)} "
                        f"(significant={n_sig}, "
                        f"topK={args.k if args.k is not None else 'ALL'})")
            del df_sig, agg, agg_sorted, loci_sorted, top_loci, df_final, sel_vars, sel_loci
        # 6) Aggregazione multiclass
        if args.multi == "union":
            union_loci = (sorted(set().union(*per_class_selected_loci.values())) if per_class_selected_loci else [])
            logger.info(f"[AGGR] union loci across classes: {len(union_loci)}")
            # Controllo interno: nessun duplicato nella union
            assert len(union_loci) == len(set(union_loci)), ("[AGGR] union_loci contains duplicates, which should be impossible.")
            # Tutti i loci devono esistere come colonne del cgMLST
            missing = [l for l in union_loci if l not in cg.columns]
            if missing:
                logger.error(f"[AGGR] Some union loci not found in cgMLST columns: "
                             f"{missing[:10]} (showing up to 10).")
                sys.exit(3)
        else:
            union_loci = []
        del dist_aligned_path
    # 7) Output finali (fuori dal tmpdir)
    # 7a) filtered_subset_cgMLST.csv (solo per 'union')
    if args.multi == "union":
        keep_set = set(union_loci)
        keep_cols = [c for c in cg.columns if c in keep_set]
        filtered = cg.loc[:, keep_cols].copy()
        filtered.insert(0, "ceppoID", filtered.index.astype(str))
        filtered.index = pd.RangeIndex(len(filtered))
        filtered_out = os.path.join(args.out, "filtered_subset_cgMLST.csv")
        filtered.to_csv(filtered_out, index=False)
        logger.info(f"[SAVE] {filtered_out} rows={len(filtered)} cols={len(keep_cols)}")
        del keep_set, keep_cols, filtered
    else:
        logger.info("[SAVE] 'per_class' strategy: filtered_subset_cgMLST.csv not generated.")
    # 7b) selected_features.obj (pickle)
    obj_out = os.path.join(args.out, "selected_features.obj")
    payload = {
        "params": {
            "pyseer": args.pyseer,
            "block_size": args.block_size,
            "workers": args.workers,
            "cpu_per_block": args.cpu,
            "mds_components": args.mds,
            "fdr_q": args.fdr,
            "maf_min": args.maf,
            "multi_strategy": args.multi,
            "classes": classes,
            "alpha_patterns": getattr(args, "alpha_patterns", None),
            "use_patterns_bonferroni": getattr(args, "use_patterns_bonferroni", False),
            "k_top": getattr(args, "k", None),
        },
        "per_class": {
            cls: {
                "selected_variants": per_class_selected_variants.get(cls, []),
                "selected_loci": per_class_selected_loci.get(cls, []),
            }
            for cls in classes
        },
        "union_selected_loci": union_loci,
    }
    with open(obj_out, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"[SAVE] {obj_out} (per_class + union)")
    # 7c) results.txt
    results_out = os.path.join(args.out, "results.txt")
    with open(results_out, "w", encoding="utf-8") as f:
        f.write("# GWAS Selector (pyseer wrapper) — Multiclass OVR\n")
        f.write(f"samples={len(cg)} loci_in={cg.shape[1]} classes={classes}\n")
        if getattr(args, "use_patterns_bonferroni", False) and bonf_p_thr is not None:
            f.write(f"Bonferroni pattern-based threshold (alpha_patterns={args.alpha_patterns}): "
                    f"p_thr={bonf_p_thr:.3e}\n")
        f.write(f"FDR_threshold={args.fdr}\n")
        if getattr(args, "k", None) is not None:
            f.write(f"topK_per_class={args.k}\n")
        if args.multi == "union":
            f.write(f"Union selected loci: {len(union_loci)}\n")
        f.write("\n")
        for cls in classes:
            df_cls = per_class_results[cls]
            sel_vars = set(per_class_selected_variants[cls])
            sel_df = df_cls[df_cls["variant"].isin(sel_vars)].copy()
            if not sel_df.empty:
                sel_df = sel_df.sort_values(["qval", "pval"])
            stats = per_class_stats.get(cls, {})
            f.write(f"## Class '{cls}'\n")
            f.write(f"  significant_variants_after_threshold: {stats.get('n_sig', 0)}\n")
            f.write(f"  final_selected_loci: {stats.get('n_loci_final', 0)}\n")
            f.write(f"  final_selected_variants: {len(sel_vars)}\n")
            f.write("variant\tpval\tqval\n")
            for _, r in sel_df.head(200).iterrows():
                f.write(f"{r['variant']}\t{r['pval']:.3e}\t{r['qval']:.3e}\n")
            f.write("\n")
        del df_cls, sel_vars, sel_df, stats
    logger.info(f"[SAVE] {results_out}")
    dt = time.perf_counter() - t0
    logger.info(f"[END] done in {dt/60:.2f} min | {mem_usage_str()}")

if __name__ == "__main__":
    main()


# python3 GWAS/GWAS_based_selector.py GWAS/GWAS_INPUT/3k_subset_cgMLST_monoID.csv GWAS/GWAS_INPUT/3k_subset_phenotypes_monoID.csv GWAS/GWAS_INPUT/3k_distance_clean.tsv --out GWAS/GWAS_OUTPUT/dflt --log GWAS/GWAS_OUTPUT/dflt/log.txt --pyseer pyseer --id-col id --phen-col target_IZSAM --block-size 1000 --workers 3 --cpu 3 --mds 10 --maf 0.001 --use-patterns-bonferroni --alpha-patterns 0.001 --min-case-count 5 --min-control-count 5 --fdr 0.001 --multi union

