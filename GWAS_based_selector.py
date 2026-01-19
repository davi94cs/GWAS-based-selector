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
import shlex # Shell utilities: safely split/join command-line strings, escape arguments for shell
import re  # Regular expressions: searching, matching, and replacing text with patterns
import sys  # Access to Python runtime details: command-line args (sys.argv), standard I/O streams, exit codes, interpreter info
import time  # Time utilities: sleep, timestamps, simple timing/benchmarking
import math  # Fast, low-level math functions/constants (e.g., sqrt, log, pi)
import hashlib  # Hashing algorithms (MD5, SHA1, SHA256) for data integrity checks and unique IDs
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

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for a pyseer-first cgMLST multiclass (OVR) GWAS wrapper.

    Design goals (enforced by defaults):
    1) Pyseer-first minimal default:
       - Running the tool with no optional flags performs only the preprocessing
         needed to run pyseer with *pyseer defaults* (no wrapper-side selection).
       - Wrapper-side filters/selection (case/control QC, BH-FDR, pattern Bonferroni,
         topK) are OFF by default and must be explicitly enabled.

    2) Reproducibility and auditability:
       - Every optional behaviour is controlled by an explicit flag and will be
         recorded in logs/results (handled elsewhere).
       - Stable output artifacts are always written under --out (default: selector_out).

    3) Flexibility:
       - Users can change pyseer parameters either via typed flags (common ones)
         or via a generic pass-through (--pyseer-extra).

    Notes on biological correctness:
    - Allele frequency and missingness filtering should be handled by pyseer
      (e.g., --min-af/--max-missing). The wrapper must not implement its own
      MAF filter in pyseer-first mode, to avoid discrepancies due to missing handling.
    - Pattern-based Bonferroni (if enabled) uses pyseer native pattern hashing
      (--output-patterns) and the official count_patterns.py utility.
    - RTAB (--pres) must be strictly binary in pyseer; missing cgMLST calls are
      encoded as 0. To mitigate bias, an optional locus-level missingness QC
      can be enabled via --max-locus-missing.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    ap = argparse.ArgumentParser(description="Pyseer-first multiclass (OVR) GWAS on cgMLST using a square distance matrix.")
    # Positional inputs
    ap.add_argument("cgmlst_csv", help="cgMLST table (sampleID + loci categorical/allelic values)")
    ap.add_argument("phen_csv", help="Phenotypes table (sampleID + phenotype category)")
    ap.add_argument("distance_tsv", help="Square distance matrix (sampleID x sampleID; diagonal=0)")
    # Orchestration / performance
    ap.add_argument("--pyseer", default="pyseer", help="Pyseer executable (default: pyseer in PATH)")
    ap.add_argument("--block-size", type=int, default=2000,
                    help="Number of allele-specific variants (locus_allele) per RTAB block")
    ap.add_argument("--workers", type=int, default=3,
                    help="Number of parallel pyseer blocks (processes)")
    ap.add_argument("--cpu", type=int, default=3,
                    help="CPU cores per pyseer run (passed to pyseer --cpu)")
    ap.add_argument("--mds", type=int, default=15,
                    help="Number of MDS dimensions for population structure correction "
                         "(passed to pyseer as --max-dimensions)")
    # Output paths (stable artifacts)
    ap.add_argument("--out", default="selector_out",
                    help="Output folder for final artifacts. The tool will always write: "
                         "log.txt, results.txt, filtered_loci.csv, loci.obj in this folder.")
    ap.add_argument("--log", default=None, help="Log file path. If not set, defaults to <out>/log.txt")
    # Intermediate files (minimal policy: keep only patterns + errors when needed)
    ap.add_argument("--utils-dir", default=os.path.join("GWAS", "utils"),
                    help="Directory for minimal intermediate files generated during the run. "
                         "In this pipeline we keep only: pattern files (if enabled) and error diagnostics (if failures). "
                         "Default: GWAS/utils")
    # Input column naming
    ap.add_argument("--id-col", default="id", help="Sample ID column name in both cgMLST and phenotype files")
    ap.add_argument("--phen-col", default="target_IZSAM", help="Phenotype column name in the phenotype file")
    # Optional QC: locus-level missingness filter (OFF by default)
    ap.add_argument("--max-locus-missing", type=float, default=None,
                    help="Optional QC (OFF by default): drop loci with missingness fraction > this threshold BEFORE RTAB generation. "
                    "Useful because RTAB (--pres) must be binary and missing calls are encoded as 0. Example: 0.05")
    # Wrapper-side optional QC / selection (OFF by default)
    ap.add_argument("--prefilter-mac", type=int, default=None,
                    help="Optional wrapper-side prefilter (OFF by default): keep only allele-specific variants "
                    "with MAC >= this value and <= N-MAC (symmetrical), computed from cgMLST calls. "
                    "This reduces RTAB size and pyseer runtime. Example: --prefilter-mac 10")
    ap.add_argument("--enable-case-control-filter", action="store_true",
                    help="Enable a phenotype-dependent QC filter that drops variants with too few "
                         "carriers in cases or controls (OFF by default; not part of pyseer).")
    ap.add_argument("--min-case-count", type=int, default=5,
                    help="Used only if --enable-case-control-filter is set: minimum number of case "
                         "samples carrying the allele to keep the variant.")
    ap.add_argument("--min-control-count", type=int, default=5,
                    help="Used only if --enable-case-control-filter is set: minimum number of control "
                         "samples carrying the allele to keep the variant.")
    ap.add_argument("--enable-bh", action="store_true",
                    help="Enable Benjamini-Hochberg FDR correction (OFF by default).")
    ap.add_argument("--bh-q", type=float, default=0.05,
                    help="Used only if --enable-bh is set: q-value threshold for significance.")
    ap.add_argument("--topk", type=int, default=None,
                    help="Optional Top-K locus selection per class (OFF by default). "
                         "If set, keep at most K loci per class ranked by best q-value "
                         "(if BH enabled) or best p-value (if BH disabled).")
    ap.add_argument("--enable-pattern-bonferroni", action="store_true",
                    help="Enable pattern-based Bonferroni thresholding using pyseer --output-patterns "
                         "and count_patterns.py (OFF by default).")
    ap.add_argument("--alpha-patterns", type=float, default=None,
                    help="Used only if --enable-pattern-bonferroni is set: alpha for pattern-based "
                         "Bonferroni threshold.")
    ap.add_argument("--count-patterns", default=os.path.join("GWAS", "count_patterns.py"),
                    help="Path to pyseer count_patterns.py (default: GWAS/count_patterns.py). "
                         "Used only if --enable-pattern-bonferroni is set.")
    ap.add_argument("--multi", choices=["union", "per_class"], default="union",
                    help="How to combine selected loci across classes when locus selection is enabled. "
                         "union: output union across classes; per_class: keep per-class only.")
    # Pyseer parameters (typed + pass-through)
    ap.add_argument("--pyseer-min-af", type=float, default=None,
                    help="Pass to pyseer as --min-af (allele frequency filter). If not provided, pyseer default is used.")
    ap.add_argument("--pyseer-max-af", type=float, default=None,
                    help="Pass to pyseer as --max-af. If not provided, pyseer default is used.")
    ap.add_argument("--pyseer-max-missing", type=float, default=None,
                    help="Pass to pyseer as --max-missing. If not provided, pyseer default is used.")
    ap.add_argument("--pyseer-filter-pvalue", type=float, default=None,
                    help="Pass to pyseer as --filter-pvalue (fast prefilter). If not provided, pyseer default is used.")
    ap.add_argument("--pyseer-lrt-pvalue", type=float, default=None,
                    help="Pass to pyseer as --lrt-pvalue (LRT threshold). If not provided, pyseer default is used.")
    ap.add_argument("--pyseer-print-filtered", action="store_true",
                    help="Pass to pyseer as --print-filtered (debug: report filtered variants).")
    ap.add_argument("--pyseer-extra", action="append", default=[],
                    help="Additional arguments passed verbatim to pyseer (repeatable). "
                         "Example: --pyseer-extra \"--lineage-effects\"")
    args = ap.parse_args()
    # Post-parse normalization & checks
    if args.log is None:
        args.log = os.path.join(args.out, "log.txt")
    if args.enable_pattern_bonferroni and args.alpha_patterns is None:
        ap.error("--enable-pattern-bonferroni requires --alpha-patterns to be set.")
    if args.enable_pattern_bonferroni and not os.path.exists(args.count_patterns):
        ap.error(f"count_patterns.py not found at: {args.count_patterns}")
    if args.prefilter_mac is not None and args.prefilter_mac <= 0:
        ap.error("--prefilter-mac must be a positive integer.")
    if args.enable_bh and (args.bh_q is None or not (0.0 < args.bh_q <= 1.0)):
        ap.error("--enable-bh requires --bh-q in (0,1].")
    if args.topk is not None and args.topk <= 0:
        ap.error("--topk must be a positive integer.")
    if args.block_size <= 0:
        ap.error("--block-size must be a positive integer.")
    if args.workers <= 0:
        ap.error("--workers must be a positive integer.")
    if args.cpu <= 0:
        ap.error("--cpu must be a positive integer.")
    if args.max_locus_missing is not None and not (0.0 <= args.max_locus_missing < 1.0):
        ap.error("--max-locus-missing must be in [0,1).")
    return args

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

def setup_logger(log_path: str, level_console: int = logging.INFO, level_file: int = logging.DEBUG) -> logging.Logger:
    """
    Configure and return a process-wide logger that writes to both a log file and stdout.

    Step-by-step:
    1) Ensure the parent directory for `log_path` exists.
    2) Create/retrieve a named logger used throughout the application.
    3) Remove any existing handlers to avoid duplicate logs and to guarantee that
       the current invocation controls the destination file (important for reproducibility).
    4) Attach:
       - a FileHandler (DEBUG-level by default) writing UTF-8 logs to `log_path`
       - a StreamHandler (INFO-level by default) writing to stdout
    5) Disable propagation to the root logger to prevent duplicated messages in
       certain environments.

    Why:
    - Re-running the tool (or importing it in notebooks/tests) can otherwise lead
      to duplicated log lines or logs being written to an unintended previous file.
    - Stable, always-present logs are part of the required artifacts (log.txt).

    Parameters
    ----------
    log_path : str
        Path to the log file to create/overwrite.
    level_console : int
        Logging level for console output (stdout).
    level_file : int
        Logging level for file output.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger("GWAS_selector")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Remove existing handlers to guarantee determinism and avoid duplicate logs.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level_file)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_console)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
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

def safe_label(s: str) -> str:
    """
    Convert an arbitrary class label into a filesystem-safe token.

    - Replaces whitespace with underscores.
    - Removes characters outside [A-Za-z0-9._-].
    - Collapses multiple underscores.
    """
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    s = re.sub(r"_+", "_", s)
    return s or "NA"

def enumerate_variants_per_locus(cg: pd.DataFrame, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Enumerate all observed allele-specific cgMLST variants in "locus_allele" format.

    Step-by-step:
    1) Iterate over cgMLST loci (columns).
    2) For each locus, collect unique, non-missing allele values.
    3) Emit one variant identifier per observed allele as "locus_allele".

    Why:
    - In a pyseer-first workflow, allele frequency and missingness filtering must be
      delegated to pyseer (e.g., --min-af, --max-missing), because pyseer computes
      these quantities on the exact input matrix it tests (RTAB) and with its own
      missingness semantics.
    - The wrapper only needs a deterministic list of candidate variants to build
      RTAB blocks; it should not impose a frequency filter outside pyseer by default.

    Performance notes:
    - Works locus-by-locus, avoiding construction of large intermediate matrices.
    - Uses pandas unique() per locus, which is efficient for cgMLST-sized tables.

    Parameters
    ----------
    cg : pd.DataFrame
        cgMLST table indexed by sample ID, with loci as columns and alleles as values.
        Missing alleles must be represented as NaN.
    logger : logging.Logger or None
        Optional logger for summary stats.

    Returns
    -------
    List[str]
        List of unique observed "locus_allele" identifiers.
    """
    out: List[str] = []
    for locus in cg.columns:
        s = cg[locus]
        if s is None:
            continue
        alleles = s.dropna().astype(str).unique()
        # Deterministic order improves reproducibility (optional but recommended)
        # Sort only within-locus to keep memory low.
        for a in sorted(alleles.tolist()):
            out.append(f"{locus}_{a}")
    if logger is not None:
        logger.info(f"[ENUM] loci={cg.shape[1]} | variants_enumerated={len(out)}")
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

def prefilter_variants_by_mac_per_locus(cg: pd.DataFrame, mac: int, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Enumerate allele-specific variants (locus_allele) and apply a symmetric MAC prefilter.

    A variant (locus_allele) is kept if:
      carriers >= mac AND carriers <= (N - mac),
    where carriers is the number of samples matching the allele, and N is the
    number of samples in the aligned dataset.

    This is an optional, wrapper-side computational prefilter:
    - OFF by default to keep a strict pyseer-first baseline
    - when enabled, reduces RTAB size and pyseer parsing/testing overhead

    Missingness handling:
    - Missing cgMLST calls (NaN) are excluded from allele counts (dropna).
      This is consistent with RTAB encoding where a missing call cannot contribute
      to allele presence and is effectively treated as absence.

    Parameters
    ----------
    cg : pd.DataFrame
        cgMLST table indexed by sample ID, loci as columns, categorical alleles as values.
        Missing values must be NaN.
    mac : int
        Minor allele count threshold (symmetric).
    logger : logging.Logger or None
        Optional logger.

    Returns
    -------
    List[str]
        List of "locus_allele" variants passing the MAC criterion.
    """
    if mac <= 0:
        raise ValueError("mac must be a positive integer.")
    n = int(cg.shape[0])
    if n <= 1:
        return []
    if mac >= n:
        raise ValueError(f"mac must be < number of samples (mac={mac}, N={n}).")
    out: List[str] = []
    total_alleles_observed = 0
    total_variants_kept = 0
    for locus in cg.columns:
        s = cg[locus].dropna()
        if s.empty:
            continue
        counts = s.astype(str).value_counts()
        total_alleles_observed += int(counts.shape[0])
        keep = counts[(counts >= mac) & (counts <= (n - mac))].index.tolist()
        total_variants_kept += len(keep)
        for a in sorted(keep):
            out.append(f"{locus}_{a}")
    if logger is not None:
        logger.info(
            f"[PREFILTER_MAC] mac={mac} | N={n} | loci={cg.shape[1]} | "
            f"alleles_observed={total_alleles_observed} | variants_kept={len(out)}")
    return out

def filter_loci_by_missingness(cg: pd.DataFrame, max_locus_missing: float, logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Optionally filter cgMLST loci by per-locus missingness rate.

    Why:
    - RTAB (--pres) must be binary, so missing cgMLST calls are encoded as 0.
      Filtering loci with high missingness mitigates missingness-driven artifacts.

    Returns
    -------
    (filtered_cg, dropped_loci)
    """
    if not (0.0 <= max_locus_missing < 1.0):
        raise ValueError("max_locus_missing must be in [0, 1).")
    miss_rate = cg.isna().mean(axis=0)
    dropped = miss_rate[miss_rate > max_locus_missing].index.astype(str).tolist()
    cg_filt = cg.drop(columns=dropped) if dropped else cg
    if logger is not None:
        logger.info(
            f"[MISSING_QC] max_locus_missing={max_locus_missing:.3f} | "
            f"dropped_loci={len(dropped)} | kept_loci={cg_filt.shape[1]}")
    return cg_filt, dropped

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
    Write a pyseer-compatible RTAB presence/absence matrix for a block of allele-specific variants.

    Pyseer constraint (RTAB via --pres):
    - The RTAB matrix must be strictly binary (0/1). Missing tokens (e.g. empty, NA)
      are not supported and cause pyseer to raise 'Rtab file not binary' or parsing errors.
    - Therefore, missing cgMLST calls are encoded as 0 (absence).

    Step-by-step:
    1) Validate the presence of `sample_col`.
    2) Group variants by locus and deduplicate alleles to avoid duplicate RTAB rows.
    3) For each locus/allele, compute NA-safe allele presence (missing -> False) and write:
       - "1" if present, "0" otherwise.

    Parameters
    ----------
    cgmlst_df : pd.DataFrame
        DataFrame containing `sample_col` and loci columns. Missing calls must be NaN.
    sample_col : str
        Sample ID column name.
    block_variants : List[str]
        Variants in "locus_allele" format.
    out_path : str
        Output RTAB file path.
    logger : logging.Logger
        Logger instance.
    """
    if sample_col not in cgmlst_df.columns:
        raise ValueError(f"Missing sample column '{sample_col}' in cgMLST dataframe")
    samples = cgmlst_df[sample_col].astype(str).tolist()
    # Defensive: tabs/newlines in sample IDs would break RTAB format.
    if any(("\t" in s) or ("\n" in s) or ("\r" in s) for s in samples):
        bad = [s for s in samples if ("\t" in s) or ("\n" in s) or ("\r" in s)][:5]
        raise ValueError(f"Sample IDs contain tab/newline characters (e.g., {bad}). RTAB would be malformed.")
    # Group alleles by locus; deduplicate within locus.
    locus_to_alleles: Dict[str, List[str]] = {}
    for v in block_variants:
        if "_" not in v:
            continue
        locus, allele = v.split("_", 1)
        locus_to_alleles.setdefault(locus, []).append(allele)
    for locus in list(locus_to_alleles.keys()):
        locus_to_alleles[locus] = sorted(set(locus_to_alleles[locus]))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("variant\t" + "\t".join(samples) + "\n")
        for locus, alleles in locus_to_alleles.items():
            if locus not in cgmlst_df.columns:
                continue
            col = cgmlst_df[locus]
            for allele in alleles:
                present = col.astype("string").eq(allele).fillna(False).to_numpy(dtype=bool)
                bits = np.where(present, "1", "0")  # strictly binary
                f.write(f"{locus}_{allele}\t" + "\t".join(bits.tolist()) + "\n")
    logger.info(f"[RTAB] wrote {out_path} | variants={len(block_variants)} samples={len(samples)}")

def run_pyseer_block(pyseer_bin: str, phen_path: str, distances_square_path: str, mds_components: int, rtab_path: str, out_path: str, cpu_per_block: int,
                     pyseer_extra_args: List[str], logger: logging.Logger, patterns_out_path: Optional[str] = None, utils_error_dir: Optional[str] = None) -> None:
    """
    Run pyseer on one RTAB block and write association results to `out_path`.

    Intermediate files policy:
    - `out_path` is typically inside a TemporaryDirectory (deleted after run).
    - cmd/stderr are written to `utils_error_dir` ONLY if pyseer fails (to save disk space).

    Pattern workflow:
    - If patterns_out_path is provided, pyseer is invoked with:
        --output-patterns <patterns_out_path>
      This should be used only when pattern-based Bonferroni is enabled.

    Parameters
    ----------
    patterns_out_path : Optional[str]
        If provided, enables pyseer native pattern hashing output for this block.
    utils_error_dir : Optional[str]
        If provided, stores cmd/stderr files on failure only.
    """
    cmd = [
        pyseer_bin,
        "--phenotypes", phen_path,
        "--pres", rtab_path,
        "--distances", distances_square_path,
        "--max-dimensions", str(mds_components),
        "--cpu", str(cpu_per_block),
    ]
    if patterns_out_path is not None:
        cmd += ["--output-patterns", patterns_out_path]
    cmd += list(pyseer_extra_args)
    logger.info(f"[BLOCK] run pyseer on {os.path.basename(rtab_path)} → {os.path.basename(out_path)} "
                f"| cpu={cpu_per_block} | {mem_usage_str()}")
    logger.debug(f"[BLOCK] cmd: {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        with open(out_path, "w", encoding="utf-8") as fout:
            res = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        logger.error(f"[BLOCK] pyseer binary not found: {pyseer_bin}")
        raise
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        logger.error(f"[BLOCK] pyseer FAILED in {dt:.2f}s (returncode={res.returncode})")
        # Persist minimal debugging artifacts only on failure
        if utils_error_dir is not None:
            os.makedirs(utils_error_dir, exist_ok=True)
            # Use block directory name as identifier if possible
            block_id = os.path.basename(os.path.dirname(out_path))  # e.g., block_00097
            cmd_path = os.path.join(utils_error_dir, f"{block_id}.cmd.txt")
            err_path = os.path.join(utils_error_dir, f"{block_id}.stderr.txt")
            with open(cmd_path, "w", encoding="utf-8") as f:
                f.write(" ".join(cmd) + "\n")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(res.stderr or "")
            logger.error(f"[BLOCK] debug saved: {err_path} ; {cmd_path}")
        # Print head for quick diagnosis
        head = "\n".join((res.stderr or "").splitlines()[:50])
        if head:
            logger.error("[BLOCK] STDERR (first 50 lines):\n" + head)
        raise RuntimeError("pyseer failed for block")
    try:
        size_mb = os.path.getsize(out_path) / 1024 / 1024
    except Exception:
        size_mb = float("nan")
    logger.info(f"[BLOCK] pyseer OK in {dt:.2f}s out_size={size_mb:.1f}MB | {mem_usage_str()}")

def filter_variants_by_pattern_threshold(cg: pd.DataFrame, block_variants: List[str], y_bin: pd.Series, min_case: int = 5, min_control: int = 5,
                                         logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Optional phenotype-dependent QC filter for allele-specific variants.

    Step-by-step:
    1) Align y_bin to cg index.
    2) For each variant (locus_allele):
       - build presence mask among NON-MISSING samples at that locus
       - compute carriers in cases and controls
       - drop variants that are monomorphic among observed calls or too sparse
         in either group (min_case/min_control)
    3) Return kept variants.

    Why:
    - This filter is NOT part of pyseer and must be OFF by default in pyseer-first
      runs. When enabled, it acts as a guardrail against variants that provide
      essentially no within-group information and can cause unstable estimates
      (e.g., near-complete separation).
    - Missing cgMLST calls are excluded from carrier counts, preventing technical
      missingness from being interpreted as allele absence.

    Notes:
    - If you want strict pyseer equivalence, do not enable this filter; use pyseer
      frequency filters (e.g., --min-af) instead.

    Returns
    -------
    List[str]
        Variants that pass the QC thresholds.
    """
    y_bin = y_bin.loc[cg.index]
    case_mask = (y_bin.values == 1)
    control_mask = (y_bin.values == 0)
    keep: List[str] = []
    n_total = len(cg)
    cols = cg.columns
    for v in block_variants:
        if "_" not in v:
            continue
        locus, allele = v.split("_", 1)
        if locus not in cols:
            continue
        col = cg[locus]
        non_missing = ~col.isna()
        if not non_missing.any():
            continue
        col_nm = col[non_missing].astype(str).to_numpy()
        present_nm = (col_nm == allele)
        total_count = int(present_nm.sum())
        n_obs = int(non_missing.sum())
        # Monomorphic among observed calls => no information
        if total_count == 0 or total_count == n_obs:
            continue
        # Map case/control on observed-only subset
        case_nm = case_mask[non_missing.to_numpy()]
        control_nm = control_mask[non_missing.to_numpy()]
        case_count = int((present_nm & case_nm).sum())
        control_count = int((present_nm & control_nm).sum())
        if case_count < min_case or control_count < min_control:
            continue
        keep.append(v)
    if logger is not None:
        logger.info(f"[THRESH] block_variants={len(block_variants)} → kept={len(keep)} "
                    f"(min_case={min_case}, min_control={min_control}, N={n_total})")
    return keep

def run_ovr_gwas_for_class(label: str, y_bin: pd.Series, cg: pd.DataFrame, distances_square_path: str, all_variants: List[str], pyseer_bin: str,
                           mds_components: int, block_size: int, workers: int, cpu_per_block: int, pyseer_extra_args: List[str], enable_case_control_filter: bool,
                           min_case_count: int, min_control_count: int, enable_pattern_bonferroni: bool, utils_run_dir: str, logger: logging.Logger) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Run a one-vs-rest GWAS for one class using pyseer, in blocks.

    Disk policy:
    - RTAB blocks, per-block out.tsv, and phenotype files are created in a TemporaryDirectory
      and deleted automatically at the end of this function.
    - Only essential files are written under utils_run_dir:
      - patterns_all.txt (only if enable_pattern_bonferroni)
      - error diagnostics (stderr/cmd) only if a block fails

    Pattern workflow:
    - If enable_pattern_bonferroni=True, each block is run with pyseer --output-patterns
      (temp file), then all per-block pattern files are concatenated into a single
      shell-safe file under <utils_run_dir>/patterns/OVR_<safe_label>_patterns_all.txt.

    Returns
    -------
    (gwas_df, patterns_all_path)
    """
    label_fs = safe_label(label)
    n_pos = int(y_bin.sum())
    n_neg = int(len(y_bin) - n_pos)
    logger.info(f"[OVR:{label}] start | positives={n_pos} negatives={n_neg}")
    blocks = chunk_list(all_variants, block_size)
    n_blocks = len(blocks)
    logger.info(
        f"[OVR:{label}] n_blocks={n_blocks} (~{block_size} vars/block) "
        f"| case_control_filter={'ON' if enable_case_control_filter else 'OFF'} "
        f"| output_patterns={'ON' if enable_pattern_bonferroni else 'OFF'}")
    patterns_dir = os.path.join(utils_run_dir, "patterns")
    errors_dir = os.path.join(utils_run_dir, "errors")
    if enable_pattern_bonferroni:
        os.makedirs(patterns_dir, exist_ok=True)
    # Precompute once
    cg_reset = cg.reset_index().rename(columns={cg.index.name or "index": "ceppoID"})
    if "ceppoID" not in cg_reset.columns and "index" in cg_reset.columns:
        cg_reset = cg_reset.rename(columns={"index": "ceppoID"})
    with tempfile.TemporaryDirectory() as tmpdir:
        # phenotype file for this class (temp) - use label_fs to avoid spaces in file name
        phen_path = os.path.join(tmpdir, f"phen_{label_fs}.tsv")
        phen_df = pd.DataFrame({"sample": y_bin.index.astype(str), "phenotype": y_bin.astype(int).values})
        phen_df.to_csv(phen_path, sep="\t", header=True, index=False)
        del phen_df
        out_paths: List[str] = []
        block_pattern_paths: List[str] = []
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for bi, block_variants in enumerate(blocks):
                if enable_case_control_filter:
                    filtered_block_variants = filter_variants_by_pattern_threshold(
                        cg=cg,
                        block_variants=block_variants,
                        y_bin=y_bin,
                        min_case=min_case_count,
                        min_control=min_control_count,
                        logger=logger)
                else:
                    filtered_block_variants = block_variants
                if not filtered_block_variants:
                    continue
                bdir = os.path.join(tmpdir, f"block_{bi:05d}")
                os.makedirs(bdir, exist_ok=True)
                rtab_path = os.path.join(bdir, "block.rtab")
                write_rtab_block_stream(
                    cgmlst_df=cg_reset,
                    sample_col="ceppoID",
                    block_variants=filtered_block_variants,
                    out_path=rtab_path,
                    logger=logger)
                out_path = os.path.join(bdir, "out.tsv")
                out_paths.append(out_path)
                patterns_out_path = None
                if enable_pattern_bonferroni:
                    patterns_out_path = os.path.join(bdir, "patterns.txt")
                    block_pattern_paths.append(patterns_out_path)
                futures.append(
                    ex.submit(
                        run_pyseer_block,
                        pyseer_bin,
                        phen_path,
                        distances_square_path,
                        mds_components,
                        rtab_path,
                        out_path,
                        cpu_per_block,
                        pyseer_extra_args,
                        logger,
                        patterns_out_path,
                        errors_dir))
            if not futures:
                raise ValueError(f"[OVR:{label}] No blocks submitted to pyseer (0 variants after optional QC).")
            for fut in as_completed(futures):
                fut.result()
        # Concatenate patterns into a persistent file in utils (only if enabled)
        patterns_all_path: Optional[str] = None
        if enable_pattern_bonferroni:
            patterns_all_path = os.path.join(patterns_dir, f"OVR_{label_fs}_patterns_all.txt")
            n_lines_written = 0
            with open(patterns_all_path, "w", encoding="utf-8", newline="\n") as w:
                for p in block_pattern_paths:
                    if not os.path.exists(p):
                        continue
                    with open(p, "r", encoding="utf-8", errors="replace") as r:
                        for line in r:
                            s = line.strip()
                            if not s:
                                continue
                            w.write(s + "\n")
                            n_lines_written += 1
            logger.info(f"[OVR:{label}] patterns saved: {patterns_all_path} | lines={n_lines_written}")
        # Merge per-block outputs
        gwas_list: List[pd.DataFrame] = []
        for pth in out_paths:
            df = pd.read_csv(pth, sep="\t")
            cols = {c.lower(): c for c in df.columns}
            vcol = cols.get("variant", cols.get("pattern"))
            pcol = None
            for key in ("lrt-pvalue", "pvalue", "pval"):
                if key in cols:
                    pcol = cols[key]
                    break
            if vcol is None or pcol is None:
                raise ValueError(f"[OVR:{label}] missing variant/pvalue columns in {pth} (columns={list(df.columns)})")
            gwas_list.append(df[[vcol, pcol]].rename(columns={vcol: "variant", pcol: "pval"}))
        if not gwas_list:
            raise ValueError(f"[OVR:{label}] no GWAS results to combine (empty block outputs).")
        gwas = (
            pd.concat(gwas_list, axis=0, ignore_index=True)
            .dropna(subset=["pval"])
            .sort_values("pval", ascending=True))
        return gwas, patterns_all_path

### STATISTICAL FILTER ---> BH_FDR, COUNT_PATTERNS ###

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Compute Benjamini–Hochberg (BH) adjusted p-values (q-values) for FDR control.

    Step-by-step:
    1) Convert input to a 1D float array.
    2) Identify finite p-values; NaN/inf entries are left as NaN in the output.
    3) Sort finite p-values increasingly and assign ranks (1..m).
    4) Compute BH q-values: q_i = p_i * m / rank_i.
    5) Enforce monotonicity by applying a reverse cumulative minimum.
    6) Map q-values back to the original order and clip to [0, 1].

    Why:
    - BH controls the expected false discovery rate under standard assumptions and
      is commonly used in microbial GWAS post-processing to obtain q-values.

    Parameters
    ----------
    pvals : np.ndarray
        1D array-like of raw p-values.

    Returns
    -------
    np.ndarray
        Array of q-values with the same shape as input. Non-finite p-values yield NaN q-values.
    """
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("bh_fdr expects a 1D array of p-values.")
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    p_fin = p[finite]
    m = p_fin.size
    if m == 0:
        return out
    order = np.argsort(p_fin)
    ranked = p_fin[order]
    denom = np.arange(1, m + 1, dtype=float)
    q = ranked * m / denom
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    q_back = np.empty_like(q)
    q_back[order] = q
    out[finite] = q_back
    return out

def run_count_patterns(count_patterns_path: str, patterns_all_path: str, alpha: float, out_dir: str, logger: logging.Logger, tag: Optional[str] = None) -> Optional[float]:
    """
    Compute pattern-based Bonferroni threshold using pyseer's official count_patterns.py.

    Robust behaviour:
    - Returns None (instead of raising) when a threshold is not definable, e.g.:
      * patterns file is empty
      * count_patterns.py reports Patterns: 0
      * count_patterns.py crashes with ZeroDivisionError (alpha / 0)

    Saves stdout/stderr to out_dir (small, useful, reproducible). If `tag` is provided,
    outputs are saved as count_patterns_<tag>.stdout.txt / .stderr.txt to avoid overwrites.

    Parameters
    ----------
    count_patterns_path : str
        Path to count_patterns.py.
    patterns_all_path : str
        Path to concatenated patterns file for a class.
    alpha : float
        Family-wise alpha.
    out_dir : str
        Directory where stdout/stderr logs will be saved.
    logger : logging.Logger
        Logger instance.
    tag : Optional[str]
        Optional label used to disambiguate outputs per class (should be filesystem-safe).

    Returns
    -------
    Optional[float]
        Parsed p-value threshold, or None if not definable.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    if not os.path.exists(count_patterns_path):
        raise FileNotFoundError(f"count_patterns.py not found: {count_patterns_path}")
    if not os.path.exists(patterns_all_path):
        raise FileNotFoundError(f"patterns file not found: {patterns_all_path}")
    # If the patterns file is empty, no threshold can be computed.
    try:
        if os.path.getsize(patterns_all_path) == 0:
            logger.warning(f"[BONF] patterns file is empty: {patterns_all_path}. Skipping.")
            return None
    except Exception:
        # If size check fails for any reason, we proceed and let count_patterns decide.
        pass
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, count_patterns_path, "--alpha", str(alpha), patterns_all_path]
    logger.info(f"[BONF] running count_patterns.py | cmd={' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    suffix = f"_{tag}" if tag else ""
    stdout_path = os.path.join(out_dir, f"count_patterns{suffix}.stdout.txt")
    stderr_path = os.path.join(out_dir, f"count_patterns{suffix}.stderr.txt")
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(res.stdout or "")
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(res.stderr or "")
    txt = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    # Handle known failure mode: alpha / 0, or internal sort/path issues.
    if res.returncode != 0:
        # If it's a "division by zero" / Patterns 0 case, skip gracefully.
        if ("ZeroDivisionError" in err) or ("Patterns:\t0" in txt) or ("Patterns: 0" in txt):
            logger.warning(f"[BONF] count_patterns.py returned no valid patterns (Patterns=0). Skipping. See: {stderr_path}")
            return None
        logger.error(f"[BONF] count_patterns.py failed (returncode={res.returncode})")
        logger.error(f"[BONF] see: {stderr_path}")
        raise RuntimeError("count_patterns.py failed")
    # If stdout is empty, cannot parse threshold.
    if not txt:
        logger.warning(f"[BONF] count_patterns.py produced empty stdout. Skipping. See: {stdout_path}")
        return None
    # If it explicitly reports 0 patterns, skip.
    if ("Patterns:\t0" in txt) or ("Patterns: 0" in txt):
        logger.warning(f"[BONF] Patterns=0 according to count_patterns.py. Skipping. See: {stdout_path}")
        return None
    # Parse threshold (targeted parse first)
    m = re.search(r"threshold[^0-9eE.+-]*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", txt, flags=re.IGNORECASE)
    if m:
        thr = float(m.group(1))
        logger.info(f"[BONF] parsed threshold={thr:.3e}")
        return thr
    # Fallback: last float-like token
    nums = re.findall(r"([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", txt)
    if not nums:
        logger.warning(f"[BONF] Could not parse numeric threshold from stdout. Skipping. See: {stdout_path}")
        return None
    thr = float(nums[-1])
    logger.info(f"[BONF] parsed threshold={thr:.3e} (fallback)")
    return thr

### MAIN (FULL PROCESS + OUTPUT AGGREGATOR) ###

def main() -> None:
    """
    Main entry point: runs multiclass OVR GWAS via pyseer and produces stable outputs in args.out.

    Disk policy:
    - args.out: final artifacts (log.txt, results.txt, filtered_<cgmlst>.csv, selected_loci.obj)
    - args.utils_dir: minimal intermediates only:
      - patterns files (only if pattern Bonferroni enabled)
      - error diagnostics (only if a block fails)

    Output policy (always produces a usable filtered cgMLST):
    - Define three locus sets:
        * selected_loci_union: loci passing wrapper-side selection (BH/bonf/topK)
        * tested_loci_union: loci that appear in pyseer outputs (after any upstream filters)
        * input_loci: loci present in cgMLST (post missingness QC, if enabled)
    - Loci used for filtered cgMLST are chosen as:
        selected_loci_union (if non-empty) ->
        tested_loci_union   (else if non-empty) ->
        input_loci          (fallback)
      This guarantees filtered_<cgmlst>.csv is never "empty" and is meaningful even when
      no explicit post-GWAS selection is enabled.
    """
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if getattr(args, "log", None) in (None, ""):
        args.log = os.path.join(args.out, "log.txt")
    logger = setup_logger(args.log)
    t0 = time.perf_counter()
    # Per-run utils dir (minimal content by design)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    utils_run_dir = os.path.join(args.utils_dir, f"run_{run_id}")
    os.makedirs(utils_run_dir, exist_ok=True)
    # Log pyseer version (reviewer-proof)
    try:
        ver = subprocess.run([args.pyseer, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f"[PYSEER_VERSION] {(ver.stdout.strip() or ver.stderr.strip())}")
    except Exception as e:
        logger.warning(f"[PYSEER_VERSION] could not query pyseer version: {e}")
    logger.info(f"[START] cgmlst={args.cgmlst_csv} phen={args.phen_csv} dist={args.distance_tsv}")
    logger.info(
        f"[PARAMS] pyseer={args.pyseer} block_size={args.block_size} workers={args.workers} "
        f"cpu_per_block={args.cpu} mds_dims={args.mds} multi={args.multi} | {mem_usage_str()}")
    logger.info(
        f"[WRAPPER] prefilter_mac={getattr(args, 'prefilter_mac', None)} | "
        f"case_control={'ON' if args.enable_case_control_filter else 'OFF'} "
        f"(min_case={args.min_case_count}, min_control={args.min_control_count}) | "
        f"BH={'ON' if args.enable_bh else 'OFF'} (q<={args.bh_q}) | "
        f"pattern_bonf={'ON' if args.enable_pattern_bonferroni else 'OFF'} "
        f"(alpha={args.alpha_patterns}, count_patterns={args.count_patterns}) | "
        f"topk={args.topk} | max_locus_missing={args.max_locus_missing}")
    # Build pyseer extra args (pyseer-first)
    pyseer_extra_args: List[str] = []
    if args.pyseer_min_af is not None:
        pyseer_extra_args += ["--min-af", str(args.pyseer_min_af)]
    if args.pyseer_max_af is not None:
        pyseer_extra_args += ["--max-af", str(args.pyseer_max_af)]
    if args.pyseer_max_missing is not None:
        pyseer_extra_args += ["--max-missing", str(args.pyseer_max_missing)]
    if args.pyseer_filter_pvalue is not None:
        pyseer_extra_args += ["--filter-pvalue", str(args.pyseer_filter_pvalue)]
    if args.pyseer_lrt_pvalue is not None:
        pyseer_extra_args += ["--lrt-pvalue", str(args.pyseer_lrt_pvalue)]
    if args.pyseer_print_filtered:
        pyseer_extra_args += ["--print-filtered"]
    for extra in getattr(args, "pyseer_extra", []) or []:
        pyseer_extra_args += shlex.split(extra)
    logger.info(f"[PYSEER] extra args: {' '.join(pyseer_extra_args) if pyseer_extra_args else '(pyseer defaults)'}")
    # Load and align cgMLST + phenotypes
    cg = pd.read_csv(args.cgmlst_csv)
    ph = pd.read_csv(args.phen_csv)
    if args.id_col not in cg.columns:
        logger.error(f"[IO] cgMLST missing ID column '{args.id_col}'. Columns: {list(cg.columns)}")
        sys.exit(2)
    if args.id_col not in ph.columns:
        logger.error(f"[IO] phenotypes missing ID column '{args.id_col}'. Columns: {list(ph.columns)}")
        sys.exit(2)
    if args.phen_col not in ph.columns:
        logger.error(f"[IO] phenotypes missing phenotype column '{args.phen_col}'. Columns: {list(ph.columns)}")
        sys.exit(2)
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
        logger.error("[IO] No common samples between cgMLST and phenotypes.")
        sys.exit(2)
    cg = cg.loc[common]
    y_cat = y_cat.loc[common]
    classes = sorted(y_cat.unique().tolist())
    logger.info(f"[IO] aligned samples={len(common)} loci={cg.shape[1]} classes={classes} | {mem_usage_str()}")
    # Optional QC: filter loci by missingness
    dropped_loci_missing: List[str] = []
    if args.max_locus_missing is not None:
        cg, dropped_loci_missing = filter_loci_by_missingness(
            cg=cg, max_locus_missing=args.max_locus_missing, logger=logger)
    # Distance matrix alignment (pyseer --distances)
    D_full = read_square_distance_matrix(args.distance_tsv, logger)
    D = subset_order_distance(D_full, cg.index, logger)
    del D_full
    # Distance matrix file path for pyseer: keep it TEMP (not stored in utils to save disk)
    with tempfile.TemporaryDirectory() as tmpdir:
        dist_aligned_path = os.path.join(tmpdir, "distance_aligned.tsv")
        D.to_csv(dist_aligned_path, sep="\t")
        del D
        # Enumerate variants (optionally MAC-prefiltered)
        if getattr(args, "prefilter_mac", None) is not None:
            all_variants = prefilter_variants_by_mac_per_locus(cg=cg, mac=args.prefilter_mac, logger=logger)
        else:
            all_variants = enumerate_variants_per_locus(cg=cg, logger=logger)
        selection_enabled = bool(args.enable_bh or args.enable_pattern_bonferroni or (args.topk is not None))
        if not selection_enabled:
            logger.info("[SELECT] No wrapper-side selection enabled. Will still produce filtered cgMLST based on tested loci.")
        per_class_results: Dict[str, pd.DataFrame] = {}
        per_class_selected_variants: Dict[str, List[str]] = {}
        per_class_selected_loci: Dict[str, List[str]] = {}
        per_class_stats: Dict[str, Dict[str, int]] = {}
        per_class_bonf_thr: Dict[str, Optional[float]] = {}
        for cls in classes:
            y_bin = (y_cat == cls).astype(int)
            gwas_cls, patterns_all_path = run_ovr_gwas_for_class(
                label=cls,
                y_bin=y_bin,
                cg=cg,
                distances_square_path=dist_aligned_path,
                all_variants=all_variants,
                pyseer_bin=args.pyseer,
                mds_components=args.mds,
                block_size=args.block_size,
                workers=args.workers,
                cpu_per_block=args.cpu,
                pyseer_extra_args=pyseer_extra_args,
                enable_case_control_filter=args.enable_case_control_filter,
                min_case_count=args.min_case_count,
                min_control_count=args.min_control_count,
                enable_pattern_bonferroni=args.enable_pattern_bonferroni,
                utils_run_dir=utils_run_dir,
                logger=logger)
            del y_bin
            per_class_results[cls] = gwas_cls
            # Initialize stats consistently
            if gwas_cls is None or gwas_cls.empty:
                per_class_selected_variants[cls] = []
                per_class_selected_loci[cls] = []
                per_class_stats[cls] = {"n_tests": 0, "n_sig": 0, "n_loci_final": 0}
                per_class_bonf_thr[cls] = None
                continue
            per_class_stats[cls] = {"n_tests": int(len(gwas_cls)), "n_sig": 0, "n_loci_final": 0}
            if not gwas_cls["pval"].between(0, 1).all():
                logger.error(f"[OVR:{cls}] p-values out of [0,1].")
                sys.exit(3)
            # Pattern-based Bonferroni per class (optional)
            bonf_thr = None
            if args.enable_pattern_bonferroni:
                if patterns_all_path is None:
                    logger.error(f"[OVR:{cls}] pattern bonf enabled but patterns_all_path is None.")
                    sys.exit(3)
                bonf_thr = run_count_patterns(
                    count_patterns_path=args.count_patterns,
                    patterns_all_path=patterns_all_path,
                    alpha=args.alpha_patterns,
                    out_dir=os.path.join(utils_run_dir, "patterns"),
                    logger=logger,
                    tag=safe_label(cls))
                if bonf_thr is None:
                    logger.warning(
                        f"[OVR:{cls}] pattern Bonferroni skipped (no valid patterns). Proceeding without Bonferroni threshold.")
            per_class_bonf_thr[cls] = bonf_thr
            # BH q-values (optional)
            if args.enable_bh:
                gwas_cls = gwas_cls.sort_values("pval", ascending=True)
                gwas_cls["qval"] = bh_fdr(gwas_cls["pval"].values)
                per_class_results[cls] = gwas_cls
            # If no wrapper selection is enabled, keep empty selection for this class
            if not selection_enabled:
                per_class_selected_variants[cls] = []
                per_class_selected_loci[cls] = []
                continue
            # Apply selection filters (bonf and/or BH)
            mask = np.ones(len(gwas_cls), dtype=bool)
            if bonf_thr is not None:
                mask &= (gwas_cls["pval"].values <= bonf_thr)
            if args.enable_bh:
                mask &= (gwas_cls["qval"].values <= args.bh_q)
            df_sig = gwas_cls.loc[mask].copy()
            n_sig = int(len(df_sig))
            per_class_stats[cls]["n_sig"] = n_sig
            if n_sig == 0:
                per_class_selected_variants[cls] = []
                per_class_selected_loci[cls] = []
                per_class_stats[cls]["n_loci_final"] = 0
                continue
            df_sig["locus"] = df_sig["variant"].map(variant_to_locus)
            # Rank loci by best evidence (min qval if available else min pval)
            if args.enable_bh and "qval" in df_sig.columns:
                agg = df_sig.groupby("locus", sort=False).agg(best=("qval", "min"))
            else:
                agg = df_sig.groupby("locus", sort=False).agg(best=("pval", "min"))
            agg = agg.reset_index().sort_values(["best", "locus"], ascending=[True, True])
            loci_sorted = agg["locus"].tolist()
            # Top-K (optional)
            if args.topk is not None:
                loci_kept = set(loci_sorted[: args.topk])
            else:
                loci_kept = set(loci_sorted)
            df_final = df_sig[df_sig["locus"].isin(loci_kept)].copy()
            per_class_selected_variants[cls] = df_final["variant"].tolist()
            per_class_selected_loci[cls] = sorted(loci_kept)
            per_class_stats[cls]["n_loci_final"] = int(len(loci_kept))
        # Derive locus sets for outputs
        # SELECTED loci (wrapper decision; may be empty)
        selected_loci_union = sorted(set().union(*per_class_selected_loci.values())) if per_class_selected_loci else []
        # TESTED loci (pyseer outputs after any upstream filters)
        tested_loci_set: set[str] = set()
        for _cls, _df in per_class_results.items():
            if _df is None or _df.empty:
                continue
            tested_loci_set.update(_df["variant"].map(variant_to_locus).dropna().unique().tolist())
        tested_loci_union = sorted(tested_loci_set)
        # INPUT loci (post missingness QC)
        input_loci = list(cg.columns)
        # Choose basis for filtered cgMLST
        if selected_loci_union:
            filtered_basis = "selected"
            loci_for_filtered = selected_loci_union
        elif tested_loci_union:
            filtered_basis = "tested"
            loci_for_filtered = tested_loci_union
        else:
            filtered_basis = "input"
            loci_for_filtered = input_loci
        logger.info(
            f"[FILTERED_BASIS] basis={filtered_basis} "
            f"| selected_union={len(selected_loci_union)} tested_union={len(tested_loci_union)} "
            f"input={len(input_loci)} final={len(loci_for_filtered)}")
    # Final stable outputs in args.out
    # 1) Write filtered cgMLST table (restricted to loci_for_filtered; never empty by policy)
    in_base = os.path.splitext(os.path.basename(args.cgmlst_csv))[0]
    filtered_cgmlst_path = os.path.join(args.out, f"filtered_{in_base}.csv")
    keep_cols = [c for c in loci_for_filtered if c in cg.columns]
    missing_cols = [c for c in loci_for_filtered if c not in cg.columns]
    if missing_cols:
        logger.warning(
            f"[SAVE] {len(missing_cols)} loci not found in cgMLST columns (showing up to 10): {missing_cols[:10]}")
    cg_filtered = cg.loc[:, keep_cols].copy()
    cg_filtered.insert(0, "ceppoID", cg_filtered.index.astype(str))
    cg_filtered.to_csv(filtered_cgmlst_path, index=False)
    logger.info(f"[SAVE] {filtered_cgmlst_path} samples={cg_filtered.shape[0]} loci={cg_filtered.shape[1]-1}")
    del cg_filtered
    # 2) Write selected_loci.obj (pickle for reuse on external cgMLST datasets)
    selected_obj_path = os.path.join(args.out, "selected_loci.obj")
    payload = {
        "tool": "GWAS_selector_pyseer_first",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "inputs": {
            "cgmlst_csv": args.cgmlst_csv,
            "phen_csv": args.phen_csv,
            "distance_tsv": args.distance_tsv,
        },
        "paths": {
            "out_dir": os.path.abspath(args.out),
            "utils_run_dir": os.path.abspath(utils_run_dir),
            "filtered_cgmlst_csv": os.path.abspath(filtered_cgmlst_path),
        },
        "params": {
            "pyseer": args.pyseer,
            "pyseer_extra_args": pyseer_extra_args,
            "block_size": args.block_size,
            "workers": args.workers,
            "cpu_per_block": args.cpu,
            "mds_components": args.mds,
            "multi_strategy": args.multi,
            "wrapper": {
                "prefilter_mac": getattr(args, "prefilter_mac", None),
                "max_locus_missing": args.max_locus_missing,
                "dropped_loci_missingness": dropped_loci_missing,
                "enable_case_control_filter": args.enable_case_control_filter,
                "min_case_count": args.min_case_count,
                "min_control_count": args.min_control_count,
                "enable_bh": args.enable_bh,
                "bh_q": args.bh_q,
                "enable_pattern_bonferroni": args.enable_pattern_bonferroni,
                "alpha_patterns": args.alpha_patterns,
                "count_patterns": args.count_patterns,
                "topk": args.topk,
            },
            "pyseer_typed": {
                "min_af": args.pyseer_min_af,
                "max_af": args.pyseer_max_af,
                "max_missing": args.pyseer_max_missing,
                "filter_pvalue": args.pyseer_filter_pvalue,
                "lrt_pvalue": args.pyseer_lrt_pvalue,
                "print_filtered": args.pyseer_print_filtered,
            },
        },
        "classes": classes,
        "locus_sets": {
            "filtered_basis": filtered_basis,
            "input_loci": input_loci,
            "tested_loci_union": tested_loci_union,
            "selected_loci_union": selected_loci_union,
            "loci_for_filtered": loci_for_filtered,
        },
        "selection": {
            "selection_enabled": selection_enabled,
            "per_class_bonf_p_thr": per_class_bonf_thr,
            "per_class_selected_loci": per_class_selected_loci,
            "per_class_selected_variants": per_class_selected_variants,
        },
        "stats": per_class_stats,
    }
    with open(selected_obj_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"[SAVE] {selected_obj_path}")
    # 3) Write results.txt (always informative; selected if available, else top tested signals)
    results_path = os.path.join(args.out, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("# GWAS Selector (pyseer-first) — Multiclass OVR\n")
        f.write(f"samples={len(cg)} loci_in={cg.shape[1]} classes={classes}\n")
        f.write(f"cgmlst_input={args.cgmlst_csv}\n")
        f.write(f"phen_input={args.phen_csv}\n")
        f.write(f"distance_input={args.distance_tsv}\n")
        f.write(f"utils_run_dir={utils_run_dir}\n\n")
        f.write("## Parameters\n")
        f.write(f"pyseer={args.pyseer}\n")
        f.write(f"pyseer_extra_args={' '.join(pyseer_extra_args) if pyseer_extra_args else '(pyseer defaults)'}\n")
        f.write(f"block_size={args.block_size} workers={args.workers} cpu_per_block={args.cpu} mds_dims={args.mds}\n")
        f.write(f"prefilter_mac={getattr(args, 'prefilter_mac', None)}\n")
        f.write(f"max_locus_missing={args.max_locus_missing}\n")
        f.write(f"dropped_loci_missingness={len(dropped_loci_missing)}\n")
        f.write(f"case_control_filter={'ON' if args.enable_case_control_filter else 'OFF'} "
                f"(min_case={args.min_case_count}, min_control={args.min_control_count})\n")
        f.write(f"BH={'ON' if args.enable_bh else 'OFF'} (bh_q={args.bh_q})\n")
        f.write(f"pattern_bonferroni={'ON' if args.enable_pattern_bonferroni else 'OFF'} "
                f"(alpha={args.alpha_patterns}, count_patterns={args.count_patterns})\n")
        f.write(f"topk={args.topk}\n")
        f.write(f"multi={args.multi}\n\n")
        f.write("## Locus sets summary\n")
        f.write(f"selection_enabled={selection_enabled}\n")
        f.write(f"filtered_basis={filtered_basis}\n")
        f.write(f"selected_loci_union={len(selected_loci_union)}\n")
        f.write(f"tested_loci_union={len(tested_loci_union)}\n")
        f.write(f"final_loci_for_filtered_csv={len(loci_for_filtered)}\n\n")
        f.write("## Loci used for filtered cgMLST\n")
        for locus in loci_for_filtered[:500]:
            f.write(f"{locus}\n")
        if len(loci_for_filtered) > 500:
            f.write(f"... (truncated, total={len(loci_for_filtered)})\n")
        f.write("\n")
        for cls in classes:
            df_cls = per_class_results.get(cls)
            f.write(f"## Class '{cls}'\n")
            if df_cls is None or df_cls.empty:
                f.write("  (no results)\n\n")
                continue
            stats = per_class_stats.get(cls, {})
            sel_vars = set(per_class_selected_variants.get(cls, []))
            bonf_thr = per_class_bonf_thr.get(cls, None)
            f.write(f"n_tests={stats.get('n_tests', len(df_cls))}\n")
            f.write(f"n_sig_after_filters={stats.get('n_sig', 0)}\n")
            f.write(f"n_loci_final={stats.get('n_loci_final', 0)}\n")
            f.write(f"pattern_bonf_p_thr={bonf_thr}\n")
            f.write(f"final_selected_variants={len(sel_vars)}\n")
            has_q = ("qval" in df_cls.columns)
            f.write("variant\tpval\tqval\n")
            if sel_vars:
                # Selected variants (post filters)
                sel_df = df_cls[df_cls["variant"].isin(sel_vars)].copy()
                sort_cols = ["qval", "pval"] if has_q else ["pval"]
                sel_df = sel_df.sort_values(sort_cols, ascending=True)
                for _, r in sel_df.head(200).iterrows():
                    pval = r["pval"]
                    if has_q:
                        f.write(f"{r['variant']}\t{pval:.3e}\t{r['qval']:.3e}\n")
                    else:
                        f.write(f"{r['variant']}\t{pval:.3e}\tNA\n")
                if len(sel_df) > 200:
                    f.write(f"... (truncated, total={len(sel_df)})\n")
            else:
                # Exploratory: top tested variants by p-value
                top_df = df_cls.sort_values("pval", ascending=True).head(200)
                for _, r in top_df.iterrows():
                    pval = r["pval"]
                    if has_q:
                        f.write(f"{r['variant']}\t{pval:.3e}\t{r['qval']:.3e}\n")
                    else:
                        f.write(f"{r['variant']}\t{pval:.3e}\tNA\n")
                if len(df_cls) > 200:
                    f.write(f"... (truncated, total_tests={len(df_cls)})\n")
            f.write("\n")
    logger.info(f"[SAVE] {results_path}")
    dt = time.perf_counter() - t0
    logger.info(f"[END] done in {dt/60:.2f} min | {mem_usage_str()}")


if __name__ == "__main__":
    main()



