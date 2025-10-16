# Genome-Wide Association Screening with PETSc

## Background

- Imagine you have genetic data from 50,000 patients (some healthy, some sick) and want to test 1 million genetic positions to find which ones correlate with disease. 

- This is essentially testing 1 million hypotheses—computing a statistical score for each position and ranking them. The data is too large for one machine, so we distribute the computation across nodes using MPI.

- Biomedical labs routinely run genome-wide association studies (GWAS) to find variants linked to diseases (e.g., cancer risk, pharmacogenomics).

- Modern cohorts contain millions of genetic variants across hundreds of thousands of patients, which is too large for a single machine.

- MPI lets us split variants (or samples) across nodes while still producing global statistics such as allele frequencies or association scores.

- This is a classic **embarrassingly parallel** problem: each genetic variant can be tested independently, making it ideal for distributed computing.

# Statistical Model

**What We're Computing:** 

- For each genetic position, we have a value of 0, 1, or 2 for each patient (representing how many copies of a variant they have).
- We also have a binary label (0=healthy, 1=sick). 
- The question: does this genetic position appear more frequently in sick patients than healthy ones?

For each variant $j$ with genotypes $g_{ij} \in \{0,1,2\}$ (minor-allele counts) and binary phenotype $y_i \in \{0,1\}$, we compute a chi-square test:
1. Count minor and major alleles in cases vs controls.
2. Form a $2 \times 2$ contingency table of allele counts.
3. Compute the Pearson chi-square statistic: $\chi^2 = \sum \frac{(O - E)^2}{E}$ where $O$ is observed counts and $E$ is expected counts under independence.

**MPI Strategy:** Each rank owns disjoint sets of variants (columns of the genotype matrix), but all ranks need access to the full phenotype vector (which is broadcasted). PETSc distributed vectors hold the chi-square scores so we can locate the top associated variants globally.

## Demo Script (examples/15_petsc_gwas.py)

The script simulates genotypes with random minor-allele frequencies and a binary phenotype. Each MPI rank works on a contiguous block of variants, computes local allele statistics, and stores chi-square values in a PETSc distributed vector.

### How the Code Works

**1. Data Distribution (lines 20-26, 110-111)**

The `compute_local_range` function divides variants across ranks using a balanced strategy that handles remainders:
```python
start_idx, end_idx = compute_local_range(args.variants, size, rank)
local_variants = end_idx - start_idx
```
Example: 50,000 variants across 4 ranks → each rank owns ~12,500 variants (rank 0: 0-12499, rank 1: 12500-24999, etc.)

**2. Phenotype Broadcasting (lines 106-108)**

All ranks need the full phenotype vector (case/control status for all patients) since each rank's variants must be tested against all patients:
```python
phenotype = phen_rng.binomial(1, 0.5, size=args.samples)  # Rank 0 generates
comm.Bcast(phenotype, root=0)  # Broadcast to all ranks
```
This is a one-time $O(N)$ communication where $N$ is the number of patients.

**3. Local Genotype Generation (lines 29-41)**

Each rank independently generates its assigned variants using offset seeds to ensure reproducibility:
```python
rng = np.random.default_rng(seed + start_idx * 97)  # Unique seed per rank
genotypes[:, j] = rng.binomial(2, maf, size=num_samples)  # 0, 1, or 2 copies
```
This avoids storing or transferring the entire genotype matrix—each rank generates only what it needs.

**4. Chi-Square Computation (lines 44-92)**

This is pure local computation with no MPI communication. For each variant owned by the rank:
- Split patients into cases and controls using the phenotype vector
- Count minor/major alleles in each group: $minor_{cases} = \sum_{i \in cases} g_{ij}$
- Build a $2 \times 2$ contingency table:

  |           | Minor Allele | Major Allele |
  |-----------|-------------|--------------|
  | Cases     | $O_{11}$    | $O_{12}$    |
  | Controls  | $O_{21}$    | $O_{22}$    |

- Compute expected counts under independence: $E_{ij} = \frac{(\text{row total}) \times (\text{col total})}{\text{grand total}}$
- Calculate chi-square: $\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$

Higher $\chi^2$ indicates stronger association between the variant and disease status.

**5. Distributed Vector Assembly (lines 123-133)**

Each rank stores its local chi-square results in a PETSc distributed vector:
```python
chi2_vec = PETSc.Vec().createMPI(args.variants, bsize=local_variants, comm=PETSc.COMM_WORLD)
chi2_arr = chi2_vec.getArray()  # Get local portion
chi2_arr[:] = chi2_local  # Fill with computed values
chi2_vec.assemblyBegin()
chi2_vec.assemblyEnd()  # Finalize parallel assembly
```
The vector is distributed: rank 0 owns entries [0:12500), rank 1 owns [12500:25000), etc.

**6. Global Maximum Finding (lines 135-144)**

PETSc's `Vec.max()` performs an `MPI_Allreduce` to find the global maximum across all ranks:
```python
max_val, max_idx = chi2_vec.max()  # Returns (value, global index)
```
Then we determine which rank owns this index and retrieve its minor allele frequency using ownership ranges and broadcasting.

**7. Top-K Reporting (lines 151-161)**

Each rank sorts its local variants and sends top-K to rank 0:
```python
local_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by chi-square
gathered = comm.gather(top_local, root=0)  # Gather to rank 0
```
Rank 0 merges and re-sorts to find the global top-K variants.

### Key MPI Patterns Demonstrated

1. **Broadcast:** Phenotype vector shared across all ranks (one-to-all communication)
2. **Embarrassingly parallel:** Chi-square computation is fully independent per variant
3. **Distributed data structures:** PETSc vectors provide natural abstraction for distributed results
4. **Collective reductions:** `Vec.max()` and `comm.gather()` aggregate results across ranks
5. **Ownership-based access:** Each rank queries which indices it owns to retrieve metadata

This demonstrates the core pattern used in production GWAS pipelines that scale to millions of variants across clusters.

## Scaling Notes

Swap the synthetic generator for real VCF or PLINK readers; the PETSc/MPI communication remains unchanged. Add covariates or logistic regression by extending the local gradient to TAO. Run on clusters with `mpirun -n 32 python examples/15_petsc_gwas.py --samples 50000 --variants 1000000`.
