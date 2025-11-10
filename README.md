# FORGE

FORGE is an automation toolkit for running high-throughput quantum chemistry workflows on Compute Canada. It’s designed to make complex job submission, provenance tracking, and data organization as simple as a few CLI commands.

---

## 🚀 Quickstart

### 1. Generate a plan from a CSV
Prepare a CSV like this:

```csv
id,type,structure,charge,mult,template,depends_on,functional,basis,grid,extras
ligA_M06_optfreq,optfreq,examples/ligA.xyz,0,1,orca/optfreq.inp.j2,,M06;wB97M-V,Def2-SVP,Grid4;Grid5,{"route":"M06 Opt Freq RIJCOSX"}
ligB_r2SCAN_opt,optfreq,examples/ligB.xyz,0,,orca/optfreq.inp.j2,,r2SCAN-3c,,Grid5,{"route":"r2SCAN-3c Opt"}
```

Convert it into a YAML plan that FORGE can run:

```bash
forge jobs from-csv \
  --csv jobs.csv \
  --mapping templates/mapping/plan.mapping.yaml \
  --combine plans/plan.yaml \
  --plan-schema labtools/schemas/plan.schema.json \
  --validate
```

This will generate a `plan.yaml` containing `version: 1` and a list of jobs.

### 2. Submit the plan
```bash
forge sbatch-submit \
  --plan plans/plan.yaml \
  --profile medium \
  --account def-smaley
```

FORGE renders all necessary inputs, builds sbatch scripts from templates, and submits jobs to the SLURM scheduler.

### 3. Check results
Each job directory will contain packed results (`.tar.gz`) and logs. FORGE cleans temporary files automatically.

---

## 🧩 Major Components

### 1. Universal CLI (`forge`)
Main entry point for all functionality:
- `forge jobs from-csv` – build plans from tabular input
- `forge submit-job` – run a single input file
- `forge submit-array` – run a batch of inputs
- `forge sbatch-submit` – submit a full plan
- `forge tsgen` – transition state generation and verification

### 2. Templates
- `templates/sbatch/*.sbatch.j2` → SLURM submission scripts
- `templates/orca/*.inp.j2` → ORCA input templates (must follow ORCA 6.1 syntax)
- `templates/mapping/*.yaml` → how CSV columns map to plan fields

### 3. Schemas
Located in `labtools/schemas/`:
- `plan.schema.json` – defines what a valid job plan looks like
- Validation ensures your inputs are consistent before submission

### 4. Provenance Tracking
Each plan and job can include provenance metadata (`.prov.json`) containing:
- SHA-256 hashes of source files
- Mapping/CSV/defaults references
- Expanded parameters and timestamps

---

## 🧠 TS Workflows

FORGE’s `tsgen` command automates **transition-state search and verification** through a multi-level workflow with strict provenance tracking and automatic restart logic.

### Levels of theory
| Level | Method | Purpose |
|:------|:--------|:--------|
| **L0** | XTB2 (GFN-2-xTB) | Cheap initial guess & IDPP path generation |
| **L1** | r²SCAN-3c | Geometry refinement and pre-optimization |
| **L2** | M06/Def2-SVP + RIJCOSX (default) | Full TS optimization & frequency verification |
| **L3** | DLPNO-CCSD(T)/Def2-TZVPP | Optional high-accuracy single-point |

Users can override the functional or basis at each level through the plan or command line.

### Run the pipeline
```bash
forge tsgen pipeline \
  --reactant react.xyz \
  --product prod.xyz \
  --name my_reaction \
  --levels L0 L1 L2 \
  --verify \
  --provenance
```

**Workflow:**
1. **IDPP construction** builds an interpolated path between reactant and product.
2. **L0–L1 pre-optimization** produces a refined starting geometry.
3. **L2 TS optimization** runs a full DFT-level TS search using ORCA 6.1.
4. **Fingerprint verification** confirms the correct imaginary mode.
5. **Packing & provenance** archive all intermediate artifacts into tarballs for traceability.

### Fingerprint verification
The expected imaginary mode is stored as a fingerprint file (`.yaml`):
```yaml
atoms: [1, 2, 3, 4]
reference_vector: [0.12, -0.08, ...]
localization_threshold: 0.7
cosine_overlap_min: 0.9
```
After a successful frequency calculation, FORGE compares the calculated eigenvector to this reference, computes cosine overlap, and writes results to `verify.json`. Jobs only promote to production when overlap exceeds the set threshold.

### Automatic restart logic
Each step knows how to resume safely using prior outputs:

- **OptTS** → reuse geometry and MOs with `%moinp`.

- **HybridHess** → reuse partial Hessian blocks.

- **PHVA** → reuse frequency Hessian as guess.

- **IRC** → resume from last converged point.


Restarts are automatically generated and provenance entries record the event and source geometry/Hessian identifiers.

### Example full run
```bash
forge tsgen pipeline \
  --reactant R.xyz \
  --product P.xyz \
  --plan plans/ts_pipeline.yaml \
  --verify \
  --restart-on-fail \
  --account def-smaley
```

---

## 🧰 Profiles & Job Types

### Job types (schema-enforced)
- `optfreq` – optimize + frequency
- `sp` – single point
- `sp_triplet` – single point (triplet)
- `nmr` – NMR calculation

### Submission profiles
Profiles live in config and control resources:
- `test` – 1 core, 10 min walltime
- `short` – 4 cores, 2 hr
- `medium` – 8 cores, 12 hr
- `long` – 8 cores, 48 hr

Override manually:
```bash
forge submit-job --inp job.inp --time 24:00:00 --mem 32G --pal 8 --account def-smaley
```

---

## 📦 Output Handling

After each job finishes:
1. FORGE **packs** heavy files (`*.gbw`, `*.hess`, `*.out`, etc.) into a `.tar.gz`.
2. Moves the tarball and summary files to the top-level directory.
3. Deletes transient intermediates.

This ensures your workspace stays clean.

---

## 🧾 Best Practices

- Keep all job plans in `plans/` and debug YAMLs in `plans/debug_jobs/`.
- Store raw input geometries in `examples/` or `inputs/`.
- Use consistent job IDs: e.g., `ligA_M06_optfreq`.
- Verify all ORCA keywords are 6.1-compliant before submitting.
- Commit your plan, mapping, templates, and results tarballs for reproducibility.

---

## 🧩 Common Commands

| Task | Command |
|------|----------|
| Generate plan | `forge jobs from-csv --csv jobs.csv --mapping templates/mapping/plan.mapping.yaml --combine plans/plan.yaml --plan-schema labtools/schemas/plan.schema.json --validate` |
| Submit plan | `forge sbatch-submit --plan plans/plan.yaml --profile medium --account def-smaley` |
| Submit single | `forge submit-job --inp path/to/file.inp --profile short --account def-smaley` |
| Submit array | `forge submit-array --inps 'dir/*.inp' --profile long --account def-smaley` |
| TS pipeline | `forge tsgen pipeline --reactant r.xyz --product p.xyz --levels L0 L1 L2 --verify --account def-smaley` |
| TS verify | `forge tsgen verify --ts ts_opt.out --fingerprint refs/my_ts.fprint.yaml --out verify.json` |

---

## 💬 Troubleshooting

**Schema validation failed:** Check `type`, `structure`, `template` fields; ensure `mult` is an integer.

**Enum/typing error:** Lists split by `;`, JSON columns must be valid.

**Nothing packed:** Confirm the sbatch template includes correct output file patterns.

**TS verify fails:** Likely a wrong fingerprint or atom mismatch.

**SLURM slot errors:** Use a smaller profile or confirm `--account` is correct.

---

### 🧭 TL;DR

Start from CSV → generate plan → validate → submit. Use the TSGen luxury pipeline for multi-level DFT TS searches with automatic verification and restart handling. Stick to ORCA 6.1 keywords. Commit plans and outputs for reproducibility.
