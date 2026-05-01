# FORGE (lab-tools)

FORGE is a workflow framework for ORCA-based
computational chemistry.

It is designed to automate: - ORCA job creation - Plan generation from
CSV / mapping workflows - Fanout expansion (e.g., method × basis
campaigns) - Input rendering - SLURM submission - Drone-based autonomous
execution - Provenance tracking - Parsing, aggregation, and dataset
generation

FORGE is our group's primary computational infrastructure for
reproducible, scalable quantum chemistry workflows.


------------------------------------------------------------------------

# Installation

Editable install from repository root:

``` bash
pip install -e .
```

Main CLI:

``` bash
forge --help
```

------------------------------------------------------------------------

# Core Command Families

``` bash
forge job ...
forge plan ...
forge submit ...
forge watch ...
forge scan ...
forge parse ...
forge mark ...
forge clean ...
```

------------------------------------------------------------------------

# Quick Start

## 1. Single ORCA Job

Create a standard job:

``` bash
forge job create \
  --xyz water.xyz \
  --task optfreq \
  --method B3LYP \
  --basis def2-SVP \
  --outdir jobs
```

This creates:

``` text
jobs/
└── water_optfreq/
    ├── job.inp
    ├── job.sbatch
    ├── plan_entry.json
    └── READY
```

Submit immediately:

``` bash
forge job create ... --submit
```

## 2. CSV → Plan Workflow

FORGE supports semicolon-delimited fanout for campaign generation.

Example CSV:

``` csv
id,structure,method,basis
h2_001,h2.xyz,B3LYP;PBE0,def2-SVP;def2-TZVP
```

Generate plan:

``` bash
forge plan-from-csv \
  --csv jobs.csv \
  --mapping mapping.yaml \
  --outdir plan
```

As a practical note, it's convenient to have one primary mapping file located somewhere and give the path to it here. (e.g., --mapping $HOME/mapping.yaml)

Render:

``` bash
forge plan render \
  --plan plan/planentries.jsonl \
  --outdir build
```

------------------------------------------------------------------------

# Drones (Recommended Primary Run Mode)

## What drones are:

Drones are autonomous worker jobs submitted to SLURM that continuously
poll a queue directory, pick up available READY jobs, run them, and
repeat until walltime expires.

## Why drones are preferred:

-   Better queue efficiency
-   Fewer scheduler submissions
-   Continuous throughput
-   Excellent for large campaigns
-   More flexible than standard arrays
-   Easier restart/replenishment

## Standard drone workflow:

### Step 1: Render jobs

``` bash
forge plan render --plan plan/planentries.jsonl --outdir build
```

### Step 2: Ready jobs

From inside the directory containing the jobs (usually build/jobs), create a READY sentinel in each job directory.
``` bash
find . -mindepth 1 -type d -exec touch {}/READY \;
```

### Step 3: Launch drones
Still inside of the jobs/ folder, run
``` bash
forge submit drone --queue-dir . --n --mem-per-cpu --nprocs --time
```
where:
- --n = number of drones to submit
- --mem-per-cpu = GB per core
- --nprocs = number of cores
- --time = time in hh:mm:ss format

For example, if you wish to submit 10 drones, each requesting 8 cores and 4GB mem for 1 day
```Bash
forge submit drone --queue-dir . --n 10 --nprocs 8 --mem-per-cpu 4G --time 24:00:00
```



# Core Concepts

## Job

A single ORCA calculation directory:

``` text
job.inp
job.sbatch
plan_entry.json
READY
```

## PlanEntry

A normalized JSON representation of: - system - task - parameters -
provenance

## Fanout

Automatic expansion of list-valued parameters into campaigns:

``` text
B3LYP;PBE0 × def2-SVP;def2-TZVP
```

------------------------------------------------------------------------

# Status Files

``` text
READY     job prepared
STARTED   job running
DONE      job completed successfully
FAIL      job failed
GOOD/BAD  manual or QC promotion states
```

------------------------------------------------------------------------

# Operational Utilities

# `forge scan`

Scans a directory for job statuses.

``` bash
forge scan build/jobs
```

Purpose: - Post job inspection

# `forge mark`

This creates various post run status files in the job directories

``` bash
forge mark --csv results.csv
```

Typical uses: - Promote successful jobs - Mark failures

# `forge clean`

This clears out failed job directories. It re-writes the input file with the last set of coordinates ORCA produced.

``` bash
forge clean build/jobs
```

Typical uses: - Preparing jobs for restart


# `forge parse`

Extracts structured ORCA results into machine-readable JSONL.

``` bash
forge parse build/jobs --out parsed.jsonl
```

Output: - downstream analysis - CSV conversion - Parquet archival - QC
workflows



------------------------------------------------------------------------

# Common Student Workflow

``` bash
forge plan-from-csv ...
forge plan render ...
forge scan build/jobs
forge submit-drone build/jobs
forge parse build/jobs --out parsed.jsonl
```

------------------------------------------------------------------------

# Common Failure Modes

## "No .inp file found"

Render failure or wrong directory.

## "Run directory already exists"

Use:

``` bash
--exists overwrite
```

## Incorrect method / basis

Usually mapping or fanout configuration.

## Queue delay

SLURM wait time ≠ FORGE failure.

## SCF / ORCA crash

FORGE succeeded; chemistry failed.

------------------------------------------------------------------------

# Safety / Best Practices

## Always:

-   Use `--dry-run` when learning
-   Validate one job before scaling
-   Preserve `plan_entry.json`
-   Prefer drones for campaigns
-   Parse outputs into structured datasets

## Avoid:

-   Manual edits to generated inputs unless necessary
-   Large campaigns without test runs
-   Deleting provenance/state markers blindly

------------------------------------------------------------------------

# Templates

FORGE uses Jinja2 templates for:

``` text
templates/orca/
templates/sbatch/
```

These control: - ORCA input structure - `%maxcore` - `%pal` - SCF
settings - SLURM directives

------------------------------------------------------------------------

# Student Onboarding Recommendation

Minimum competency:

``` bash
forge job create
forge plan-from-csv
forge plan render
forge scan
forge submit-drone
forge parse
forge mark
forge clean
```