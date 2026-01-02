"""
TSGen 2.1 | Controller
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import subprocess
import time

from ..submit import dispatch
from .tsgen_orca import read_final_xyz, parse_frequencies_and_modes

# Fingerprint utilities
from .fp.fingerprint import (
    load_fingerprint,
    extract_ts_mode_from_orca,
    cosine_similarity,
    mode_localization_fraction,
)


# ======================================================================
# CONTROLLER
# ======================================================================

class TSGenController:

    def __init__(self, plan, debug: bool = False):
        self.plan = plan
        self.debug = debug
        self.log = plan.logger

        # -------------------------------
        # Fingerprint (verification only)
        # -------------------------------
        self.fingerprint = None
        self.ref_mode = None
        self.cos_thresh = None
        self.loc_thresh = None

        if not debug and plan.fingerprint_file is not None:
            fp = load_fingerprint(plan.fingerprint_file)
            self.fingerprint = fp
            self.ref_mode = fp.ref_mode
            self.cos_thresh = fp.threshold_cosine
            self.loc_thresh = fp.threshold_localization

        # -------------------------------
        # Template mapping — LOCKED
        # -------------------------------
        self.template_map = {
            "L0": "tsgen_L0_worker.sbatch.j2",
            "L1": "tsgen_L1_worker.sbatch.j2",
            "L2": "tsgen_L2_worker.sbatch.j2",
            "L3": "tsgen_L3_worker.sbatch.j2",
        }

    # ==================================================================
    # MAIN PIPELINE
    # ==================================================================
    def run(self) -> Dict[str, Any]:

        self.log.info("TSGen2.1: starting pipeline")

        # -------------------------------
        # Mode enforcement (soft gate)
        # -------------------------------
        if self.plan.mode == "confirmatory" and self.fingerprint is None:
            self.log.warning(
                "Confirmatory mode requested but no fingerprint provided. "
                "Proceeding without fingerprint enforcement."
            )

        # ---------------- L0 ----------------
        l0_jobs = self._run_stage("L0")
        self._apply_templates(l0_jobs)
        self._execute_jobs(l0_jobs)
        l0_xyz = self._collect_xyz(l0_jobs)

        if self.debug:
            l0_survivors = l0_xyz
        else:
            l0_survivors = self._filter_L0(l0_xyz)
            if not l0_survivors:
                return self._fail("No L0 survivors")

        # ---------------- L1 ----------------
        l1_jobs = self._run_stage("L1", upstream=l0_survivors)
        self._apply_templates(l1_jobs)
        self._execute_jobs(l1_jobs)
        l1_xyz = self._collect_xyz(l1_jobs)

        if self.debug:
            l1_survivors = l1_xyz
        else:
            l1_survivors = self._filter_LN(l1_xyz, "L1")
            if not l1_survivors:
                return self._fail("No L1 survivors")

        # ---------------- L2 ----------------
        l2_jobs = self._run_stage("L2", upstream=l1_survivors)
        self._apply_templates(l2_jobs)
        self._execute_jobs(l2_jobs)
        l2_out = self._collect_out(l2_jobs)

        if self.debug:
            l2_survivors = l2_out
        else:
            l2_survivors = self._filter_L2(l2_out)
            if not l2_survivors:
                return self._fail("No L2 survivors")

        # ---------------- VERIFY ----------------
        verify = self._run_stage("VERIFY", upstream=l2_survivors)

        # ---------------- L3 (optional) ----------------
        l3_xyz_list = []
        if self.plan.do_l3:
            l3_xyz_list = self._run_L3(l2_survivors, verify)

        # ---------------- DONE ----------------
        self.log.info("TSGen2.1 pipeline completed successfully")

        return {
            "status": "success",
            "l0_xyz": [str(x) for x in l0_xyz],
            "l1_xyz": [str(x) for x in l1_xyz],
            "l2_out": [str(x) for x in l2_out],
            "verify": verify,
            "l3_xyz": [str(x) for x in l3_xyz_list],
        }

    # ==================================================================
    # TEMPLATE OVERRIDE
    # ==================================================================
    def _apply_templates(self, jobs):
        for j in jobs:
            stage = j["stage"]
            j["sbatch_template"] = self.template_map[stage]

    # ==================================================================
    # STAGE RUNNER
    # ==================================================================
    def _run_stage(self, name: str, upstream=None):
        fn = self.plan.stages[name]

        # VERIFY is controller-owned
        if name == "VERIFY":
            return fn(self, upstream)

        return fn(self.plan) if upstream is None else fn(self.plan, upstream)

    # ==================================================================
    # JOB EXECUTION (MODEL A)
    # ==================================================================
    def _execute_jobs(self, jobs):
        if self.debug:
            return self._local_fake_jobs(jobs)
        return self._slurm_jobs(jobs)

    # ---------------- DEBUG MODE ----------------
    def _local_fake_jobs(self, jobs):
        atoms, coords = self.plan.reactant_xyz

        for j in jobs:
            jobdir = Path(j["jobdir"])
            name = j["job_name"]

            new = coords.copy()
            new[:, 0] += 0.02

            xyz = jobdir / f"{name}.xyz"
            xyz.write_text(
                f"{len(atoms)}\nFake\n" +
                "\n".join(f"{a} {x:.6f} {y:.6f} {z:.6f}"
                          for a, (x, y, z) in zip(atoms, new))
            )

            out = jobdir / f"{name}.out"
            out.write_text(
                """
VIBRATIONAL FREQUENCIES
  1: -500.00

NORMAL MODES
     1
 1   0.1 0.0 -0.1
 2  -0.1 0.0  0.1

CARTESIAN COORDINATES (ANGSTROEM)
----------------------------------
"""
            )

        return True

    # ---------------- SLURM MODE ----------------
    def _slurm_jobs(self, jobs):
        submitted = []

        for j in jobs:
            jid = dispatch(
                j["input"],
                mode="job",
                profile=self.plan.profile,
                sbatch_template=j["sbatch_template"],
                extra_params={
                    "job_name": j["job_name"],
                    "jobdir": str(j["jobdir"]),
                },
            )
            j["job_id"] = jid
            submitted.append(j)

        pending = {j["job_id"]: j for j in submitted}

        while pending:
            done = []
            for jid in list(pending):
                try:
                    status = subprocess.check_output(
                        ["squeue", "-j", str(jid)], text=True
                    )
                    if str(jid) not in status:
                        done.append(jid)
                except Exception:
                    done.append(jid)
            for jid in done:
                pending.pop(jid, None)
            if pending:
                time.sleep(5)

        return True

    # ==================================================================
    # RESULT COLLECTORS
    # ==================================================================
    def _collect_xyz(self, jobs):
        return [
            Path(j["jobdir"]) / f"{j['job_name']}.xyz"
            for j in jobs
            if (Path(j["jobdir"]) / f"{j['job_name']}.xyz").exists()
        ]

    def _collect_out(self, jobs):
        return [
            Path(j["jobdir"]) / f"{j['job_name']}.out"
            for j in jobs
            if (Path(j["jobdir"]) / f"{j['job_name']}.out").exists()
        ]

    # ==================================================================
    # FINGERPRINT FILTERS
    # ==================================================================
    def _filter_L0(self, xyz_list):
        scored = []
        for xyz in xyz_list:
            out = xyz.with_suffix(".out")
            freqs, _ = parse_frequencies_and_modes(out)
            im = [abs(f) for f in freqs if f < 0]
            if im:
                scored.append((max(im), xyz))

        if not scored:
            return []

        scored.sort(reverse=True, key=lambda x: x[0])
        return [xyz for _, xyz in scored[:3]]

    def _filter_LN(self, xyz_list, stage):
        if self.plan.mode == "exploratory":
            return xyz_list

        survivors = []

        for xyz in xyz_list:
            out = xyz.with_suffix(".out")

            try:
                _, mode = extract_ts_mode_from_orca(
                    out,
                    fingerprint=self.fingerprint
                )
            except Exception as e:
                self.log.warning(
                    f"[{stage}] rejecting {out.name}: {e}"
                )
                continue

            cos = cosine_similarity(self.ref_mode, mode)
            loc = mode_localization_fraction(
                mode,
                self.fingerprint.atom_indices
            )

            if cos >= self.cos_thresh and loc >= self.loc_thresh:
                survivors.append(xyz)

        return survivors

    def _filter_L2(self, out_list):
        if self.plan.mode == "exploratory":
            return out_list

        survivors = []

        for out in out_list:
            try:
                _, mode = extract_ts_mode_from_orca(
                    out,
                    fingerprint=self.fingerprint
                )
            except Exception as e:
                self.log.warning(
                    f"[L2] rejecting {out.name}: {e}"
                )
                continue

            cos = cosine_similarity(self.ref_mode, mode)
            loc = mode_localization_fraction(
                mode,
                self.fingerprint.atom_indices
            )

            if cos >= self.cos_thresh and loc >= self.loc_thresh:
                survivors.append(out)

        return survivors

    # ==================================================================
    # L3
    # ==================================================================
    def _run_L3(self, l2_survivors, verify_results):
        l3_inputs = []

        for v, out_path in zip(verify_results, l2_survivors):
            if not v.get("passed"):
                continue

            geo = read_final_xyz(out_path)
            if not geo:
                continue

            atoms, coords = geo
            xyzpath = out_path.with_suffix(".verified.xyz")

            with xyzpath.open("w") as f:
                f.write(f"{len(atoms)}\nverified TS\n")
                for a, (x, y, z) in zip(atoms, coords):
                    f.write(f"{a} {x:.6f} {y:.6f} {z:.6f}\n")

            l3_inputs.append(xyzpath)

        if not l3_inputs:
            return []

        l3_jobs = self.plan.stages["L3"](self.plan, l3_inputs)
        self._apply_templates(l3_jobs)
        self._execute_jobs(l3_jobs)

        return l3_inputs

    # ==================================================================
    # FAIL
    # ==================================================================
    def _fail(self, msg):
        self.log.error(msg)
        return {"status": "failed", "reason": msg}
