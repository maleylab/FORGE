# src/labtools/orca/collect.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os, re

EH_TO_EV = 27.211386245988
EH_TO_KCAL_MOL = 627.509474

# ===================== Models =====================
@dataclass
class SCFSummary:
    converged: Optional[bool] = None
    iterations: Optional[int] = None
    energy_final_au: Optional[float] = None
    cpu_time: Optional[str] = None
    tol_energy_Eh: Optional[float] = None
    tol_max_grad_Eh_per_bohr: Optional[float] = None
    tol_rms_grad_Eh_per_bohr: Optional[float] = None
    tol_max_disp_bohr: Optional[float] = None
    tol_rms_disp_bohr: Optional[float] = None
    strict_convergence: Optional[bool] = None

@dataclass
class OptSummary:
    status: Optional[str] = None  # "OPTIMIZATION RUN DONE" | "FAILED" | None
    steps: Optional[int] = None

@dataclass
class MethodSetup:
    # From SCF SETTINGS table
    method: Optional[str] = None             # e.g. DFT(GTOs)
    xc_exchange: Optional[str] = None        # e.g. M06
    xc_correlation: Optional[str] = None     # e.g. M06
    hybrid_fraction: Optional[float] = None  # e.g. 0.27
    rij_cosx: Optional[bool] = None          # True if "RIJ-COSX ... on"
    ri_coulomb: Optional[bool] = None        # True if "RI-approximation ... turned on"
    grid: Optional[str] = None               # If printed
    # Basis
    basis_main: Optional[str] = None         # def2-SVP
    basis_aux_j: Optional[str] = None        # def2/J
    basis_aux_c: Optional[str] = None        # def2-SVP/C
    # Other toggles seen in the table (optional)
    relativistic: Optional[str] = None
    solvent_model: Optional[str] = None

@dataclass
class RunMeta:
    orca_version: Optional[str] = None
    pal_threads: Optional[int] = None
    omp_threads: Optional[int] = None
    job_type_guess: Optional[str] = None
    charge: Optional[int] = None
    multiplicity: Optional[int] = None

@dataclass
class FreqSummary:
    n_imag: Optional[int] = None
    lowest_cm1: Optional[float] = None
    zpe_au: Optional[float] = None

@dataclass
class Thermochemistry:
    electronic_energy_au: Optional[float] = None
    zpe_au: Optional[float] = None
    zpe_kcal_mol: Optional[float] = None
    thermal_vib_corr_au: Optional[float] = None
    thermal_rot_corr_au: Optional[float] = None
    thermal_trans_corr_au: Optional[float] = None
    total_thermal_energy_au: Optional[float] = None
    total_thermal_correction_au: Optional[float] = None
    non_thermal_zpe_correction_au: Optional[float] = None
    total_correction_au: Optional[float] = None
    thermal_enthalpy_correction_au: Optional[float] = None
    total_enthalpy_au: Optional[float] = None
    final_entropy_term_au: Optional[float] = None         # +T·S (Eh)
    total_entropy_correction_au: Optional[float] = None    # −T·S (Eh)
    gibbs_free_energy_au: Optional[float] = None
    g_minus_electronic_au: Optional[float] = None
    temperature_K: Optional[float] = None
    entropy_kcal_mol_K: Optional[float] = None             # derived if temperature_K present

@dataclass
class OrbitalSet:
    homo_Eh: Optional[float] = None
    lumo_Eh: Optional[float] = None
    homo_eV: Optional[float] = None
    lumo_eV: Optional[float] = None
    gap_eV: Optional[float] = None
    gap_eV_reported: Optional[float] = None

@dataclass
class ParsedORCA:
    path: str
    meta: RunMeta
    method: MethodSetup
    scf: SCFSummary
    opt: OptSummary
    freq: Optional[FreqSummary]
    thermo: Optional[Thermochemistry]
    orbitals_alpha: Optional[OrbitalSet]
    orbitals_beta: Optional[OrbitalSet]
    warnings: Optional[List[str]]
    errors: Optional[List[str]]
    final_single_point_energy_au: Optional[float]

# ===================== Helpers =====================
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"

_SLURM_STDNAME = re.compile(r"^(slurm[-_].*|.*\.\d+\.out|.*\.\d+\.err)$", re.I)

def _to_float(s: str) -> Optional[float]:
    try: return float(s)
    except: return None

def _bool_from_on_off(word: str) -> Optional[bool]:
    w = word.strip().lower()
    if w in ("on", "true"): return True
    if w in ("off", "false"): return False
    return None

# ===================== Streaming parser =====================
def parse_orca_out_file(out_path: Path) -> dict:
    """
    Fast single-pass parser for ORCA 6.1 .out files.
    Extracts methodology (SCF SETTINGS), basis info, convergence tolerances,
    thermochemistry, orbital gap, meta, etc.
    """
    out_path = Path(out_path)

    meta = RunMeta()
    method = MethodSetup()
    scf = SCFSummary()
    opt = OptSummary()
    freq: Optional[FreqSummary] = None
    thermo = Thermochemistry()
    warnings: List[str] = []
    errors: List[str] = []

    # HOMO/LUMO
    orb_alpha = OrbitalSet()
    orb_beta: Optional[OrbitalSet] = None
    in_orbital_block = False
    current_spin = "alpha"
    saw_orb_row_alpha = False
    saw_orb_row_beta = False
    last_alpha_occ: Optional[float] = None
    first_alpha_virt: Optional[float] = None
    last_beta_occ: Optional[float] = None
    first_beta_virt: Optional[float] = None

    # SCF SETTINGS table & convergence tolerances
    in_scf_settings = False
    in_conv_tolerances = False

    # Basis sections
    in_orb_basis = False
    in_auxj_basis = False
    in_auxc_basis = False

    # Thermochemistry regions
    in_inner_energy_summary = False

    # Other flags
    saw_opt_done = False
    saw_freq_banner = False

    # precompiled line regexes (fast)
    rx_version = re.compile(r"Program Version\s*([0-9.]+)", re.I)
    rx_pal = re.compile(r"\bPAL\s*=\s*(\d+)", re.I)
    rx_omp = re.compile(r"Number of threads.*?(\d+)", re.I)
    rx_charge = re.compile(r"Total Charge\s*:\s*([-\d]+)", re.I)
    rx_mult = re.compile(r"Multiplicity\s*:\s*([-\d]+)", re.I)
    rx_final_spe = re.compile(r"FINAL SINGLE POINT ENERGY\s+("+_NUM+")", re.I)
    rx_scf_conv = re.compile(r"\bSCF CONVERGED\b", re.I)
    rx_scf_not = re.compile(r"\bSCF NOT CONVERGED\b", re.I)
    rx_scf_iter = re.compile(r"Number of iterations\s*:\s*(\d+)", re.I)
    rx_scf_time = re.compile(r"Total CPU time for SCF\s*:\s*([^\n]+)", re.I)

    rx_scf_settings_start = re.compile(r"^\s*SCF SETTINGS\s*$", re.I)
    rx_conv_tol_start = re.compile(r"^\s*Convergence Tolerances\s*:\s*$", re.I)

    # Key-value lines of SCF SETTINGS table (.... columns)
    rx_kv = re.compile(r"(.+?)\s+\.{3,}\s+(.+?)\s*$")

    # Specific toggles we care about in SCF SETTINGS
    rx_method = re.compile(r"Density Functional\s+Method\s+\.{3,}\s+(.*)", re.I)
    rx_xc_ex = re.compile(r"Exchange Functional\s+Exchange\s+\.{3,}\s+(.*)", re.I)
    rx_xc_co = re.compile(r"Correlation Functional\s+Correlation\s+\.{3,}\s+(.*)", re.I)
    rx_hyb = re.compile(r"Fraction HF Exchange.*?\.{3,}\s+("+_NUM+")", re.I)
    rx_rij = re.compile(r"RI-approximation to the Coulomb term is turned (on|off)", re.I)
    rx_rijcosx = re.compile(r"RIJ[- ]?COSX.*?\.{3,}\s+(on|off)", re.I)
    rx_grid = re.compile(r"Grid\s*[:=]\s*(.*)", re.I)

    # Basis
    rx_orb_basis_hdr = re.compile(r"^-{5,}\s*Orbital basis set information", re.I)
    rx_auxj_basis_hdr = re.compile(r"^-{5,}\s*AuxJ basis set information", re.I)
    rx_auxc_basis_hdr = re.compile(r"^-{5,}\s*AuxC basis set information", re.I)
    rx_basis_line = re.compile(r"utilizes the (?:auxiliary )?basis:\s*([A-Za-z0-9+\-_/]+)", re.I)

    # FREQ / ZPE
    rx_freq_banner = re.compile(r"\bVIBRATIONAL ANALYSIS\b|\bFREQUENC(?:Y|IES)\b", re.I)
    rx_n_imag = re.compile(r"Number of imaginary frequencies\s*:\s*(\d+)", re.I)
    rx_lowest = re.compile(r"Lowest (?:frequency|wavenumber)\s*:\s*("+_NUM+")\s*cm", re.I)
    rx_zpe_line = re.compile(r"Zero point energy\s*:\s*("+_NUM+")\s*Eh", re.I)

    # Inner energy summary + thermochemistry
    rx_inner_hdr = re.compile(r"Summary of contributions to the inner energy U", re.I)
    rx_elec = re.compile(r"Electronic energy\s*\.{3,}\s*("+_NUM+")\s*Eh", re.I)
    rx_zpe = re.compile(r"Zero point energy\s*\.{3,}\s*("+_NUM+")\s*Eh\s*("+_NUM+")\s*kcal/mol", re.I)
    rx_vib = re.compile(r"Thermal vibrational correction\s*\.{3,}\s*("+_NUM+")\s*Eh", re.I)
    rx_rot = re.compile(r"Thermal rotational correction\s*\.{3,}\s*("+_NUM+")\s*Eh", re.I)
    rx_trans = re.compile(r"Thermal translational correction\s*\.{3,}\s*("+_NUM+")\s*Eh", re.I)
    rx_tot_therm_E = re.compile(r"Total thermal energy\s*\.{0,}\s*("+_NUM+")\s*Eh", re.I)

    rx_tot_therm_corr = re.compile(r"Total thermal correction\s*\.{0,}\s*("+_NUM+")\s*Eh", re.I)
    rx_nonthermal_zpe = re.compile(r"Non-thermal\s*\(ZPE\)\s*correction\s*\.{0,}\s*("+_NUM+")\s*Eh", re.I)
    rx_total_corr = re.compile(r"Total correction\s*\.{0,}\s*("+_NUM+")\s*Eh", re.I)

    rx_enth_corr = re.compile(r"Thermal Enthalpy correction\s*\.{0,}\s*("+_NUM+")\s*Eh", re.I)
    rx_total_enth = re.compile(r"Total Enthalpy\s*\.{0,}\s*("+_NUM+")\s*Eh", re.I)

    rx_entropy_final = re.compile(r"Final entropy term\s*\.{0,}\s*("+_NUM+")\s*Eh", re.I)
    rx_entropy_total = re.compile(r"Total entropy correction\s*\.{0,}\s*("+_NUM+")\s*Eh", re.I)
    rx_gibbs = re.compile(r"Final Gibbs free energy\s*.*?\s*("+_NUM+")\s*Eh", re.I)
    rx_g_minus_e = re.compile(r"G-E\(el\)\s*\.{0,}\s*("+_NUM+")\s*Eh", re.I)
    rx_tempK = re.compile(r"Temperature\s*[:=]\s*("+_NUM+")\s*K", re.I)

    # Orbitals table (compact scan)
    rx_orb_hdr = re.compile(r"^\s*ORBITAL\s+ENERGIES", re.I)
    rx_spin_up = re.compile(r"SPIN\s+UP\s+ORBITALS", re.I)
    rx_spin_dn = re.compile(r"SPIN\s+DOWN\s+ORBITALS", re.I)
    rx_orb_row = re.compile(r"^\s*\d+\s+("+_NUM+")\s+("+_NUM+")\s+("+_NUM+")\s*$")  # idx occ Eh eV

    # Job type barks
    rx_opt_done = re.compile(r"OPTIMIZATION RUN DONE", re.I)
    rx_ts = re.compile(r"\bTSOPT\b|\bTransition\s+State\b", re.I)
    rx_nmr = re.compile(r"\bNMR\b|\bshielding tensor\b", re.I)
    rx_sp_banner = re.compile(r"\bSINGLE\s+POINT\b", re.I)

    rx_warn = re.compile(r"^\s*Warning[: ].*$", re.I)
    rx_err  = re.compile(r"^\s*Error[: ].*$", re.I)

    final_spe: Optional[float] = None

    with open(out_path, "r", errors="ignore") as fh:
        for line in fh:
            sline = line.rstrip("\n")

            # meta light
            m = rx_version.search(sline)
            if m and not meta.orca_version:
                meta.orca_version = m.group(1)
            m = rx_pal.search(sline)
            if m and not meta.pal_threads:
                meta.pal_threads = int(m.group(1))
            m = rx_omp.search(sline)
            if m and not meta.omp_threads:
                meta.omp_threads = int(m.group(1))
            m = rx_charge.search(sline)
            if m and meta.charge is None:
                meta.charge = int(m.group(1))
            m = rx_mult.search(sline)
            if m and meta.multiplicity is None:
                meta.multiplicity = int(m.group(1))

            # warnings/errors
            if rx_warn.match(sline): warnings.append(sline.strip())
            if rx_err.match(sline):  errors.append(sline.strip())

            # SCF convergence flags
            if rx_scf_conv.search(sline): scf.converged = True
            if rx_scf_not.search(sline): scf.converged = False
            m = rx_scf_iter.search(sline)
            if m: scf.iterations = int(m.group(1))
            m = rx_scf_time.search(sline)
            if m: scf.cpu_time = m.group(1).strip()
            m = rx_final_spe.search(sline)
            if m: final_spe = float(m.group(1))

            # SCF SETTINGS & tolerances blocks
            if rx_scf_settings_start.match(sline):
                in_scf_settings = True
                continue
            if in_scf_settings:
                # Heuristic: leave block when we hit a blank line followed by a non-indented section header,
                # but we’ll just extract needed key-values and keep going; it’s harmless to keep it on.
                m = rx_method.search(sline)
                if m: method.method = m.group(1).strip()
                m = rx_xc_ex.search(sline)
                if m: method.xc_exchange = m.group(1).strip()
                m = rx_xc_co.search(sline)
                if m: method.xc_correlation = m.group(1).strip()
                m = rx_hyb.search(sline)
                if m: method.hybrid_fraction = _to_float(m.group(1))
                m = rx_rij.search(sline)
                if m: method.ri_coulomb = (m.group(1).lower() == "on")
                m = rx_rijcosx.search(sline)
                if m: method.rij_cosx = _bool_from_on_off(m.group(1))
                # Sometimes grid info appears as "Grid : something"
                m = rx_grid.search(sline)
                if m and not method.grid:
                    method.grid = m.group(1).strip()

            if rx_conv_tol_start.match(sline):
                in_conv_tolerances = True
                continue
            if in_conv_tolerances:
                # Energy Change TolE .... 1.0e-06 Eh
                if "TolE" in sline:
                    m = re.search(r"TolE\s*\.*\s*("+_NUM+")\s*Eh", sline, re.I)
                    if m: scf.tol_energy_Eh = _to_float(m.group(1))
                elif "TolMAXG" in sline:
                    m = re.search(r"TolMAXG\s*\.*\s*("+_NUM+")\s*Eh/bohr", sline, re.I)
                    if m: scf.tol_max_grad_Eh_per_bohr = _to_float(m.group(1))
                elif "TolRMSG" in sline:
                    m = re.search(r"TolRMSG\s*\.*\s*("+_NUM+")\s*Eh/bohr", sline, re.I)
                    if m: scf.tol_rms_grad_Eh_per_bohr = _to_float(m.group(1))
                elif "TolMAXD" in sline:
                    m = re.search(r"TolMAXD\s*\.*\s*("+_NUM+")\s*bohr", sline, re.I)
                    if m: scf.tol_max_disp_bohr = _to_float(m.group(1))
                elif "TolRMSD" in sline:
                    m = re.search(r"TolRMSD\s*\.*\s*("+_NUM+")\s*bohr", sline, re.I)
                    if m: scf.tol_rms_disp_bohr = _to_float(m.group(1))
                elif "Strict Convergence" in sline:
                    scf.strict_convergence = _bool_from_on_off(sline.split()[-1])
                # End of block heuristics
                if sline.strip() == "" or sline.startswith("---"):
                    # don’t force exit; safe to continue scanning
                    pass

            # Basis headers & lines
            if rx_orb_basis_hdr.search(sline):
                in_orb_basis = True; in_auxj_basis = False; in_auxc_basis = False
            elif rx_auxj_basis_hdr.search(sline):
                in_orb_basis = False; in_auxj_basis = True; in_auxc_basis = False
            elif rx_auxc_basis_hdr.search(sline):
                in_orb_basis = False; in_auxj_basis = False; in_auxc_basis = True

            if in_orb_basis or in_auxj_basis or in_auxc_basis:
                m = rx_basis_line.search(sline)
                if m:
                    basis = m.group(1).strip()
                    if in_orb_basis:
                        method.basis_main = basis
                    elif in_auxj_basis:
                        method.basis_aux_j = basis
                    elif in_auxc_basis:
                        method.basis_aux_c = basis

            # FREQ flags
            if rx_freq_banner.search(sline): saw_freq_banner = True
            m = rx_n_imag.search(sline)
            if m:
                if freq is None: freq = FreqSummary()
                freq.n_imag = int(m.group(1))
            m = rx_lowest.search(sline)
            if m:
                if freq is None: freq = FreqSummary()
                freq.lowest_cm1 = _to_float(m.group(1))
            m = rx_zpe_line.search(sline)
            if m:
                if freq is None: freq = FreqSummary()
                freq.zpe_au = _to_float(m.group(1))

            # Thermochemistry
            if rx_inner_hdr.search(sline):
                in_inner_energy_summary = True
            if in_inner_energy_summary:
                m = rx_elec.search(sline)
                if m: thermo.electronic_energy_au = _to_float(m.group(1))
                m = rx_zpe.search(sline)
                if m:
                    thermo.zpe_au = _to_float(m.group(1))
                    thermo.zpe_kcal_mol = _to_float(m.group(2))
                m = rx_vib.search(sline)
                if m: thermo.thermal_vib_corr_au = _to_float(m.group(1))
                m = rx_rot.search(sline)
                if m: thermo.thermal_rot_corr_au = _to_float(m.group(1))
                m = rx_trans.search(sline)
                if m: thermo.thermal_trans_corr_au = _to_float(m.group(1))
                m = rx_tot_therm_E.search(sline)
                if m: thermo.total_thermal_energy_au = _to_float(m.group(1))

            m = rx_tot_therm_corr.search(sline)
            if m: thermo.total_thermal_correction_au = _to_float(m.group(1))
            m = rx_nonthermal_zpe.search(sline)
            if m: thermo.non_thermal_zpe_correction_au = _to_float(m.group(1))
            m = rx_total_corr.search(sline)
            if m: thermo.total_correction_au = _to_float(m.group(1))
            m = rx_enth_corr.search(sline)
            if m: thermo.thermal_enthalpy_correction_au = _to_float(m.group(1))
            m = rx_total_enth.search(sline)
            if m: thermo.total_enthalpy_au = _to_float(m.group(1))
            m = rx_entropy_final.search(sline)
            if m: thermo.final_entropy_term_au = _to_float(m.group(1))
            m = rx_entropy_total.search(sline)
            if m: thermo.total_entropy_correction_au = _to_float(m.group(1))
            m = rx_gibbs.search(sline)
            if m: thermo.gibbs_free_energy_au = _to_float(m.group(1))
            m = rx_g_minus_e.search(sline)
            if m: thermo.g_minus_electronic_au = _to_float(m.group(1))
            m = rx_tempK.search(sline)
            if m: thermo.temperature_K = _to_float(m.group(1))

            # Orbitals
            if rx_orb_hdr.match(sline):
                in_orbital_block = True
                current_spin = "alpha"
                continue
            if in_orbital_block:
                if rx_spin_up.search(sline):
                    current_spin = "alpha"
                    continue
                if rx_spin_dn.search(sline):
                    current_spin = "beta"
                    if orb_beta is None: orb_beta = OrbitalSet()
                    continue
                mm = rx_orb_row.match(sline)
                if mm:
                    occ = _to_float(mm.group(1)) or 0.0
                    eEh = _to_float(mm.group(2))
                    eEV = _to_float(mm.group(3))
                    if current_spin == "alpha":
                        if occ > 0.0:
                            last_alpha_occ = eEh  # keep last occupied
                        elif occ == 0.0 and first_alpha_virt is None:
                            first_alpha_virt = eEh
                            # no need to parse all rows; we can still keep scanning cheap

                    else:
                        if occ > 0.0:
                            last_beta_occ = eEh
                        elif occ == 0.0 and first_beta_virt is None:
                            first_beta_virt = eEh

            # OPT status
            if rx_opt_done.search(sline):
                saw_opt_done = True

    # Fill HOMO/LUMO sets
    if last_alpha_occ is not None or first_alpha_virt is not None:
        orb_alpha.homo_Eh = last_alpha_occ
        orb_alpha.lumo_Eh = first_alpha_virt
        if last_alpha_occ is not None:
            orb_alpha.homo_eV = last_alpha_occ * EH_TO_EV
        if first_alpha_virt is not None:
            orb_alpha.lumo_eV = first_alpha_virt * EH_TO_EV
        if orb_alpha.homo_eV is not None and orb_alpha.lumo_eV is not None:
            orb_alpha.gap_eV = orb_alpha.lumo_eV - orb_alpha.homo_eV
    else:
        orb_alpha = None  # not printed in some runs

    if last_beta_occ is not None or first_beta_virt is not None:
        if orb_beta is None: orb_beta = OrbitalSet()
        orb_beta.homo_Eh = last_beta_occ
        orb_beta.lumo_Eh = first_beta_virt
        if last_beta_occ is not None:
            orb_beta.homo_eV = last_beta_occ * EH_TO_EV
        if first_beta_virt is not None:
            orb_beta.lumo_eV = first_beta_virt * EH_TO_EV
        if orb_beta.homo_eV is not None and orb_beta.lumo_eV is not None:
            orb_beta.gap_eV = orb_beta.lumo_eV - orb_beta.homo_eV

    # Derive entropy if T is known (optional)
    if thermo.temperature_K and (thermo.final_entropy_term_au or thermo.total_entropy_correction_au):
        ts_Eh = thermo.final_entropy_term_au
        if ts_Eh is None and thermo.total_entropy_correction_au is not None:
            ts_Eh = -thermo.total_entropy_correction_au
        if ts_Eh is not None and thermo.temperature_K:
            thermo.entropy_kcal_mol_K = (ts_Eh * EH_TO_KCAL_MOL) / thermo.temperature_K

    # SCF energy (from FINAL SINGLE POINT ENERGY) goes in scf.energy_final_au too
    scf.energy_final_au = final_spe
    # OPT flags
    if saw_opt_done: opt.status = "OPTIMIZATION RUN DONE"

    # FREQ: if banner seen but nothing parsed, leave None; else keep parsed object
    if not (freq and (freq.n_imag is not None or freq.zpe_au is not None or freq.lowest_cm1 is not None)):
        freq = freq if freq else None

    # Job type guess
    tags = []
    if opt.status: tags.append("OPT")
    if saw_freq_banner or (freq and (freq.zpe_au is not None or freq.n_imag is not None)): tags.append("FREQ")
    # TS?
    # Quick scan of final lines (cheap read)
    tail = ""
    try:
        with open(out_path, "r", errors="ignore") as fh2:
            fh2.seek(0, os.SEEK_END)
            sz = fh2.tell()
            fh2.seek(max(0, sz - 30000))
            tail = fh2.read()
    except:
        pass
    if re.search(r"\bTSOPT\b|\bTransition\s+State\b", tail, re.I):
        tags.append("TS")
    if re.search(r"\bNMR\b|\bshielding tensor\b", tail, re.I):
        tags.append("NMR")
    if not tags and re.search(r"\bSINGLE\s+POINT\b", tail, re.I):
        tags.append("SP")
    meta.job_type_guess = "+".join(tags) if tags else "UNKNOWN"

    # Build record
    parsed = ParsedORCA(
        path=str(out_path.resolve()),
        meta=meta,
        method=method,
        scf=scf,
        opt=opt,
        freq=freq,
        thermo=(thermo if any(getattr(thermo, k) is not None for k in thermo.__dict__) else None),
        orbitals_alpha=orb_alpha,
        orbitals_beta=orb_beta,
        warnings=(warnings or None),
        errors=(errors or None),
        final_single_point_energy_au=final_spe,
    )
    return asdict(parsed)

# ===================== Directory collector =====================
def _filesize(p: Path) -> int:
    try: return p.stat().st_size
    except: return -1

def _choose_primary_out(files: List[Path], inps: List[Path]) -> Optional[Path]:
    if not files: return None
    inp_stems = {p.stem.lower() for p in inps}
    for f in files:
        if f.stem.lower() in inp_stems:
            return f
    files = sorted(files, key=lambda p: (_filesize(p), p.stat().st_mtime), reverse=True)
    return files[0]

def collect_job_record(dirpath: Path, include_text_blobs: bool = False) -> Optional[dict]:
    dirpath = Path(dirpath)
    outs = [p for p in dirpath.glob("*.out") if not _SLURM_STDNAME.match(p.name)]
    inps = list(dirpath.glob("*.inp"))
    xyzs = list(dirpath.glob("*.xyz"))

    if not outs: return None
    primary = _choose_primary_out(outs, inps)
    if primary is None: return None

    parsed = parse_orca_out_file(primary)
    rec = {
        "dir": str(dirpath.resolve()),
        "job_name": dirpath.name,
        "primary_out": str(primary.resolve()),
        "primary_out_size": _filesize(primary),
        "parsed": parsed,
        "files": {"inp": None, "xyz": None, "artifacts": []},
    }

    if inps:
        match = [p for p in inps if p.stem.lower() == primary.stem.lower()]
        inp = match[0] if match else max(inps, key=_filesize)
        rec["files"]["inp"] = {"path": str(inp.resolve()), "size": _filesize(inp)}

    if xyzs:
        match = [p for p in xyzs if p.stem.lower() == primary.stem.lower()]
        xyz = match[0] if match else max(xyzs, key=_filesize)
        rec["files"]["xyz"] = {"path": str(xyz.resolve()), "size": _filesize(xyz)}

    for ext in (".gbw", ".engrad", ".hess", ".qfi", ".prop", ".md", ".orbitals", ".gbw.bak"):
        p = dirpath / f"{Path(primary).stem}{ext}"
        if p.exists():
            rec["files"]["artifacts"].append({"path": str(p.resolve()), "size": _filesize(p), "ext": ext})

    if include_text_blobs:
        try:
            with open(primary, "r", errors="ignore") as fh:
                head = "".join([next(fh) for _ in range(80)])
            with open(primary, "r", errors="ignore") as fh:
                fh.seek(0, os.SEEK_END)
                sz = fh.tell()
                fh.seek(max(0, sz - 8000))
                tail = fh.read()
        except Exception:
            head = ""
            tail = ""
        rec["out_head"] = head
        rec["out_tail"] = tail

    return rec

