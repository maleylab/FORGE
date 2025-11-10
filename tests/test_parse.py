from labtools.orca.parse import parse_orca_file

def test_parse_minimal(tmp_path):
    p = tmp_path / "tiny.out"
    p.write_text("""TOTAL SCF ENERGY     -1.23456789
Alpha  occ. eigenvalues --  -0.40000  -0.30000
Alpha virt. eigenvalues --   0.05000   0.20000
""", encoding="utf-8")
    rec = parse_orca_file(str(p))
    assert rec["energy_Eh"] == -1.23456789
    assert rec["HOMO_Eh"] == -0.3
    assert rec["LUMO_Eh"] == 0.05
    assert abs(rec["gap_eV"] - (0.05 - (-0.3))*27.211386245988) < 1e-6
