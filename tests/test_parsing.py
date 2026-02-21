"""
Tests for parse_aln.py parsing functions.

Uses ts-001 from the run001 dataset as ground truth — all expected values
have been verified manually against the raw files.

Tests are skipped automatically if the data directory is not mounted.
Run with:  pytest tests/test_parsing.py -v
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from aretomo3_editor.shared.parsers import (
    parse_aln_file, parse_ctf_file, parse_tlt_file, parse_mdoc_file,
)

DATA_DIR = Path('/mnt/McQueen-002/sconnell/TEST-ARETOMO-PARSE/relion')
RUN001   = DATA_DIR / 'run001'
FRAMES   = DATA_DIR / 'frames'
TS       = 'ts-001'

skip_if_no_data = pytest.mark.skipif(
    not RUN001.exists(),
    reason='run001 data directory not found',
)


# ─────────────────────────────────────────────────────────────────────────────
# parse_aln_file
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_no_data
class TestParseAln:
    @pytest.fixture(scope='class')
    def aln(self):
        return parse_aln_file(RUN001 / f'{TS}.aln')

    def test_raw_size(self, aln):
        assert aln['width']        == 5760
        assert aln['height']       == 4092
        assert aln['total_frames'] == 29

    def test_header_values(self, aln):
        assert aln['alpha_offset'] == pytest.approx(-18.6)
        assert aln['beta_offset']  == pytest.approx(0.0)
        assert aln['thickness']    == 590
        assert aln['num_patches']  == 0

    def test_aligned_frame_count(self, aln):
        # 29 total - 11 dark = 18 aligned
        assert len(aln['frames']) == 18

    def test_dark_frame_count(self, aln):
        assert len(aln['dark_frames']) == 11

    def test_sec_numbering_is_1indexed_sequential(self, aln):
        secs = [f['sec'] for f in aln['frames']]
        assert secs == list(range(1, 19))

    def test_reference_frame(self, aln):
        # Reference frame: tx=0, ty=0
        refs = [f for f in aln['frames'] if f['tx'] == 0.0 and f['ty'] == 0.0]
        assert len(refs) == 1
        assert refs[0]['sec']  == 17
        assert refs[0]['tilt'] == pytest.approx(1.38, abs=0.01)

    def test_dark_frame_fields(self, aln):
        # First dark frame: frame_a=18, frame_b=19, tilt=7.38 (corrected)
        df0 = aln['dark_frames'][0]
        assert df0['frame_a'] == 18
        assert df0['frame_b'] == 19
        assert df0['tilt']    == pytest.approx(7.38, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# parse_tlt_file
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_no_data
class TestParseTlt:
    @pytest.fixture(scope='class')
    def tlt(self):
        return parse_tlt_file(RUN001 / f'{TS}_TLT.txt')

    def test_row_count_equals_total_frames(self, tlt):
        # One row per frame including dark frames
        assert len(tlt) == 29

    def test_keys_are_1indexed(self, tlt):
        assert min(tlt.keys()) == 1
        assert max(tlt.keys()) == 29

    def test_first_acquired_frame_is_row_15(self, tlt):
        # acq_order=1 → row 15 (nominal_tilt=13.98°, first image acquired)
        row = tlt[15]
        assert row['nominal_tilt']  == pytest.approx(13.98, abs=0.01)
        assert row['acq_order']     == 1
        assert row['z_value']       == 0        # acq_order - 1
        assert row['dose_e_per_A2'] == pytest.approx(4.16, abs=0.01)

    def test_z_value_is_acq_order_minus_one(self, tlt):
        for row in tlt.values():
            assert row['z_value'] == row['acq_order'] - 1

    def test_corrected_tilt_of_row_matches_aln(self, tlt):
        # Row 15: nominal 13.98 + alpha_offset -18.6 = -4.62  → SEC 15 tilt in .aln
        assert tlt[15]['nominal_tilt'] + (-18.6) == pytest.approx(-4.62, abs=0.02)

    def test_acq_orders_are_unique(self, tlt):
        orders = [r['acq_order'] for r in tlt.values()]
        assert len(orders) == len(set(orders))

    def test_acq_orders_cover_1_to_n(self, tlt):
        orders = sorted(r['acq_order'] for r in tlt.values())
        assert orders == list(range(1, len(tlt) + 1))


# ─────────────────────────────────────────────────────────────────────────────
# parse_ctf_file
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_no_data
class TestParseCtf:
    @pytest.fixture(scope='class')
    def ctf(self):
        return parse_ctf_file(RUN001 / f'{TS}_CTF.txt')

    def test_row_count_equals_total_frames(self, ctf):
        assert len(ctf) == 29

    def test_keys_are_1indexed(self, ctf):
        assert min(ctf.keys()) == 1
        assert max(ctf.keys()) == 29

    def test_sec1_defocus_values(self, ctf):
        row = ctf[1]
        assert row['defocus1_A']      == pytest.approx(32009.17, abs=0.1)
        assert row['defocus2_A']      == pytest.approx(29657.20, abs=0.1)
        assert row['mean_defocus_um'] == pytest.approx(
            (32009.17 + 29657.20) / 2 / 1e4, rel=1e-4)
        assert row['cc']              == pytest.approx(0.0408, abs=1e-4)
        assert row['fit_spacing_A']   == pytest.approx(20.864, abs=0.01)

    def test_astig_is_abs_defocus_difference(self, ctf):
        for row in ctf.values():
            expected = abs(row['defocus1_A'] - row['defocus2_A'])
            assert row['astig_A'] == pytest.approx(expected, rel=1e-4)

    def test_mean_defocus_um_conversion(self, ctf):
        for row in ctf.values():
            assert row['mean_defocus_um'] == pytest.approx(
                row['mean_defocus_A'] / 1e4, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# parse_mdoc_file
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_no_data
class TestParseMdoc:
    @pytest.fixture(scope='class')
    def mdoc(self):
        pytest.importorskip('mdocfile')
        return parse_mdoc_file(FRAMES / f'{TS}.mdoc')

    def test_frame_count(self, mdoc):
        assert len(mdoc) == 29

    def test_keys_are_0indexed_zvalues(self, mdoc):
        assert min(mdoc.keys()) == 0
        assert max(mdoc.keys()) == 28

    def test_zvalue_0_is_first_acquired(self, mdoc):
        # ZValue=0 → acq_order=1 → nominal_tilt≈14° (first image)
        frame = mdoc[0]
        assert frame['sub_frame_path'] == \
            'Position_1_001_14.00_20260213_171849_fractions.tiff'
        assert frame['mdoc_defocus']   == pytest.approx(3.66, abs=0.01)
        assert frame['target_defocus'] == pytest.approx(-3.0,  abs=0.01)
        assert frame['num_subframes']  == 8
        assert frame['stage_x']        == pytest.approx(-430.61, abs=0.01)
        assert frame['stage_y']        == pytest.approx(-277.62, abs=0.01)
        assert frame['stage_z']        == pytest.approx(97.83,   abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation: SEC ↔ _TLT.txt ↔ mdoc ZValue
# ─────────────────────────────────────────────────────────────────────────────

@skip_if_no_data
class TestCrossValidation:
    @pytest.fixture(scope='class')
    def all_data(self):
        pytest.importorskip('mdocfile')
        aln  = parse_aln_file(RUN001 / f'{TS}.aln')
        tlt  = parse_tlt_file(RUN001 / f'{TS}_TLT.txt')
        mdoc = parse_mdoc_file(FRAMES / f'{TS}.mdoc')
        return aln, tlt, mdoc

    def test_sec_row_corrected_tilt_matches_aln(self, all_data):
        """_TLT.txt row[SEC] → nominal + alpha_offset == .aln TILT for every frame."""
        aln, tlt, _ = all_data
        alpha = aln['alpha_offset']
        for f in aln['frames']:
            corrected = tlt[f['sec']]['nominal_tilt'] + alpha
            assert corrected == pytest.approx(f['tilt'], abs=0.02), \
                f"SEC {f['sec']}: expected tilt {f['tilt']}, got {corrected}"

    def test_z_value_links_tlt_row_to_mdoc(self, all_data):
        """z_value from _TLT.txt row matches a ZValue key in the mdoc."""
        aln, tlt, mdoc = all_data
        for f in aln['frames']:
            z = tlt[f['sec']]['z_value']
            assert z in mdoc, f"SEC {f['sec']}: z_value {z} not found in mdoc"

    def test_dark_frame_b_indexes_tlt_row(self, all_data):
        """DarkFrame frame_b is the direct 1-indexed row number in _TLT.txt."""
        aln, tlt, _ = all_data
        alpha = aln['alpha_offset']
        for df in aln['dark_frames']:
            row = tlt[df['frame_b']]
            corrected = row['nominal_tilt'] + alpha
            assert corrected == pytest.approx(df['tilt'], abs=0.02), \
                f"Dark frame_b {df['frame_b']}: expected {df['tilt']}, got {corrected}"

    def test_dark_frame_z_value_in_mdoc(self, all_data):
        """z_value for each dark frame maps to a valid mdoc ZValue."""
        aln, tlt, mdoc = all_data
        for df in aln['dark_frames']:
            z = tlt[df['frame_b']]['z_value']
            assert z in mdoc, f"Dark frame_b {df['frame_b']}: z_value {z} not in mdoc"

    def test_cumulative_dose_first_acq_is_zero(self, all_data):
        """Frame with acq_order=1 must have cumulative prior dose of 0."""
        _, tlt, _ = all_data
        sorted_rows = sorted(tlt.values(), key=lambda r: r['acq_order'])
        running = 0.0
        cum = {}
        for r in sorted_rows:
            cum[r['acq_order']] = running
            running += r['dose_e_per_A2']
        assert cum[1] == pytest.approx(0.0)

    def test_cumulative_dose_is_monotonically_increasing(self, all_data):
        """Cumulative dose increases (or stays equal) with each acquisition."""
        _, tlt, _ = all_data
        sorted_rows = sorted(tlt.values(), key=lambda r: r['acq_order'])
        running = 0.0
        prev = -1e-9
        for r in sorted_rows:
            assert running >= prev
            prev = running
            running += r['dose_e_per_A2']

    def test_all_tlt_rows_covered_by_aln_or_dark(self, all_data):
        """Every row in _TLT.txt is accounted for by an aligned SEC or a dark frame_b."""
        aln, tlt, _ = all_data
        aligned_secs  = {f['sec']      for f in aln['frames']}
        dark_frame_bs = {df['frame_b'] for df in aln['dark_frames']}
        all_accounted = aligned_secs | dark_frame_bs
        assert all_accounted == set(tlt.keys())
