"""
The visibility branch's numpy seam (docs/design/alma.md): assembly math,
split path logic, spec fields, packaging contract. No casatools/casatasks —
CASA-touching code is exercised by the validation prototype only, the same
rule as drizzlepac / the jwst stack.
"""

import numpy as np
import pytest

from autoreduce import instruments
from autoreduce.acquire import alma as alma_acquire
from autoreduce.instruments.alma import ms_name
from autoreduce.package import interferometer
from autoreduce.target import TargetSpec
from autoreduce.visibilities import (
    MsColumns,
    assemble_ms_products,
    concatenate,
    stokes_i_combine,
    uv_wavelengths_from_uvw,
)
from autoreduce.visibilities.assemble import SPEED_OF_LIGHT_M_S
from autoreduce.visibilities.split import (
    field_ms_path,
    resolve_width,
    spw_ms_path,
)


def _columns(n_pol=2, n_chan=1, n_rows=5, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_pol, n_chan, n_rows)) + 1j * rng.normal(
        size=(n_pol, n_chan, n_rows)
    )
    return MsColumns(
        data=data,
        uvw=rng.normal(scale=100.0, size=(3, n_rows)),
        weight=rng.uniform(0.5, 2.0, size=(n_pol, n_rows)),
        chan_freq=np.linspace(2.3e11, 2.31e11, n_chan),
        antenna1=np.arange(n_rows) % 3,
        antenna2=(np.arange(n_rows) + 1) % 4,
        time=np.linspace(0.0, 60.0, n_rows),
        scan=np.ones(n_rows, dtype=int),
    )


class TestUvWavelengths:
    def test_metres_to_wavelengths(self):
        uvw = np.array([[3.0], [4.0], [0.0]])
        freq = np.array([SPEED_OF_LIGHT_M_S])  # scale = 1: uv == metres
        uv = uv_wavelengths_from_uvw(uvw, freq)
        assert uv.shape == (1, 1, 2)
        assert uv[0, 0] == pytest.approx([3.0, 4.0])

    def test_each_channel_scales_its_own_frequency(self):
        uvw = np.array([[1.0, 2.0], [0.5, -1.0], [9.0, 9.0]])
        freqs = np.array([1.0e11, 2.0e11])
        uv = uv_wavelengths_from_uvw(uvw, freqs)
        assert uv.shape == (2, 2, 2)
        np.testing.assert_allclose(uv[1], 2.0 * uv[0])
        np.testing.assert_allclose(
            uv[0, :, 0], uvw[0] * freqs[0] / SPEED_OF_LIGHT_M_S
        )

    def test_bad_uvw_shape_rejected(self):
        with pytest.raises(ValueError, match="3, n_rows"):
            uv_wavelengths_from_uvw(np.zeros((2, 5)), np.array([1.0e11]))


class TestStokesICombine:
    def test_equal_weights_are_the_mean(self):
        data = np.array([[[1.0 + 1j, 3.0 + 0j]], [[3.0 + 3j, 5.0 + 0j]]])
        weight = np.ones((2, 2))
        stokes_i, sigma, keep = stokes_i_combine(data, weight)
        np.testing.assert_allclose(stokes_i[0], [2.0 + 2j, 4.0 + 0j])
        np.testing.assert_allclose(sigma, [1.0 / np.sqrt(2.0)] * 2)
        assert keep.all()

    def test_weighted_average_and_sigma(self):
        data = np.array([[[2.0 + 0j]], [[6.0 + 0j]]])
        weight = np.array([[3.0], [1.0]])
        stokes_i, sigma, _ = stokes_i_combine(data, weight)
        assert stokes_i[0, 0] == pytest.approx(3.0)  # (3*2 + 1*6) / 4
        assert sigma[0] == pytest.approx(0.5)  # 1/sqrt(4)

    def test_zero_weight_hand_is_excluded(self):
        data = np.array([[[1.0 + 0j]], [[100.0 + 0j]]])
        weight = np.array([[2.0], [0.0]])
        stokes_i, sigma, keep = stokes_i_combine(data, weight)
        assert stokes_i[0, 0] == pytest.approx(1.0)
        assert sigma[0] == pytest.approx(1.0 / np.sqrt(2.0))
        assert keep.all()

    def test_dead_rows_are_dropped_not_zero_filled(self):
        data = np.ones((2, 1, 3), dtype=complex)
        weight = np.array([[1.0, 0.0, 1.0], [1.0, -1.0, 1.0]])
        stokes_i, _, keep = stokes_i_combine(data, weight)
        assert list(keep) == [True, False, True]
        assert stokes_i.shape == (1, 2)

    def test_all_dead_raises(self):
        with pytest.raises(ValueError, match="positive weight"):
            stokes_i_combine(np.ones((2, 1, 2), dtype=complex), np.zeros((2, 2)))


class TestAssemble:
    def test_shapes_and_flattening(self):
        columns = _columns(n_chan=3, n_rows=4)
        result = assemble_ms_products(columns)
        assert result.visibilities.shape == (12, 2)
        assert result.uv_wavelengths.shape == (12, 2)
        assert result.noise_map.shape == (12, 2)
        assert result.provenance["n_visibilities"] == 12
        assert result.provenance["n_rows_dropped_zero_weight"] == 0

    def test_noise_is_equal_on_real_and_imag_and_shared_by_channels(self):
        columns = _columns(n_chan=2, n_rows=3)
        result = assemble_ms_products(columns)
        np.testing.assert_allclose(result.noise_map[:, 0], result.noise_map[:, 1])
        # Channel blocks of one row share the row weight.
        np.testing.assert_allclose(result.noise_map[:3, 0], result.noise_map[3:, 0])
        expected = 1.0 / np.sqrt(columns.weight.sum(axis=0))
        np.testing.assert_allclose(result.noise_map[:3, 0], expected)

    def test_dropped_rows_counted(self):
        columns = _columns(n_rows=4)
        weight = columns.weight.copy()
        weight[:, 2] = 0.0
        columns = MsColumns(
            data=columns.data,
            uvw=columns.uvw,
            weight=weight,
            chan_freq=columns.chan_freq,
            antenna1=columns.antenna1,
            antenna2=columns.antenna2,
            time=columns.time,
            scan=columns.scan,
        )
        result = assemble_ms_products(columns)
        assert result.provenance["n_rows_dropped_zero_weight"] == 1
        assert result.visibilities.shape == (3, 2)

    def test_concatenate_stacks_blocks(self):
        a = assemble_ms_products(_columns(n_rows=3, seed=1))
        b = assemble_ms_products(_columns(n_rows=5, seed=2))
        combined = concatenate([a, b], labels=["uid1/spw1", "uid1/spw2"])
        assert combined.visibilities.shape == (8, 2)
        assert combined.provenance["n_visibilities"] == 8
        assert set(combined.provenance["blocks"]) == {"uid1/spw1", "uid1/spw2"}

    def test_mismatched_labels_rejected(self):
        a = assemble_ms_products(_columns())
        with pytest.raises(ValueError, match="labels"):
            concatenate([a], labels=["x", "y"])


class TestMsColumnsContract:
    def test_shape_mismatch_is_loud(self):
        good = _columns(n_rows=4)
        with pytest.raises(ValueError, match="weight"):
            MsColumns(
                data=good.data,
                uvw=good.uvw,
                weight=good.weight[:, :2],
                chan_freq=good.chan_freq,
                antenna1=good.antenna1,
                antenna2=good.antenna2,
                time=good.time,
                scan=good.scan,
            )


class TestSplitPaths:
    def test_ms_names_match_the_recipe_convention(self):
        assert ms_name("A002_X_1") == "uid___A002_X_1.ms.split.cal"
        assert (
            field_ms_path("/w", "A002_X_1", "G09v1.40").name
            == "uid___A002_X_1_G09v1.40.ms.split.cal"
        )
        assert (
            spw_ms_path("/w", "A002_X_1", "G09v1.40", "2", 240).name
            == "uid___A002_X_1_G09v1.40_spw_2_width_240.ms.split.cal"
        )

    def test_resolve_width_passthrough_and_collapse(self):
        assert resolve_width(240, "1", [128, 128, 3840, 3840]) == 240
        assert resolve_width(0, "2", [128, 128, 3840, 3840]) == 3840

    def test_resolve_width_bad_spw(self):
        with pytest.raises(ValueError, match="out of range"):
            resolve_width(0, "7", [128, 128])


class TestAlmaSpecAndRegistry:
    def test_alma_adapter_is_visibility_domain(self):
        adapter = instruments.get("alma")
        assert adapter.domain == "visibility"
        assert adapter.archive == "alma"
        assert instruments.get("acs_wfc").domain == "imaging"

    def test_spec_yaml_round_trip(self, tmp_path):
        path = tmp_path / "target.yaml"
        path.write_text(
            "name: g09v140\n"
            "ra: 135.0\n"
            "dec: 0.5\n"
            "instrument: alma\n"
            "alma_uids: [A002_Xb9b1b9_X3046, A002_Xb99cbd_X2456]\n"
            "alma_field: G09v1.40\n"
            "alma_spws: [1, 2]\n"
            "alma_width: 240\n"
        )
        spec = TargetSpec.from_yaml(path)
        assert spec.alma_uids == ("A002_Xb9b1b9_X3046", "A002_Xb99cbd_X2456")
        assert spec.alma_spws == ("1", "2")
        assert spec.alma_width == 240
        assert spec.alma_ms_dir is None

    def test_negative_width_rejected(self):
        with pytest.raises(ValueError, match="alma_width"):
            TargetSpec(name="x", ra=0.0, dec=0.0, alma_width=-1)


class TestAcquireLocal:
    def test_resolve_calibrated_ms(self, tmp_path):
        for uid in ("A1", "A2"):
            (tmp_path / ms_name(uid)).mkdir()
        paths = alma_acquire.resolve_calibrated_ms(tmp_path, ["A1", "A2"])
        assert [p.name for p in paths] == [ms_name("A1"), ms_name("A2")]

    def test_missing_uid_is_loud_and_lists_present(self, tmp_path):
        (tmp_path / ms_name("A1")).mkdir()
        with pytest.raises(FileNotFoundError, match="A2") as err:
            alma_acquire.resolve_calibrated_ms(tmp_path, ["A1", "A2"])
        assert ms_name("A1") in str(err.value)

    def test_a_file_is_not_a_measurement_set(self, tmp_path):
        (tmp_path / ms_name("A1")).touch()
        with pytest.raises(FileNotFoundError):
            alma_acquire.resolve_calibrated_ms(tmp_path, ["A1"])


class TestVisibilityPipelineGuards:
    def test_missing_spec_fields_are_loud(self, tmp_path):
        from autoreduce.pipeline import reduce_target

        spec = TargetSpec(name="x", ra=0.0, dec=0.0, instrument="alma")
        with pytest.raises(ValueError, match="alma_uids"):
            reduce_target(
                spec, cache_root=tmp_path / "cache", output_root=tmp_path / "out"
            )

    def test_neither_ms_dir_nor_project_code_is_loud(self, tmp_path):
        from autoreduce.pipeline import reduce_target

        spec = TargetSpec(
            name="x",
            ra=0.0,
            dec=0.0,
            instrument="alma",
            alma_uids=("A1",),
            alma_field="F",
            alma_spws=("1",),
        )
        with pytest.raises(ValueError, match="alma_ms_dir"):
            reduce_target(
                spec, cache_root=tmp_path / "cache", output_root=tmp_path / "out"
            )


class TestInterferometerPackage:
    def test_writes_triplet_and_sidecars(self, tmp_path):
        fits = pytest.importorskip("astropy.io.fits")
        n = 7
        products = interferometer.write_products(
            tmp_path,
            visibilities=np.random.default_rng(0).normal(size=(n, 2)),
            uv_wavelengths=np.random.default_rng(1).normal(size=(n, 2)),
            noise_map=np.full((n, 2), 0.1),
            sidecars={"antennas": np.zeros((2, n))},
        )
        assert products == [
            "data.fits",
            "uv_wavelengths.fits",
            "noise_map.fits",
            "antennas.fits",
        ]
        data = fits.getdata(tmp_path / "data.fits")
        assert data.shape == (n, 2)
        # FITS is big-endian on disk; 8-byte float is the contract.
        assert data.dtype.kind == "f" and data.dtype.itemsize == 8

    def test_wrong_shape_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="al.Interferometer"):
            interferometer.write_products(
                tmp_path,
                visibilities=np.zeros((5, 2)),
                uv_wavelengths=np.zeros((4, 2)),
                noise_map=np.full((5, 2), 0.1),
            )

    def test_nonpositive_noise_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="positive"):
            interferometer.write_products(
                tmp_path,
                visibilities=np.zeros((3, 2)),
                uv_wavelengths=np.zeros((3, 2)),
                noise_map=np.zeros((3, 2)),
            )

    def test_non_finite_rejected(self, tmp_path):
        vis = np.zeros((3, 2))
        vis[1, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            interferometer.write_products(
                tmp_path,
                visibilities=vis,
                uv_wavelengths=np.zeros((3, 2)),
                noise_map=np.full((3, 2), 0.1),
            )
