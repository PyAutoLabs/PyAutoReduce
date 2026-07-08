import pytest

from autoreduce import instruments
from autoreduce.drizzle.combine import drizzle_kwargs_for
from autoreduce.target import TargetSpec


class TestWFC3Adapters:
    def test_both_channels_registered(self):
        assert "wfc3_uvis" in instruments.registered_keys()
        assert "wfc3_ir" in instruments.registered_keys()

    def test_uvis_is_acs_like(self):
        uvis = instruments.get("wfc3_uvis")
        assert uvis.calibrated_suffix == "FLC"
        assert uvis.supports_cte_correction
        assert uvis.reference_env_key == "iref"
        assert uvis.native_scale == pytest.approx(0.0396)
        assert uvis.recommended_final_scale == pytest.approx(0.0396)

    def test_ir_is_the_different_path(self):
        ir = instruments.get("wfc3_ir")
        # No CTE correction exists for the IR channel: _flt, not _flc.
        assert ir.calibrated_suffix == "FLT"
        assert not ir.supports_cte_correction
        assert ir.reference_env_key == "iref"
        assert ir.native_scale == pytest.approx(0.128)
        # Under-sampled detector: recommendation is a finer output grid.
        assert ir.recommended_final_scale < ir.native_scale

    def test_wfc3_crds_subpath_owned_by_adapter(self):
        for key in ("wfc3_uvis", "wfc3_ir"):
            assert instruments.get(key).crds_reference_subpath == "references/hst/wfc3"

    def test_acs_recommendation_unchanged(self):
        # Phase-1 regression: the ACS path must not change.
        assert instruments.get("acs_wfc").recommended_final_scale == pytest.approx(0.05)


class TestWFC3DrizzleKwargs:
    def test_uvis_kwargs_at_bayer_scale(self):
        spec = TargetSpec(
            name="j0252", ra=43.19, dec=0.666, instrument="wfc3_uvis",
            filter_name="F390W", final_scale=0.0396, final_pixfrac=1.0,
        )
        kwargs = drizzle_kwargs_for(spec, instruments.get("wfc3_uvis"), 4)
        assert kwargs["final_scale"] == pytest.approx(0.0396)
        assert kwargs["final_pixfrac"] == pytest.approx(1.0)
        assert kwargs["final_units"] == "cps"
        assert kwargs["driz_cr"]

    def test_ir_single_exposure_branch_still_applies(self):
        spec = TargetSpec(
            name="x", ra=0.0, dec=0.0, instrument="wfc3_ir",
            filter_name="F160W", final_scale=0.065,
        )
        kwargs = drizzle_kwargs_for(spec, instruments.get("wfc3_ir"), 1)
        assert not kwargs["driz_cr"]

    def test_ir_fine_grid_correlation_factor(self):
        # Drizzling 0.128 -> 0.065 with pixfrac 0.8: s < p branch engaged.
        from autoreduce.noise.rms import casertano_r

        ir = instruments.get("wfc3_ir")
        s = ir.scale_ratio(0.065)
        assert s < 0.8
        assert casertano_r(0.8, s) > casertano_r(0.8, 1.0)


def test_crds_environment_uses_adapter_subpath(tmp_path, monkeypatch):
    import os

    from autoreduce.acquire.crds import configure_environment

    monkeypatch.delenv("iref", raising=False)
    monkeypatch.delenv("CRDS_PATH", raising=False)
    env = configure_environment(tmp_path, instruments.get("wfc3_ir"))
    assert env["iref"].endswith("references/hst/wfc3/")
    assert os.environ["iref"] == env["iref"]
