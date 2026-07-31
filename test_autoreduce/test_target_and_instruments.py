import pytest

from autoreduce import instruments
from autoreduce.target import TargetSpec


class TestTargetSpec:
    def test_defaults_match_design_doc(self):
        spec = TargetSpec(name="lens", ra=2.0, dec=-0.1)
        assert spec.final_scale == 0.05
        assert spec.final_pixfrac == 0.8
        assert spec.cutout_shape == (281, 281)
        assert spec.psf_shape == (21, 21)
        assert spec.psf_full_shape == (61, 61)

    def test_yaml_round_trip(self, tmp_path):
        path = tmp_path / "target.yaml"
        path.write_text(
            "name: slacs0008-0004\n"
            "ra: 2.012333\n"
            "dec: -0.068944\n"
            "proposal_ids: [10886]\n"
            "cutout_shape: [281, 281]\n"
            "final_pixfrac: 0.6\n"
        )
        spec = TargetSpec.from_yaml(path)
        assert spec.proposal_ids == ("10886",)
        assert spec.final_pixfrac == 0.6

    def test_even_psf_shape_rejected(self):
        with pytest.raises(ValueError, match="odd"):
            TargetSpec(name="x", ra=0.0, dec=0.0, psf_shape=(20, 20))

    def test_pixfrac_bounds(self):
        with pytest.raises(ValueError):
            TargetSpec(name="x", ra=0.0, dec=0.0, final_pixfrac=0.0)
        with pytest.raises(ValueError):
            TargetSpec(name="x", ra=0.0, dec=0.0, final_pixfrac=1.2)

    def test_dec_bounds(self):
        with pytest.raises(ValueError):
            TargetSpec(name="x", ra=0.0, dec=91.0)


class TestInstrumentRegistry:
    def test_acs_wfc_registered(self):
        adapter = instruments.get("acs_wfc")
        assert adapter.native_scale == 0.05
        assert adapter.calibrated_suffix == "FLC"
        assert adapter.reference_env_key == "jref"

    def test_unknown_key_raises_with_choices(self):
        with pytest.raises(KeyError, match="acs_wfc"):
            instruments.get("nircam")

    def test_double_registration_rejected(self):
        with pytest.raises(ValueError):
            instruments.register(instruments.ACS_WFC)

    def test_scale_ratio(self):
        assert instruments.get("acs_wfc").scale_ratio(0.05) == pytest.approx(1.0)
        assert instruments.get("acs_wfc").scale_ratio(0.03) == pytest.approx(0.6)


class TestReferenceSyncDecision:
    def test_hst_syncs_by_default_even_when_references_present(self):
        # Reference files update independently of the exposure cache, so a
        # fully-cached rerun still re-syncs (#63).
        from autoreduce.acquire.crds import should_sync

        assert should_sync("hst", sync_references=True, present=True)
        assert should_sync("hst", sync_references=True, present=False)

    def test_explicit_offline_opt_out_needs_a_populated_cache(self):
        from autoreduce.acquire.crds import should_sync

        assert not should_sync("hst", sync_references=False, present=True)
        with pytest.raises(RuntimeError, match="offline run needs"):
            should_sync("hst", sync_references=False, present=False)

    def test_non_hst_never_runs_explicit_bestrefs(self):
        from autoreduce.acquire.crds import should_sync

        assert not should_sync("jwst", sync_references=True, present=False)
        assert not should_sync("keck", sync_references=True, present=False)

    def test_spec_defaults_to_syncing(self):
        assert TargetSpec(name="x", ra=0.0, dec=0.0).sync_references is True


def test_drizzle_kwargs_single_vs_multi_exposure():
    from autoreduce.drizzle.combine import drizzle_kwargs_for

    spec = TargetSpec(name="x", ra=0.0, dec=0.0)
    adapter = instruments.get("acs_wfc")
    multi = drizzle_kwargs_for(spec, adapter, 4)
    single = drizzle_kwargs_for(spec, adapter, 1)
    assert multi["driz_cr"] and multi["median"] and multi["blot"]
    # SLACS-V caveat: single exposures cannot median-combine.
    assert not (single["driz_cr"] or single["median"] or single["blot"])
    assert single["final_units"] == "cps"
    assert single["final_wht_type"] == "IVM"
    with pytest.raises(ValueError):
        drizzle_kwargs_for(spec, adapter, 0)
