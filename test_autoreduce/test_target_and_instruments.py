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

    def test_cr_dials_default_to_current_behaviour(self):
        # The default flip to a per-frame route is human-gated on SLACS
        # validation (#61); "auto" adds no drizzle pass (#62).
        spec = TargetSpec(name="x", ra=0.0, dec=0.0)
        assert spec.cr_method == "driz_cr"
        assert spec.psf_star_pass == "auto"

    def test_invalid_cr_method_rejected(self):
        with pytest.raises(ValueError, match="cr_method"):
            TargetSpec(name="x", ra=0.0, dec=0.0, cr_method="lacosmic")

    def test_invalid_psf_star_pass_rejected(self):
        with pytest.raises(ValueError, match="psf_star_pass"):
            TargetSpec(name="x", ra=0.0, dec=0.0, psf_star_pass="always")


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


class TestDqBitsDial:
    """
    The exposure-count-keyed DQ-bits dial (issue #65 leg 1). Values mirror
    STScI's MDRIZTAB reference files; before this, no bits keyword was ever
    emitted and every reduction inherited drizzlepac's package default
    `final_bits="0"` — no bit treated as good, every flagged pixel rejected.
    """

    def _bits(self, key, n, **spec_kwargs):
        from autoreduce.drizzle.combine import drizzle_kwargs_for

        spec = TargetSpec(name="x", ra=0.0, dec=0.0, **spec_kwargs)
        kwargs = drizzle_kwargs_for(spec, instruments.get(key), n)
        return kwargs["driz_sep_bits"], kwargs["final_bits"]

    @pytest.mark.parametrize("key", ["acs_wfc", "wfc3_uvis"])
    def test_acs_like_rows(self, key):
        # Single exposure: every bit good — there is nothing to fill a masked
        # pixel with, so the standard recipe keeps flagged pixels.
        assert self._bits(key, 1) == (65535, 65535)
        # N >= 2: 336 = 16 + 64 + 256 (hot, warm, saturated).
        assert self._bits(key, 2) == (336, 336)
        assert self._bits(key, 7) == (336, 336)
        assert 336 == 16 | 64 | 256

    def test_wfc3_ir_rows_differ_between_the_two_columns(self):
        # The IR table is why this is a table and not a pair: at N = 2-3 the
        # separate drizzle keeps every bit while the final drizzle is already
        # at 528 = 512 + 16 (blob + hot).
        assert self._bits("wfc3_ir", 1) == (65535, 65535)
        assert self._bits("wfc3_ir", 2) == (65535, 528)
        assert self._bits("wfc3_ir", 3) == (65535, 528)
        assert self._bits("wfc3_ir", 4) == (528, 528)
        assert 528 == 512 | 16

    def test_five_exposure_f160w_mosaic_would_not_have_holed(self):
        # The regression this leg exists for: an HST program 14653 F160W
        # mosaic with five exposures lands on the numimages >= 4 row, where
        # STScI passes exactly the blob bit (512) that punched the 123-px
        # hole.
        _, final_bits = self._bits("wfc3_ir", 5)
        assert final_bits & 512

    def test_target_spec_overrides_the_adapter_at_every_n(self):
        assert self._bits("acs_wfc", 4, final_bits=0) == (336, 0)
        assert self._bits("acs_wfc", 1, driz_sep_bits=8) == (8, 65535)
        assert self._bits("wfc3_ir", 5, final_bits=65535, driz_sep_bits=65535) == (
            65535,
            65535,
        )

    def test_non_hst_adapters_emit_no_bits_keywords(self):
        # jwst_image3 / nirc2_native have no such keyword; the key must be
        # ABSENT rather than 0, since 0 is drizzlepac's "no bit is good".
        from autoreduce.drizzle.combine import drizzle_kwargs_for

        for key in ("nircam_sw", "nirc2_narrow"):
            adapter = instruments.get(key)
            assert adapter.dq_bits_for(4) is None
            kwargs = drizzle_kwargs_for(
                TargetSpec(name="x", ra=0.0, dec=0.0), adapter, 4
            )
            assert "final_bits" not in kwargs
            assert "driz_sep_bits" not in kwargs

    def test_invalid_bits_raise_at_spec_construction(self):
        with pytest.raises(ValueError, match="final_bits"):
            TargetSpec(name="x", ra=0.0, dec=0.0, final_bits=-1)
        with pytest.raises(ValueError, match="driz_sep_bits"):
            TargetSpec(name="x", ra=0.0, dec=0.0, driz_sep_bits="336")

    def test_zero_exposures_still_raises(self):
        with pytest.raises(ValueError, match="at least one exposure"):
            instruments.get("acs_wfc").dq_bits_for(0)

    def test_provenance_records_value_and_source(self):
        from autoreduce.drizzle.combine import dq_bits_provenance

        adapter = instruments.get("wfc3_ir")
        default = dq_bits_provenance(TargetSpec(name="x", ra=0.0, dec=0.0), adapter, 5)
        assert default["final_bits"] == 528
        assert default["final_bits_source"] == "adapter_mdriztab"
        assert default["adapter_row_min_exposures"] == 4
        assert default["n_exposures"] == 5

        overridden = dq_bits_provenance(
            TargetSpec(name="x", ra=0.0, dec=0.0, final_bits=65535), adapter, 5
        )
        assert overridden["final_bits"] == 65535
        assert overridden["final_bits_source"] == "target_spec"
        # A partial override leaves the other column on the adapter default.
        assert overridden["driz_sep_bits_source"] == "adapter_mdriztab"

        unset = dq_bits_provenance(
            TargetSpec(name="x", ra=0.0, dec=0.0), instruments.get("nircam_sw"), 4
        )
        assert unset["final_bits"] is None
        assert unset["final_bits_source"] == "unset"
        assert "adapter_row_min_exposures" not in unset


class TestCrMethodDrizzleKwargs:
    """The cr_method routes through drizzle_kwargs_for (issue #61)."""

    def test_driz_cr_route_unchanged(self):
        # Regression: the default route is byte-identical to the pre-dial
        # behaviour — no resetbits key, so AstroDrizzle's own default rules.
        from autoreduce.drizzle.combine import drizzle_kwargs_for

        spec = TargetSpec(name="x", ra=0.0, dec=0.0)
        kwargs = drizzle_kwargs_for(spec, instruments.get("acs_wfc"), 4)
        assert kwargs["driz_cr"] and kwargs["median"] and kwargs["blot"]
        assert "resetbits" not in kwargs

    def test_deepcr_route_is_plain_mean_with_resetbits_zero(self):
        # The #61 trap: AstroDrizzle's default resetbits=4096 clears exactly
        # the DQ bit the per-frame masks were written into.
        from autoreduce.drizzle.combine import drizzle_kwargs_for

        spec = TargetSpec(name="x", ra=0.0, dec=0.0, cr_method="deepcr")
        kwargs = drizzle_kwargs_for(spec, instruments.get("acs_wfc"), 4)
        assert not (kwargs["driz_cr"] or kwargs["median"] or kwargs["blot"])
        assert kwargs["resetbits"] == 0

    def test_single_exposure_branch_unaffected_by_cr_method(self):
        from autoreduce.drizzle.combine import drizzle_kwargs_for

        adapter = instruments.get("acs_wfc")
        default = drizzle_kwargs_for(TargetSpec(name="x", ra=0.0, dec=0.0), adapter, 1)
        deepcr = drizzle_kwargs_for(
            TargetSpec(name="x", ra=0.0, dec=0.0, cr_method="deepcr"), adapter, 1
        )
        assert default == deepcr
        assert "resetbits" not in deepcr

    def test_invalid_cr_method_raises_at_spec_construction(self):
        with pytest.raises(ValueError, match="cr_method"):
            TargetSpec(name="x", ra=0.0, dec=0.0, cr_method="median")

    def test_star_pass_kwargs_ignore_cr_flags_without_clearing_them(self):
        # psf_star_pass="no_cr" (#62): plain mean, prior CR DQ flags treated
        # as good via final_bits, never cleared (frame products still need
        # the science pass's flags in the inputs).
        from autoreduce.drizzle.combine import CR_DQ_BIT, star_pass_kwargs_for

        spec = TargetSpec(name="x", ra=0.0, dec=0.0)
        kwargs = star_pass_kwargs_for(spec, instruments.get("acs_wfc"), 4)
        assert not (kwargs["driz_cr"] or kwargs["median"] or kwargs["blot"])
        assert kwargs["resetbits"] == 0
        assert kwargs["final_bits"] & CR_DQ_BIT
        # It ORs onto whatever the science pass resolved, so it inherits the
        # bits dial (#65) rather than the old `.get("final_bits", 0)` zero.
        assert kwargs["final_bits"] == 336 | CR_DQ_BIT


class TestDqCrFlagWrite:
    """The pure DQ update behind cr_method='deepcr' (issue #61)."""

    def test_sets_clears_and_preserves_dtype(self):
        import numpy as np

        from autoreduce.drizzle.combine import CR_DQ_BIT, dq_with_cr_flags

        dq = np.array([[0, CR_DQ_BIT], [16, CR_DQ_BIT | 16]], dtype=np.int16)
        mask = np.array([[True, False], [False, True]])
        out = dq_with_cr_flags(dq, mask)
        assert out.dtype == np.int16
        # Set where masked; stale CR bits cleared; other bits untouched.
        assert out.tolist() == [[CR_DQ_BIT, 0], [16, CR_DQ_BIT | 16]]

    def test_idempotent_on_rerun(self):
        import numpy as np

        from autoreduce.drizzle.combine import dq_with_cr_flags

        dq = np.array([[0, 32], [8192, 0]], dtype=np.int16)
        mask = np.array([[True, True], [False, False]])
        once = dq_with_cr_flags(dq, mask)
        assert dq_with_cr_flags(once, mask).tolist() == once.tolist()

    def test_shape_mismatch_is_loud(self):
        import numpy as np

        from autoreduce.drizzle.combine import dq_with_cr_flags

        with pytest.raises(ValueError, match="shape"):
            dq_with_cr_flags(np.zeros((2, 2), np.int16), np.zeros((3, 3), bool))
