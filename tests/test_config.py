from pathlib import Path

from dgdp.config import load_config


def test_load_config_reads_synthetic_defaults():
    cfg = load_config(Path("configs/milestone1.synthetic.toml"))

    assert cfg.seed == 20260602
    assert cfg.n_galaxies == 12
    assert cfg.projections_per_galaxy == 6
    assert cfg.image_size == 96
    assert cfg.max_inclination_deg == 60.0
    assert cfg.radial_bins_kpc == (0.0, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 13.0, 16.0)


def test_load_config_reads_clean_milestone2b_image_settings():
    cfg = load_config(Path("configs/milestone2b.clean3d.toml"))

    assert cfg.image_size == 192
    assert cfg.pixel_scale_kpc == 0.35
    assert cfg.psf_sigma_pixels == 0.0
    assert cfg.noise_sigma_fraction == 0.0
