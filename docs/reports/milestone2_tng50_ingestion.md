# Milestone 2: TNG50 Ingestion

## Status

TNG50 ingestion pipeline implemented. Awaiting remote cluster execution.

## Pipeline Overview

1. Build TNG50 candidate manifest from group catalog
2. Extract compact stellar particle files using offset-based reading
3. Build residual benchmark table from TNG50 particles
4. Train summary residual MDN on TNG50 data
5. Evaluate metrics and calibrate uncertainties

## TNG50 Tiny Benchmark Metrics

- Manifest: outputs/tng50_milestone2/manifest.csv
- Residual table: outputs/tng50_milestone2/residual_table.npz
- Metrics: outputs/tng50_milestone2/metrics.json

The first run uses a tiny selected sample and is intended to validate ingestion,
not to make a scientific performance claim.

## Remote Commands



## Files Created

- src/dgdp/tng50_catalog.py - Group catalog reader with TNG50GroupCatalog dataclass
- src/dgdp/bar_catalog.py - Optional bar catalog inspection and join
- src/dgdp/tng50.py - Offset-based stellar particle extraction (expanded)
- scripts/build_tng50_manifest.py - Remote manifest builder
- scripts/extract_tng50_particles.py - Remote particle extractor
- scripts/build_tng50_benchmark.py - Remote benchmark builder
- scripts/cluster_dgdp.py - Cluster CLI (expanded with run-tng50)
- configs/milestone2.cluster.toml - Cluster configuration
