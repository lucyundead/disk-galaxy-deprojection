"""Model artifacts + the training MDN.

Kept import-free so the package (and the bundled ``dgdp_fixed_dict.npz`` located via
``importlib.resources``) loads without torch. Import the trainer MDN explicitly where needed:
``from dgdp.models.mdn import SummaryResidualMDN`` (requires the ``[train]`` extra).
"""
