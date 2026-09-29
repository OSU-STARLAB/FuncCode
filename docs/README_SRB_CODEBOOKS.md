# Sparse Residual Branch Codebooks Patch

This patch adds `--cluster-method branch_residual`.

SRB-Codebooks store `spline_id` for every edge, but store a residual base index only for a selected fraction of important edges.

For all edges:

```text
base_pred = conditional_base_codebook[spline_id]
```

For selected residual edges:

```text
base_weight = base_pred + residual_base_codebook[residual_id]
```

For non-selected edges:

```text
base_weight = base_pred
```

New options:

```bash
--residual-fraction 0.25
--residual-base-clusters 8
--residual-selection error   # or magnitude
```

Install by unzipping this patch inside your current codebase:

```bash
unzip SRB_Codebooks_patch.zip
```

Smoke test:

```bash
bash run_srb_codebooks.sh smoke
```

Main experiments:

```bash
bash run_srb_codebooks.sh compare_k16
bash run_srb_codebooks.sh compare_k32
bash run_srb_codebooks.sh all
```
