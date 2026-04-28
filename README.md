# SelfGNN for merchant recommendation system

## Train-only feature rebuild

Rebuild node and edge features from the exported training split only:

```powershell
.\scripts\build_train_only_features.ps1 -Data all
```

```bash
bash scripts/build_train_only_features.sh --data all
```

The generated `feature_meta.json` defines feature groups for ablations:
`value`, `time`, `category`, `repeat`, `degree`, `all`, and `all_plus_degree`.
`all` excludes degree/popularity; use `all_plus_degree` to test those signals.

Run node-feature ablations:

```powershell
.\scripts\run_feature_ablations.ps1 -Dataset yelp-merchant -Device cuda -RunNodeEdge
```

```bash
bash scripts/run_feature_ablations.sh --data yelp-merchant --device cuda --run-node-edge
```
