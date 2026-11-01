# Palmer Penguins dataset provenance

Source: [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) by Allison Horst, Alison Hill, and Kristen Gorman. Data collected by Kristen Gorman and Palmer Station Antarctica LTER. The source makes the data available under CC0. Citation: Horst AM, Hill AP, Gorman KB (2020), doi: [10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218).

`penguins_original.csv` is copied unchanged from [the project CSV](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv). It contains 344 rows. The DAG never uses its species labels.

`scripts/prepare_penguins.py` extracts only bill length, bill depth, flipper length, and body mass. Seed 42 selects 34 complete rows for prediction; 310 remaining rows go to training input, including two rows with missing measurements. Source row indices are zero-based and stored in `partition.json`. The DAG removes those missing rows and splits the remaining 308 rows into 246 training and 62 validation rows.

| File | SHA-256 |
|---|---|
| `penguins_original.csv` | `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93` |
| `penguins_training.csv` | `11daf39088d5408eaae921f9519221ed3b0c6f706d6916c051cd424ae0a5ab18` |
| `penguins_predict.csv` | `90b4989262166268f1a550f3c6da208fe6ef56a5acbe9d41691f455d6738d83f` |
| `partition.json` | `9c11c01d61ee1fabe73f6b2c179fb5b1783eccf8122fdce10c8e35fd3129573a` |
