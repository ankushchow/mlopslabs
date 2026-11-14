# Auto MPG source

Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning Repository.
[Dataset and citation](https://doi.org/10.24432/C5859H).
[Source download](https://archive.ics.uci.edu/static/public/9/auto+mpg.zip).
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`auto-mpg.data` is included unchanged from the UCI archive.
SHA-256: `48b830e11feee5572525f8f1691ddb9d38d3d7b7063edcd8fca672c2a5e17d8d`.

The loader removes six rows with missing horsepower, retains original zero-based row IDs,
and uses six numeric predictors. It excludes the vehicle name and categorical origin code.
The target is `mpg`. These are this lab's processing choices; the bundled source is unmodified.
