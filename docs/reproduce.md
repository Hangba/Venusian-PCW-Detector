# Reproduce the Venus Express PCW study

The reference workflow accompanies the Chinese-language manuscript in [paper/main.typ](../paper/main.typ). It uses four model stages with three seeds each and a separate 2013 holdout checkpoint.

## Replay the reference results

Figure generation uses Poppler's `pdftocairo`. Manuscript compilation uses Typst 0.14 or later. Use a font with Chinese glyph coverage when rendering the reference figures or manuscript.

```bash
venusian-pcw verify
venusian-pcw replay
venusian-pcw paper --generated
```

Replay regenerates the tables and ten figures from frozen inputs and uses the reference copies of three figures that need raw MAG. Set `VENUS_EXPRESS_RAW_DATA_DIR` and pass `replay --raw-figures` to regenerate all 13. The `paper` command without `--generated` uses the archived figures.

| Reference result | Value |
|---|---|
| 2013 holdout review | 245 final candidates, 38 PCWs among the top 50 reviewed |
| Mission occurrence | 1,150 clusters at threshold 0.70, 0.42468 clusters per observed day |
| Distance-adjusted wave energy | 796 events, Pearson r = 0.695 with annual sunspot number |
| Dayside neutral hydrogen | 78 events, 6.55 cm⁻³ at 25,000 km, 95% interval 4.99–8.72 cm⁻³ |
| Near-radial IMF sample | An induced magnetosphere boundary identified in 32 of 50 intervals |

Numerical outputs are in `build/summary.json`, with the compiled manuscript at `build/paper/venus-pickup-ion-paper.pdf`.

## From raw MAG

```bash
export VENUS_EXPRESS_RAW_DATA_DIR=/path/to/VEX_1Hz
venusian-pcw catalogue --workers 4
venusian-pcw psd-audit --workers 4
venusian-pcw search holdout --workers 4
venusian-pcw search mission --workers 6
venusian-pcw exposure --workers 6
venusian-pcw replay --raw-search --raw-density
venusian-pcw paper --generated
```

`catalogue` independently measures the 946 annotated PCW events and writes the results under `build/raw_catalogue/`. The mission search builds a separate automatically detected catalogue. Search supports `--day YYYY-MM-DD` for diagnostics and `--resume` to reuse completed daily coarse scores.

The bow-shock geometry benchmark uses crossing positions recovered from the reference vector figures and the reported conic parameters. The holdout and near-radial IMF reviews use the supplied human annotations. See the [input definitions](data.md) for details.

## Training and fixed-window evaluation

```bash
venusian-pcw prepare-windows MStandard
venusian-pcw train MStandard --seed 2026 --from-raw

for stage in MSeed MExtend1 MExtend2 MStandard; do
  venusian-pcw prepare-windows "$stage" --role validation
done
venusian-pcw prepare-windows fixed_test
venusian-pcw evaluate-windows
```

Train any of MSeed, MExtend1, MExtend2, or MStandard with seed 2026, 2027, or 2028. The separate `holdout2013` stage uses seed 2026. For exact comparisons with the original features, point `VENUS_EXPRESS_TRAINING_INPUT_DIR` to the frozen NPZ root listed in the [input protocol](../inputs/protocol.json). Use `replay --rescore-windows` to run the original features through the reference checkpoints.

## Fonts

Both `replay` and `paper` accept repeatable `--font-family` and `--font-path` flags. List preferred installed families in fallback order or provide one or more directories of font files. The bundled New Computer Modern fonts and available system fonts supply defaults. The preferred families can differ between figures and manuscript text.

```bash
venusian-pcw replay --font-family "DejaVu Sans" --font-family "Noto Sans CJK SC"
venusian-pcw paper --generated --font-path /path/to/fonts --font-family "Noto Serif CJK SC"
```

`VENUSIAN_PCW_FONT_PATH` adds font directories for both commands, separated by the platform's path separator. Font settings apply when a figure is rendered. Text embedded in an archived PDF retains that PDF's fonts.
