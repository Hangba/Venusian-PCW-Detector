# Venusian PCW Detector

Detect proton cyclotron waves (PCWs) in Venus Express magnetometer observations and analyze their occurrence, wave energy, and relationship to the Venusian plasma environment.

- **Search continuous MAG records** with a trained detector, event-boundary refinement, and physical screening.
- **Analyze an event catalogue** using your own UTC intervals or the included 946-event reference list.
- **Rebuild model inputs** from raw observations and fixed window lists for training or evaluation.
- **Reproduce the reference study** with frozen inputs, model checkpoints, and figure scripts.

## Install

Use Python 3.12 for the pinned environment. Install from a checkout so that the command can find the study inputs and checkpoints.

```bash
uv venv --python 3.12
uv pip install -r requirements-repro.txt -e .
source .venv/bin/activate
venusian-pcw --help
```

## Start with a day of data

Set the path to a directory of calibrated 1 Hz VSO BAM files. See the [data formats and sources](docs/data.md) for the MAG layout and the event CSV schema.

```bash
export VENUS_EXPRESS_RAW_DATA_DIR=/path/to/VEX_1Hz
venusian-pcw cache --day 2006-04-24
venusian-pcw catalogue --day 2006-04-24
venusian-pcw search mission --day 2007-02-04
```

`cache` builds reusable daily magnetic and polarization features. `catalogue` measures the listed events and accepts `--events /path/to/events.csv`. `search` detects new candidates in continuous records. Caches and results are written to `build/`. Supply MAG data separately.

## Workflows

| Task | Command |
|---|---|
| Analyze all events in a catalogue | `venusian-pcw catalogue --events events.csv --workers 4` |
| Search the mission with the reference checkpoint | `venusian-pcw search mission --workers 6` |
| Rebuild training and validation windows | `venusian-pcw prepare-windows MStandard` |
| Train on locally rebuilt features | `venusian-pcw train MStandard --seed 2026 --from-raw` |
| Replay the study from frozen inputs | `venusian-pcw replay` |

The [reproduction guide](docs/reproduce.md) covers the study benchmarks, model splits, and figure generation, including [font selection](docs/reproduce.md#fonts).
