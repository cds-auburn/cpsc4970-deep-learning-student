# Electricity demand excerpt

Source: [NESO Historic Demand Data 2024](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2024),
published by Great Britain's National Energy System Operator under the
[NESO Open Data Licence](https://www.neso.energy/data-portal/neso-open-licence).

Supported by National Energy SO Open Data.

| File | Contents |
|---|---|
| `neso-half-hourly.csv` | The `ND` (National Demand, MW) column for the 48 settlement periods of each day, January 8–15, 2024: 384 rows. |
| `neso-hourly.csv` | The mean of each hour's two half-hourly values: 192 rows. `hour_start_utc` marks the start of the hour. |

Each half-hourly value covers an equal interval, so their mean is the hour's
average power in MW, not energy. January has no clock change in Great Britain,
so the settlement periods of these eight days align with UTC hours.

The lecture uses the first 168 hourly values (January 8–14, Monday through
Sunday) as the model's history and the last 24 (Monday, January 15) only for
comparison with the forecast.

`tools/prepare_electricity_demo.py` downloads the source file, checks its
SHA-256 hash against `models/electricity/manifest.json`, and regenerates both
excerpts. The complete source file stays in the ignored cache.
