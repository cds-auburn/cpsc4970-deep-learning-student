"""Fetch the pinned model; reproduce the January 8–15 NESO hourly excerpt."""

import csv
import hashlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    cache = HERE / "artifacts/cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name, spec in manifest["files"].items():
        path = cache / name
        if not path.exists():
            with urlopen(spec["url"], timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
                raise ValueError(f"Upstream file changed: {name}; do not silently replace it")
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Cached file has wrong hash: {path}")

    # January has no clock changes. Use only these eight complete GMT days.
    rows = list(csv.DictReader(io.StringIO((cache / "demanddata_2024.csv").read_text())))
    selected = []
    hourly = []
    for day_index in range(8):
        date = datetime(2024, 1, 8) + timedelta(days=day_index)
        day = [r for r in rows if datetime.strptime(r["SETTLEMENT_DATE"], "%d-%b-%Y") == date]
        day.sort(key=lambda r: int(r["SETTLEMENT_PERIOD"]))
        assert [int(r["SETTLEMENT_PERIOD"]) for r in day] == list(range(1, 49))
        for r in day:
            selected.append([date.date().isoformat(), r["SETTLEMENT_PERIOD"], r["ND"]])
        for hour in range(24):
            # Equal-duration half hours: average power (MW), not summed energy.
            demand = (float(day[2 * hour]["ND"]) + float(day[2 * hour + 1]["ND"])) / 2
            hourly.append([(date + timedelta(hours=hour)).isoformat() + "Z", demand])
    for name, header, values in [
        ("neso-half-hourly.csv", ["settlement_date", "settlement_period", "national_demand_mw"], selected),
        ("neso-hourly.csv", ["hour_start_utc", "national_demand_mw"], hourly),
    ]:
        with (HERE / name).open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(values)
        print(name, len(values), hashlib.sha256((HERE / name).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
