"""Cut OS Open UPRN (GB CSV zip) down to a box.

    python filter_uprn.py <osopenuprn_csv.zip> <south> <west> <north> <east> <out.csv>

Source: OS Open UPRN, Open Government Licence v3 (contains OS data).
"""
import csv
import io
import sys
import zipfile


def main() -> int:
    src = sys.argv[1]
    south, west, north, east = map(float, sys.argv[2:6])
    out = sys.argv[6]
    kept = 0
    with zipfile.ZipFile(src) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as raw, open(out, "w", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                lat, lon = float(row["LATITUDE"]), float(row["LONGITUDE"])
                if south <= lat <= north and west <= lon <= east:
                    writer.writerow(row)
                    kept += 1
    print(f"kept {kept} UPRNs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
