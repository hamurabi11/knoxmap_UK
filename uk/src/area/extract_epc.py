"""Pull one local authority's EPC certificates out of the national zips.

    python extract_epc.py <domestic-csv.zip> <non-domestic-csv.zip> <LA code> <out dir>

Writes epc_domestic.csv and epc_nondomestic.csv. Address and postcode
fields are dropped on purpose: they are Royal Mail's, not OGL, and the join
is by UPRN only.
Source: MHCLG Energy Performance of Buildings Data, OGL v3 (non-address fields).
"""
import csv
import io
import os
import sys
import zipfile

DROP = {"address", "address1", "address2", "address3", "postcode", "posttown"}


def extract(zip_path: str, la: str, out_path: str) -> int:
    kept = 0
    writer = None
    with zipfile.ZipFile(zip_path) as z, open(out_path, "w", newline="", encoding="utf-8") as out:
        for name in sorted(n for n in z.namelist() if n.startswith("certificates")):
            with z.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
                cols = [c for c in reader.fieldnames if c not in DROP]
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=cols + ["source_file"],
                                            extrasaction="ignore")
                    writer.writeheader()
                n = 0
                for row in reader:
                    if row.get("local_authority") == la:
                        row["source_file"] = name
                        writer.writerow(row)
                        n += 1
                kept += n
                print(f"  {os.path.basename(zip_path)} {name}: {n}", flush=True)
    return kept


def main() -> int:
    dom, nondom, la, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    print("domestic:", extract(dom, la, os.path.join(out, "epc_domestic.csv")), flush=True)
    print("non-domestic:", extract(nondom, la, os.path.join(out, "epc_nondomestic.csv")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
