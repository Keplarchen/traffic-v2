"""Parse the GEANT directed traffic matrix .tgz into NumPy arrays.

Reads SNDlib XML demand matrices from
'directed-geant-uhlig-15min-over-4months-ALL.tgz' and produces:

    flows.npy           [T, 462]  float32          traffic in Mbps
    timestamps.npy      [T]       datetime64[m]    sample times
    sd_pair_names.npy   [462]     '<U...'          'src->tgt' labels

Self-loops (i->i) are dropped. Demands missing from any single XML
are treated as 0.
"""

import argparse
import re
import tarfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

NS = {"sndlib": "http://sndlib.zib.de/network"}
TS_RE = re.compile(r"demandMatrix-geant-uhlig-15min-(\d{8})-(\d{4})\.xml$")


def parse_timestamp(name):
    m = TS_RE.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


def get_node_ids(root):
    nodes = root.find("sndlib:networkStructure/sndlib:nodes", NS)
    return [n.attrib["id"] for n in nodes.findall("sndlib:node", NS)]


def iter_demands(root):
    demands = root.find("sndlib:demands", NS)
    if demands is None:
        return
    for d in demands.findall("sndlib:demand", NS):
        src = d.find("sndlib:source", NS).text.strip()
        tgt = d.find("sndlib:target", NS).text.strip()
        val = float(d.find("sndlib:demandValue", NS).text.strip())
        yield src, tgt, val


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tgz",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "directed-geant-uhlig-15min-over-4months-ALL.tgz",
        help="Path to the .tgz archive",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Output directory for .npy files",
    )
    args = parser.parse_args()

    tgz_path: Path = args.tgz
    out_dir: Path = args.out

    if not tgz_path.exists():
        raise FileNotFoundError(f"tgz not found: {tgz_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Scanning {tgz_path.name}")
    with tarfile.open(tgz_path, "r:gz") as tar:
        members = []
        for m in tar.getmembers():
            if not m.isfile():
                continue
            ts = parse_timestamp(m.name)
            if ts is None:
                continue
            members.append((ts, m.name))
    members.sort()
    T = len(members)
    if T == 0:
        raise RuntimeError("No demandMatrix XML files found inside the .tgz")
    print(
        f"      Found {T} XML files; "
        f"time range {members[0][0]} .. {members[-1][0]}"
    )

    print("[2/3] Reading first XML for canonical node ordering")
    with tarfile.open(tgz_path, "r:gz") as tar:
        with tar.extractfile(members[0][1]) as f:
            root = ET.parse(f).getroot()
            nodes = get_node_ids(root)
    N = len(nodes)
    sd_pairs = [(s, t) for s in nodes for t in nodes if s != t]
    P = len(sd_pairs)
    sd_idx = {pair: i for i, pair in enumerate(sd_pairs)}
    print(f"      {N} nodes -> {P} directed SD pairs (self-loops dropped)")
    print(f"      nodes: {nodes}")

    print(f"[3/3] Parsing {T} XML files")
    flows = np.zeros((T, P), dtype=np.float32)
    timestamps = np.empty(T, dtype="datetime64[m]")
    n_skipped = 0
    n_unknown_pair = 0

    with tarfile.open(tgz_path, "r:gz") as tar:
        for i, (ts, name) in enumerate(members):
            with tar.extractfile(name) as f:
                try:
                    root = ET.parse(f).getroot()
                except ET.ParseError as e:
                    print(f"      WARN: parse failure {name}: {e}")
                    n_skipped += 1
                    continue
            for src, tgt, val in iter_demands(root):
                idx = sd_idx.get((src, tgt))
                if idx is None:
                    n_unknown_pair += 1
                    continue
                flows[i, idx] = val
            timestamps[i] = np.datetime64(ts, "m")
            if (i + 1) % 1000 == 0 or (i + 1) == T:
                print(f"      {i+1}/{T}")

    if n_skipped:
        print(f"      WARNING: skipped {n_skipped} unparseable files")
    if n_unknown_pair:
        print(
            f"      WARNING: {n_unknown_pair} demands referenced unknown nodes "
            f"(treated as missing)"
        )

    np.save(out_dir / "flows.npy", flows)
    np.save(out_dir / "timestamps.npy", timestamps)
    np.save(
        out_dir / "sd_pair_names.npy",
        np.array([f"{s}->{t}" for s, t in sd_pairs]),
    )

    nz_frac = (flows > 0).mean()
    print()
    print(f"Saved to {out_dir}")
    print(f"  flows.npy           shape={flows.shape}  dtype={flows.dtype}")
    print(f"  timestamps.npy      shape={timestamps.shape}")
    print(f"  sd_pair_names.npy   shape=({P},)")
    print()
    print(
        f"Stats: min={flows.min():.3f}  max={flows.max():.3f}  "
        f"mean={flows.mean():.3f}  median={np.median(flows):.3f}"
    )
    print(f"Nonzero fraction: {nz_frac:.1%}")


if __name__ == "__main__":
    main()
