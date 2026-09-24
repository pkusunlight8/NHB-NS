"""Run the public core analyses or refit models from licensed source records."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "src"
DATA = ROOT / "data"


def invoke(name, output, *arguments):
    env = os.environ.copy()
    env["NHB_OUTPUT_DIR"] = str(output)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NUMBA_CACHE_DIR"] = str(output / "tmp")
    subprocess.run([sys.executable, "-B", str(SCRIPT / name), *map(str, arguments)], env=env, check=True)


def prepare_public_inputs(output):
    for endpoint in ("independence", "multidomain"):
        destination = output / "analysis" / f"imach_{endpoint}"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATA / "model_parameters" / f"{endpoint}.npz", destination / "fit.npz")
        for stem in ("p00", "p01", "p10", "p11"):
            native = output / "analysis" / f"native_{endpoint}" / stem
            native.mkdir(parents=True, exist_ok=True)
            shutil.copy2(DATA / "native_imach" / endpoint / stem / f"E_{stem}.txt", native / f"E_{stem}.txt")


def verify_public(output):
    import pandas as pd

    clinical = pd.read_csv(output / "results" / "independence_inequality.csv")
    clinical = clinical.query('contrast == "joint" and metric == "favourable"').set_index("female")
    prediction = pd.read_csv(output / "results" / "prediction_metrics.csv")
    prediction = prediction.query('fold == "All"').set_index(["target", "model"])
    need = json.loads((output / "results" / "need_allocation_summary.json").read_text(encoding="utf-8"))
    native = pd.read_csv(output / "results" / "native_imach_expectancies.csv")

    checks = {
        "male_independent_year_gap": float(clinical.loc[0, "estimate"]),
        "female_independent_year_gap": float(clinical.loc[1, "estimate"]),
        "economic_prediction_r2": float(prediction.loc[("Equal-weight service score", "Economic baseline"), "r_squared"]),
        "pca_economic_prediction_r2": float(prediction.loc[("Training-fold principal component", "Economic baseline"), "r_squared"]),
        "need_benchmark_targets": int(need["n_site_years"]),
        "need_matched_targets": int(need["with_need_matched_peer"]),
        "need_candidate_pairs": int(need["without_need_matching_pair_count"]),
        "stable_matched_pairs": int(need["stable_pairs"]),
        "median_gap_percent": float(need["gap_median_among_eligible"]),
        "maximum_imach_difference_years": float(native.absolute_difference.max()),
    }
    assert abs(checks["male_independent_year_gap"] - 3.45) < 0.01
    assert abs(checks["female_independent_year_gap"] - 3.66) < 0.01
    assert abs(checks["economic_prediction_r2"] - 0.7055745248015197) < 1e-6
    assert abs(checks["pca_economic_prediction_r2"] - 0.715799739292219) < 1e-6
    assert checks["need_benchmark_targets"] == 167
    assert checks["need_matched_targets"] == 74
    assert checks["need_candidate_pairs"] == 284
    assert checks["stable_matched_pairs"] == 31
    assert abs(checks["median_gap_percent"] - 47.8) < 0.1
    assert checks["maximum_imach_difference_years"] < 0.0002
    (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    public = commands.add_parser("public", help="Recompute key results from aggregate public inputs")
    public.add_argument("--output", type=Path, default=ROOT / "outputs")
    fit = commands.add_parser("fit", help="Refit the monthly model from licensed participant inputs")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--intervals", type=Path, required=True)
    fit.add_argument("--windows", type=Path, required=True)
    fit.add_argument("--charls-core", type=Path, required=True)
    fit.add_argument("--city-panel", type=Path, default=DATA / "city_year_light_services.csv")
    fit.add_argument("--imach-exe", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == "public":
        prepare_public_inputs(output)
        for endpoint in ("independence", "multidomain"):
            invoke("07_expectancy.py", output, endpoint)
        invoke("11_verify_native.py", output)
        invoke("01_service_prediction.py", output)
        invoke("05_benchmark_extensions.py", output)
        invoke("09_need_allocation.py", output)
        verify_public(output)
    else:
        invoke("02_prepare_health.py", output, "--intervals", args.intervals, "--windows", args.windows,
               "--charls-core", args.charls_core, "--city-panel", args.city_panel)
        for endpoint in ("independence", "multidomain"):
            invoke("03_fit_imach_likelihood.py", output, endpoint)
            invoke("07_expectancy.py", output, endpoint)
            if args.imach_exe:
                invoke("08_native_imach.py", output, endpoint, "--imach-exe", args.imach_exe)
        if args.imach_exe:
            invoke("11_verify_native.py", output)


if __name__ == "__main__":
    main()
