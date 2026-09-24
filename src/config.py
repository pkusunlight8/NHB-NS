"""Repository-relative locations shared by the core analysis scripts."""

from pathlib import Path
import hashlib
import json
import os

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "data"
RUN = Path(os.environ.get("NHB_OUTPUT_DIR", PACKAGE / "outputs")).resolve()
SEED = 20260921

SERVICES = ["doctors_per_10000", "beds_per_10000", "buses_per_10000"]
SES = ["gdp_per_registered_resident_z", "retail_per_registered_resident_z", "urbanization_percent_z"]
SATELLITE = [
    "rad_p90_log_z", "rad_p99_log_z", "rad_auc_log_z", "rad_max_log_z",
    "bright_core_gt10_fraction_z", "lit_area_gt0_fraction_z", "bright_night_days_gt5_z",
    "consecutive_bright_run_gt5_z", "monthly_amplitude_log_z", "intra_year_cv_z",
    "winter_summer_contrast_z", "seasonal_phase_winter_peak_z", "radiance_gini_z",
    "radiance_entropy_z", "very_bright_gt50_fraction_z", "bright_core_concentration_z",
]

for folder in ("analysis", "results", "qa", "tmp"):
    (RUN / folder).mkdir(parents=True, exist_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, obj):
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item) for item in value]
        return value.item() if hasattr(value, "item") else value

    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")
