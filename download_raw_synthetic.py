
from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict, load_dataset


DATASETS = {
    "transaction.csv": "margiela00/synthetic_data_v2",
    "customer.csv": "margiela00/synthetic_customer_360_v1",
}


def export_first_split(dataset_name: str, output_path: Path) -> None:
    ds = load_dataset(dataset_name)

    if isinstance(ds, DatasetDict):
        split_name = next(iter(ds.keys()))
        table = ds[split_name].to_pandas()
    else:
        table = ds.to_pandas()

    table.to_csv(output_path, index=False)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, dataset_name in DATASETS.items():
        output_path = output_dir / filename
        print(f"Loading {dataset_name}...")
        export_first_split(dataset_name, output_path)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()