import argparse
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _pick_problem_type_field(table_schema) -> Optional[Any]:
    try:
        import pyarrow as pa
    except Exception:
        return None

    if table_schema.get_field_index("extra_info") != -1:
        extra = table_schema.field("extra_info")
        if pa.types.is_struct(extra.type) and extra.type.get_field_index("problem_type") != -1:
            return ("extra_info", "problem_type")
    if table_schema.get_field_index("problem_type") != -1:
        return ("problem_type",)
    return None


def _get_problem_type_from_row(row: Dict[str, Any], field_path: Sequence[str]) -> str:
    cur: Any = row
    for k in field_path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(k)
    return cur if isinstance(cur, str) else ""


def _reservoir_sample_records(
    parquet_path: str,
    *,
    k: int,
    seed: int,
    allowed_problem_types: Sequence[str],
) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    field_path = _pick_problem_type_field(schema)
    if field_path is None:
        raise ValueError(f"Cannot find problem_type field in parquet schema: {parquet_path}")

    rng = random.Random(seed)
    allowed = set(allowed_problem_types)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    for batch in pf.iter_batches():
        rows = batch.to_pylist()
        for row in rows:
            pt = _get_problem_type_from_row(row, field_path)
            if pt not in allowed:
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append(row)
            else:
                j = rng.randrange(seen)
                if j < k:
                    reservoir[j] = row

    if not reservoir:
        raise ValueError(f"No rows matched problem_type in {allowed_problem_types} from {parquet_path}")
    return reservoir


def _align_table_to_schema(table, schema):
    import pyarrow as pa

    cols: Dict[str, Any] = {}
    for field in schema:
        name = field.name
        if name in table.column_names:
            arr = table[name]
            cols[name] = arr
        else:
            cols[name] = pa.nulls(table.num_rows, type=field.type)

    aligned = pa.table(cols, schema=schema)
    return aligned


def _write_concat_parquet(
    *,
    base_parquet_path: str,
    append_records: List[Dict[str, Any]],
    output_parquet_path: str,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if os.path.exists(base_parquet_path):
        base_pf = pq.ParquetFile(base_parquet_path)
        schema = base_pf.schema_arrow
    else:
        schema = None

    append_table = pa.Table.from_pylist(append_records)
    if schema is None:
        schema = append_table.schema
    append_table = _align_table_to_schema(append_table, schema)

    tmp_path = output_parquet_path + ".tmp"
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)

    with pq.ParquetWriter(tmp_path, schema=schema) as writer:
        if os.path.exists(base_parquet_path):
            for batch in pq.ParquetFile(base_parquet_path).iter_batches():
                writer.write_batch(batch)
        writer.write_table(append_table)

    os.replace(tmp_path, output_parquet_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videor1_parquet", required=True)
    parser.add_argument("--videoautor1_parquet", required=True)
    parser.add_argument("--output_parquet", default=None)
    parser.add_argument("--sample_size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--problem_types",
        default="ocr,free-form,regression",
        help="Comma-separated problem_type values to sample",
    )
    args = parser.parse_args()

    videor1_path = args.videor1_parquet
    videoautor1_path = args.videoautor1_parquet
    output_path = args.output_parquet or videoautor1_path

    allowed_problem_types = [x.strip() for x in args.problem_types.split(",") if x.strip()]
    sampled = _reservoir_sample_records(
        videor1_path,
        k=args.sample_size,
        seed=args.seed,
        allowed_problem_types=allowed_problem_types,
    )

    _write_concat_parquet(
        base_parquet_path=videoautor1_path,
        append_records=sampled,
        output_parquet_path=output_path,
    )


if __name__ == "__main__":
    main()

