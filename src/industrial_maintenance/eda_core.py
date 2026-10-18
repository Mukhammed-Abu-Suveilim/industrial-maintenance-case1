"""Табличный EDA стандартной библиотекой; никаких моделей и p-value."""

from __future__ import annotations

import bisect
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_COLUMNS = [
    "id", "equipment", "operating_hours", "temperature_c", "vibration_mm_s",
    "pressure_bar", "load_ratio", "operator_log", "manual_id", "fault_type",
    "severity", "required_route", "horizon_days", "failure_within_30d",
]


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def normalize_log(value, mask_numbers=False):
    text = "<missing>" if value is None or not str(value).strip() else str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\d+(?:[.,]\d+)?", "<number>", text) if mask_numbers else text


def group_id(row):
    return hashlib.sha256(normalize_log(row["operator_log"]).encode("utf-8")).hexdigest()


def load_rows(root, config):
    path = root / config["dataset"]
    if checksum(path) != config["sha256"]:
        raise ValueError("Изменилась контрольная сумма CSV; сначала проверьте источник.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError("Схема CSV не соответствует исходным именам полей.")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Пустой набор или строки с неверным числом полей.")
    ids = [row["id"] for row in rows]
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("id должен быть непустым и уникальным.")
    if any(row["failure_within_30d"] not in {"0", "1"} or not row["fault_type"].strip() for row in rows):
        raise ValueError("Неверный формат целевых переменных.")
    return rows, {"dataset": config["dataset"], "sha256": config["sha256"],
                  "rows": len(rows), "columns": EXPECTED_COLUMNS, "bytes": path.stat().st_size}


def validate_membership(rows, membership):
    ids = [entry["id"] for entry in membership]
    if len(ids) != len(set(ids)) or set(ids) != {row["id"] for row in rows}:
        raise ValueError("Разбиение должно покрывать каждый id ровно один раз.")
    if {entry["split"] for entry in membership} != {"train", "validation", "test"}:
        raise ValueError("Нужны три непустые части; не подбирайте seed ради результата.")
    lookup = {entry["id"]: entry for entry in membership}
    seen = {}
    for row in rows:
        entry = lookup[row["id"]]
        expected = group_id(row)
        if entry["group_id"] != expected:
            raise ValueError("Группа не соответствует operator_log.")
        if expected in seen and seen[expected] != entry["split"]:
            raise ValueError("Одинаковые operator_log пересекают выборки.")
        seen[expected] = entry["split"]


def prepare_splits(root, rows, config):
    protocol = {key: config[key] for key in (
        "sha256", "seed", "n_splits", "test_fold", "validation_fold", "split_group", "split_method"
    )}
    if protocol["split_method"] != "sha256_group_folds" or protocol["split_group"] != "normalized_exact_operator_log":
        raise ValueError("Этот скрипт поддерживает только описанный протокол группового разбиения.")
    folds = config["n_splits"]
    if folds < 3 or not (0 <= config["test_fold"] < folds and 0 <= config["validation_fold"] < folds) or config["test_fold"] == config["validation_fold"]:
        raise ValueError("Некорректные номера частей разбиения.")
    folder = root / "Data/splits" / config["split_version"]
    path, manifest_path = folder / "membership.csv", folder / "manifest.json"
    if path.exists() != manifest_path.exists():
        raise ValueError("Разбиение сохранено не полностью.")
    if path.exists():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata["protocol"] != protocol or metadata["membership_sha256"] != checksum(path):
            raise ValueError("Протокол или файл разбиения изменён; запрещено молча перезаписывать эту версию.")
        with path.open(encoding="utf-8", newline="") as handle:
            membership = list(csv.DictReader(handle))
    else:
        if len({group_id(row) for row in rows}) < folds:
            raise ValueError("Недостаточно групп для разбиения.")
        membership = []
        for row in rows:
            gid = group_id(row)
            digest = hashlib.sha256(f"{config['seed']}|{gid}".encode("utf-8")).hexdigest()
            fold = int(digest, 16) % folds
            partition = "test" if fold == config["test_fold"] else "validation" if fold == config["validation_fold"] else "train"
            membership.append({"id": row["id"], "split": partition, "group_id": gid})
        validate_membership(rows, membership)
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(path, membership)
        metadata = {
            "protocol": protocol, "membership_sha256": checksum(path),
            "description": "Детерминированное групповое назначение SHA-256; метки не участвуют, стратификации нет.",
            "limitation": "После маскирования чисел общие тексты могут остаться. Не проверяет будущие периоды, новые машины или неизвестные текстовые шаблоны.",
            "sizes": dict(Counter(entry["split"] for entry in membership)),
        }
        manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate_membership(rows, membership)
    return membership, metadata


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def describe(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"n": 0, "min": None, "q01": None, "q25": None, "median": None,
                "q75": None, "q99": None, "max": None, "mean": None, "std": None}
    return {"n": len(values), "min": min(values), "q01": quantile(values, .01),
            "q25": quantile(values, .25), "median": quantile(values, .5), "q75": quantile(values, .75),
            "q99": quantile(values, .99), "max": max(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None}


def ranks(values):
    result = [0.] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for index in ordered[start:end]:
            result[index] = average
        start = end
    return result


def spearman(left, right):
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if len(pairs) < 2:
        return None, len(pairs)
    a, b = map(list, zip(*pairs))
    a, b = ranks(a), ranks(b)
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return covariance / denominator if denominator else None, len(pairs)


def overlap_table(rows, membership):
    lookup = {entry["id"]: entry["split"] for entry in membership}
    partitions = {name: [row for row in rows if lookup[row["id"]] == name] for name in ("train", "validation", "test")}
    result = []
    for left, right in itertools.combinations(partitions, 2):
        a, b = partitions[left], partitions[right]
        result.append({"left": left, "right": right,
                       "shared_id": len({r['id'] for r in a} & {r['id'] for r in b}),
                       "shared_exact_log": len({group_id(r) for r in a} & {group_id(r) for r in b}),
                       "shared_masked_log": len({normalize_log(r['operator_log'], True) for r in a} & {normalize_log(r['operator_log'], True) for r in b})})
    return result


def _frequency(rows, field):
    return Counter(row[field] or "<missing>" for row in rows)


def build_tables(rows, membership, config):
    lookup = {entry["id"]: entry["split"] for entry in membership}
    train = [row for row in rows if lookup[row["id"]] == "train"]
    validation = [row for row in rows if lookup[row["id"]] == "validation"]
    tables = {}
    tables["quality"] = [{"field": field, "scope": "source_structure", "rows": len(rows),
                          "empty": sum(not row[field].strip() for row in rows),
                          "unique": len({row[field] for row in rows}),
                          "invalid_or_nonfinite_numeric": sum(bool(row[field].strip()) and number(row[field]) is None for row in rows) if field in config["numeric_columns"] else None}
                         for field in EXPECTED_COLUMNS]
    exact = Counter(tuple(row[field] for field in EXPECTED_COLUMNS if field != "id") for row in rows)
    logs = Counter(normalize_log(row["operator_log"]) for row in rows)
    tables["duplicates"] = [{"measure": "extra_full_rows_without_id", "count": sum(n - 1 for n in exact.values())},
                            {"measure": "unique_normalized_operator_log", "count": len(logs)},
                            {"measure": "extra_operator_log_instances", "count": sum(n - 1 for n in logs.values())},
                            {"measure": "unique_masked_operator_log", "count": len({normalize_log(row['operator_log'], True) for row in rows})}]
    tables["split_sizes"] = [{"split": name, "rows": sum(entry['split'] == name for entry in membership),
                              "groups": len({entry['group_id'] for entry in membership if entry['split'] == name})}
                             for name in ("train", "validation", "test")]
    tables["split_overlap"] = overlap_table(rows, membership)
    tables["target_balance"] = [{"target": target, "value": value, "n": count, "share": count / len(train), "scope": "train"}
                                for target in config["targets"] for value, count in _frequency(train, target).most_common()]
    joint = Counter((row['fault_type'], row['failure_within_30d']) for row in train)
    tables["target_relationship"] = [{"fault_type": fault, "failure_within_30d": failure, "n": count}
                                      for (fault, failure), count in sorted(joint.items())]
    tables["development_class_coverage"] = [{"target": target, "scope": scope, "value": value, "n": count}
                                            for scope, sample in [('train', train), ('validation', validation)]
                                            for target in config['targets'] for value, count in sorted(_frequency(sample, target).items())]
    numeric = config["numeric_columns"]
    tables["numeric_summary"], tables["outlier_flags"], tables["numeric_by_target"] = [], [], []
    for field in numeric:
        values = [number(row[field]) for row in train]
        summary = describe(values)
        tables["numeric_summary"].append({"field": field, **summary})
        if summary['n']:
            iqr = summary['q75'] - summary['q25']
            lower, upper = summary['q25'] - 1.5 * iqr, summary['q75'] + 1.5 * iqr
            count = sum(value is not None and (value < lower or value > upper) for value in values)
            tables["outlier_flags"].append({"field": field, "lower_iqr": lower, "upper_iqr": upper, "outside": count,
                                            "share_of_finite": count / summary['n'], "action": "Не удалять автоматически"})
        for target in config['targets']:
            for level in sorted({row[target] for row in train}):
                tables['numeric_by_target'].append({"field": field, "target": target, "class": level,
                                                   **describe([number(row[field]) for row in train if row[target] == level])})
    tables["categorical_counts"], tables["category_failure"] = [], []
    for field in config['categorical_columns']:
        for value, count in _frequency(train, field).most_common():
            subset = [row for row in train if (row[field] or '<missing>') == value]
            failures = sum(row['failure_within_30d'] == '1' for row in subset)
            tables['categorical_counts'].append({"field": field, "value": value, "n": count, "share": count / len(train)})
            tables['category_failure'].append({"field": field, "value": value, "n": count, "failures": failures, "failure_share": failures / count})
    tables['relationships'], tables['relationship_summary'] = [], []
    for left, right in [('equipment', 'manual_id'), ('fault_type', 'severity'), ('fault_type', 'required_route'), ('severity', 'failure_within_30d')]:
        pairs = Counter((row[left] or '<missing>', row[right] or '<missing>') for row in train)
        tables['relationships'].extend({"left_field": left, "right_field": right, "left_value": a, "right_value": b, "n": count}
                                       for (a, b), count in sorted(pairs.items()))
        forward, backward = defaultdict(set), defaultdict(set)
        for a, b in pairs:
            forward[a].add(b)
            backward[b].add(a)
        tables['relationship_summary'].append({"left_field": left, "right_field": right,
                                              "left_determines_right_on_train": all(len(values) == 1 for values in forward.values()),
                                              "right_determines_left_on_train": all(len(values) == 1 for values in backward.values()),
                                              "scope": "train; не причинность и не гарантия вне выборки"})
    tables['spearman'] = []
    columns = numeric + ['operator_log_length', 'failure_within_30d']
    numeric_values = {field: [float(len(row['operator_log'])) if field == 'operator_log_length' else number(row[field]) for row in train] for field in columns}
    for left, right in itertools.combinations(columns, 2):
        coefficient, n = spearman(numeric_values[left], numeric_values[right])
        tables['spearman'].append({"left": left, "right": right, "rho": coefficient, "n_pairs": n, "scope": "train"})
    tables['text_summary'] = [{"measure": 'operator_log_length', **describe([len(row['operator_log']) for row in train])},
                              {"measure": 'operator_log_words', **describe([len(row['operator_log'].split()) for row in train])}]
    templates = Counter(normalize_log(row['operator_log'], True) for row in train)
    tables['text_templates'] = [{"masked_text": text, "n": count} for text, count in templates.most_common()]
    tables['log_load_check'] = []
    for row in train:
        found = re.search(r'(?i)нагрузка\s+(\d+(?:[.,]\d+)?)\s*%', row['operator_log'])
        load = number(row['load_ratio'])
        if found and load is not None:
            stated = float(found[1].replace(',', '.'))
            tables['log_load_check'].append({"id": row['id'], "log_load_percent": stated, "load_ratio_percent": load * 100, "difference_percentage_points": stated - load * 100})
    tables['tokens'] = []
    words_per_row = [set(re.findall(r'[^\W\d_]{3,}', row['operator_log'].lower())) for row in train]
    total_words = Counter(word for words in words_per_row for word in words)
    for target in config['targets']:
        for level in sorted({row[target] for row in train}):
            inside = Counter(word for row, words in zip(train, words_per_row) if row[target] == level for word in words)
            n_inside = sum(row[target] == level for row in train)
            records = []
            for word, count in inside.items():
                if total_words[word] < config['min_token_documents']:
                    continue
                n_outside = len(train) - n_inside
                outside = total_words[word] - count
                ratio = ((count + 1) / (n_inside + 2)) / ((outside + 1) / (n_outside + 2))
                records.append({"target": target, "class": level, "word": word, "documents_in_class": count,
                                "class_documents": n_inside, "share_in_class": count / n_inside,
                                "share_outside_class": outside / n_outside if n_outside else None, "smoothed_share_ratio": ratio})
            tables['tokens'].extend(sorted(records, key=lambda record: (-record['smoothed_share_ratio'], record['word']))[:10])
    tables['development_comparison'], tables['development_numeric_bins'] = [], []
    for field in numeric:
        train_values = [value for row in train if (value := number(row[field])) is not None]
        edges = sorted({quantile(train_values, p / 10) for p in range(1, 10)}) if train_values else []
        for scope, sample in [('train', train), ('validation', validation)]:
            values = [number(row[field]) for row in sample]
            tables['development_comparison'].append({"field": field, "scope": scope, "kind": "numeric", **describe(values),
                                                    "empty": sum(not row[field].strip() for row in sample)})
            counts = Counter(bisect.bisect_left(edges, value) for value in values if value is not None)
            finite_n = sum(value is not None for value in values)
            for bucket in range(len(edges) + 1):
                tables['development_numeric_bins'].append({"field": field, "scope": scope, "bin": bucket,
                                                          "lower_exclusive": edges[bucket - 1] if bucket else None,
                                                          "upper_inclusive": edges[bucket] if bucket < len(edges) else None,
                                                          "n": counts[bucket], "share_of_finite": counts[bucket] / finite_n if finite_n else None})
    for field in config['categorical_columns']:
        known = set(_frequency(train, field))
        for scope, sample in [('train', train), ('validation', validation)]:
            for value, count in _frequency(sample, field).most_common():
                tables['development_comparison'].append({"field": field, "scope": scope, "kind": "categorical", "value": value,
                                                        "n": count, "share": count / len(sample), "unseen_in_train": value not in known})
    tables['feature_evidence'] = []
    for field in EXPECTED_COLUMNS:
        role = 'target' if field in config['targets'] else 'record_identifier' if field == 'id' else 'candidate'
        references = 'target_balance.csv' if role == 'target' else 'quality.csv;split_overlap.csv' if field == 'id' else 'numeric_summary.csv;numeric_by_target.csv;spearman.csv' if field in numeric else 'text_summary.csv;tokens.csv;text_templates.csv' if field == 'operator_log' else 'categorical_counts.csv;category_failure.csv;relationships.csv'
        status = 'Не вход модели по роли' if role != 'candidate' else 'Решение не принято'
        if field in {'severity', 'required_route', 'horizon_days'}:
            status = 'Ожидается описание доступности до прогноза'
        tables['feature_evidence'].append({"field": field, "role": role, "availability": config['availability'].get(field, 'not_applicable'),
                                          "evidence": references, "status": status, "team_decision": '', "decision_source": ''})
    return tables, train
