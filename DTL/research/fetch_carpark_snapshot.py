"""Bounded original Carpark API reader for a future, separately locked trial.

No crawl is run by this module. UTC instants are converted to Singapore for the
API's date_time parameter. Original bytes may be cached gzip-compressed with
both original and compressed SHA256; dynamic capacities are retained as fields.
"""
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path

IDS = ("RHM", "TGM2", "PM28", "TM43", "KLM", "SE50", "JBM", "KBM")
SG = timezone(timedelta(hours=8))
API = "https://api.data.gov.sg/v1/transport/carpark-availability"


def _datetime(value):
    parsed = datetime.fromisoformat(value)
    # The original API has explicit offsets on item timestamps, and historically
    # naive SG-local strings in update_datetime. Never interpret naive as UTC.
    return parsed.replace(tzinfo=SG) if parsed.tzinfo is None else parsed


def parse_snapshot(payload, requested_at, ids=IDS, stale_after_minutes=10):
    """Parse only C-lot availability; invalid/stale/missing values remain None."""
    if requested_at.tzinfo is None:
        raise ValueError("requested_at must explicitly specify a timezone")
    body = json.loads(payload)
    candidates = [item for item in body.get("items", [])
                  if _datetime(item["timestamp"]) <= requested_at]
    item = max(candidates, key=lambda v: _datetime(v["timestamp"])) if candidates else None
    entries = {} if item is None else {v["carpark_number"]: v for v in item.get("carpark_data", [])}
    rows = {}
    for ident in ids:
        record = entries.get(ident)
        row = {"available": None, "capacity": None, "valid": False, "reason": "missing_id"}
        if record is not None:
            update = _datetime(record["update_datetime"])
            row["update_datetime"] = update.isoformat()
            slots = [v for v in record.get("carpark_info", []) if v.get("lot_type") == "C"]
            if update > requested_at:
                row["reason"] = "future_update"
            elif requested_at - update > timedelta(minutes=stale_after_minutes):
                row["reason"] = "stale_update"
            elif len(slots) != 1:
                row["reason"] = "missing_or_duplicate_C_type"
            else:
                try:
                    availability = int(slots[0]["lots_available"])
                    capacity = int(slots[0]["total_lots"])
                    if not 0 <= availability <= capacity:
                        row["reason"] = "invalid_range"
                    else:
                        row.update(available=availability, capacity=capacity, valid=True, reason=None)
                except (KeyError, TypeError, ValueError):
                    row["reason"] = "invalid_numeric_fields"
        rows[ident] = row
    return {"requested_at": requested_at.isoformat(),
            "snapshot_timestamp": item["timestamp"] if item is not None else None,
            "lot_type": "C", "stale_after_minutes": stale_after_minutes, "carparks": rows}


def fetch_snapshot(session, requested_at, cache_dir=None, max_response_bytes=1_000_000,
                   ids=IDS, stale_after_minutes=10):
    """Fetch at most one MB, with no automatic bulk requests or retries."""
    if requested_at.tzinfo is None:
        raise ValueError("requested_at must explicitly specify a timezone")
    parameter = requested_at.astimezone(SG).replace(tzinfo=None).isoformat(timespec="seconds")
    response = session.get(API, params={"date_time": parameter}, stream=True, timeout=(15, 30))
    response.raise_for_status()
    parts, size = [], 0
    try:
        for block in response.iter_content(chunk_size=64 * 1024):
            size += len(block)
            if size > max_response_bytes:
                raise ValueError("snapshot exceeds bounded response byte budget")
            parts.append(block)
    finally:
        response.close()
    payload = b"".join(parts)
    result = parse_snapshot(payload, requested_at, ids, stale_after_minutes)
    result["provenance"] = {
        "url": response.url, "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "raw_bytes": len(payload), "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "asof_limitation": "historical snapshot endpoint, not independently verified immutable publication vintage",
    }
    if cache_dir is not None:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (requested_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json.gz")
        if path.exists():
            raise FileExistsError("refusing to replace an already cached original response")
        compressed = gzip.compress(payload, mtime=0)
        path.write_bytes(compressed)
        result["provenance"].update(cached_file=str(path.resolve()),
                                    cached_bytes=len(compressed),
                                    cached_sha256=hashlib.sha256(compressed).hexdigest())
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
