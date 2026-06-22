# Storage Strategy

The current repository can grow very quickly because raw IMU recordings and exported jump segments are stored as text CSV files. A local audit showed that CSV files dominate disk usage by far.

## Recommended Direction

Use a two-tier storage model:

1. Keep one canonical raw recording per sensor session.
2. Store jump annotations as lightweight metadata that points to the raw recording plus takeoff/landing frame indexes.
3. Build compact training windows in a cache format such as compressed `npz` or Parquet only when training needs them.
4. Rebuild the training cache whenever the annotation file, raw file timestamp, or window configuration changes.

This avoids duplicating hundreds of CSV rows for every candidate jump while keeping training fast.

## Practical Migration Path

The safest migration is incremental:

1. Keep existing CSV readers working.
2. Add a compact segment writer that stores numeric arrays as `float32` in compressed `npz`.
3. Teach training and GUI loaders to prefer compact segments when present, and fall back to CSV.
4. Add a maintenance command to convert old `data/pending/segments` and `data/annotated/**` segments in batches.
5. Once validated, stop writing per-jump CSV by default and write annotation metadata plus compact cache files.

## Expected Impact

CSV text is convenient but expensive. Converting repeated segment windows to `float32` arrays typically reduces size substantially and speeds loading because the training loader avoids CSV parsing. Keeping raw sessions canonical also prevents the same IMU samples from being duplicated across pending, annotated, and cache folders.

## Current Quick Wins

- Training already uses a cache, so repeated training runs should avoid reparsing every segment.
- Video files opened from another drive or network path are copied to `.tmp/video_cache` and read locally.
- Impossible IMU spikes are interpolated before detection/training so one corrupted row does not dominate plots or models.
