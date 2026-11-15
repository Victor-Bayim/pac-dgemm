# Historical records: not a build target

These files are copied byte-for-byte from the participant-supplied PAC evidence
archive, retained separately from the 2026-11-15 maintenance implementation.
No claim is made that this directory reconstructs the competition executable.

- `source/original/`: two incomplete source drafts. The NEON draft has typos,
  missing conditional terminators and no complete public entry; do not build it.
- `source/transcribed/dgemm.cpp`: a searchable transcription of 11 saved code
  screenshots, made on 2026-11-01. It includes the dispatcher, but is unverified
  historical material with panel-stride and tail assumptions. It is not the
  tested source of a recovered competition binary.
- `source/reference/`: supplied competition framework with its original headers
  and notes. It depends on the unavailable original `kblas.h` SDK and has API,
  reference-layout and repeated-C correctness problems. It is not used by the
  maintenance build or tests.
- `data/`: 16 stage throughput records, four rounded elapsed-time summaries, and
  worksheet transcriptions. Original values and wording are unchanged. In
  particular, historical wording about `-r=4` being warmup must not be read as
  the actual method: the retained driver selects the minimum of four durations.
- `docs/RUNNING.md`: archived settings, not a standalone reproduction recipe.
- `TRANSCRIPTION.md` and `ATTRIBUTION.md`: original source-status and attribution
  notes. Their screenshot map refers to the offline evidence package; the images
  and SDK/binaries are not redistributed here.

`manifest.json` records the SHA-256 of these copied payload files. Run
`python tools/verify_archive.py` from the repository root to verify their hashes
and recalculate the retained data. These checks do not execute DGEMM.
