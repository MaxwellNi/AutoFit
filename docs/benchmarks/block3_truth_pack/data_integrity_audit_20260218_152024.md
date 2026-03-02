# Block3 Data Integrity Audit

- generated_at_utc: **2026-02-18T15:20:24.723890+00:00**
- pointer: `/mnt/aiongpfs/projects/eint/repo_root/docs/audits/FULL_SCALE_POINTER.yaml`
- config: `/mnt/aiongpfs/projects/eint/repo_root/configs/block3_tasks.yaml`
- overall_pass: **True**

## Gate Checks

| check | passed | fails |
|---|---|---|
| None | None | 0 |
| None | None | 0 |
| None | None | 0 |
| None | None | 0 |
| None | None | 0 |
| None | None | 0 |
| None | None | 0 |
| None | None | 0 |
| None | None | 0 |
| None | None | 0 |
| None | None | 0 |

## Split Checks

| check | passed |
|---|---|
| train_before_val | True |
| val_before_test | True |
| embargo_non_negative | True |

## Leakage Policy Checks

| check | passed |
|---|---|
| all_targets_have_leak_group | True |
| groups_non_empty | True |

## Asset Snapshot

| asset | exists | file_count | total_size_bytes | schema_columns | fingerprint_sha256 |
|---|---|---:|---:|---:|---|
| offers_core_daily | True | 6 | 314487001 |  | `197d2cf831ede861e1631edd908e5fd95315907e4e5ef7df482840450531e51b` |
| offers_core_snapshot | True | 4 | 17582552006 |  | `0024425b35967825b3f9645ac4c77d8d910d5daa13bf4ac6e78ebd6eedde61a2` |
| offers_text | True | 2 | 20514631758 |  | `7ff0bcf924f1cd31a259d153cf27df1d2bed5ecbbef9cfdfb357d8f4710d5ff2` |
| edgar_store_full_daily | True | 42 | 10613645 |  | `dab4f48c64be49e18484df35d058056be9fd14094ece1f289bc893fe09cba014` |
| multiscale_full | True | 7 | 20702137 |  | `1ee924c98a7385f6f93d01bd7f2d330265a73718e358ee4b1a2e347427d54950` |
| snapshots_offer_day | True | 1 | 6557155 |  | `0d9ff5257463866095e6f787ff70bca70f15e48f4909dbb96b12a78b902b2622` |
| snapshots_cik_day | True | 1 | 4186639 |  | `f781830fd7e47c7beeb29ef9090216a7b2fb4550e3d4f537ba3c19835b7325ae` |
| analysis_dir | True | 28 | 6751159 |  | `03ba2bb93ace6e69929fc1143e083eff73713a73c7593b20c6572d8e29c73b7c` |
