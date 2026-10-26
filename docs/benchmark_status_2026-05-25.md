# Benchmark Status 2026-05-25

## Current Scope

- Cheap relay models currently evaluated without `安全可信`
- Active modules for relay runs:
  - `通用模型`
  - `高阶育人`
  - `基本教育`
- Reason:
  - InnoSpark relay may pre-filter safety prompts, which makes `安全可信` unfair for official comparison

## Endpoint Notes

- Cheap relay models:
  - `https://api.innospark.cn/v1`
  - key env: `INNOSPARK_RELAY_API_KEY`
- GPT/Claude support endpoint:
  - `http://35.220.164.252:3888/v1`
  - key env: `EXTERNAL_SUPPORT_API_KEY`
- Judge / evaluate / student endpoint:
  - current `qwen-test` notebook `9501` endpoint became unstable and later returned `404`
  - this must be repaired or replaced before further official judged runs

## Officially Usable Results

- `gpt-5.4`
  - full official run completed cleanly
  - run id: `full-gpt-5.4-20260519-v2`
- `gemini-3-flash-preview`
  - `no-safety` relay run completed
  - run id: `fair-gemini-3-flash-preview-nosafety-20260521-v1`
- `deepseek-v3.2`
  - `no-safety` relay run mostly completed
  - run id: `fair-deepseek-v3.2-nosafety-20260520-v1`
  - still worth a final cleanup check before treating as final

## Dirty / Incomplete Runs

- `gpt-5.2-pro`
  - run id: `full-gpt-5.2-pro-20260520-v1`
  - interrupted by quota exhaustion
- `doubao-seed-2-0-pro-260215`
  - run id: `fair-doubao-seed-2-0-pro-260215-nosafety-20260521-v1`
  - interrupted by local checkpoint write `PermissionError`
- `kimi-k2.6`
  - runs:
    - `fair-kimi-k2.6-nosafety-20260521-v1`
    - `fair-kimi-k2.6-nosafety-20260521-v2`
    - `fair-kimi-k2.6-nosafety-20260521-v3`
  - key findings:
    - relay has hard rate limit around `1 request/second`
    - model-specific throttling was added: `qps: 1`, `max_concurrency: 1`
    - later hit quota exhaustion, then resume attempts became messy
    - checkpoint showed many completed items, but `raw/judged` artifacts were not present
  - conclusion:
    - current Kimi runs are not directly salvageable as official final results
- `deepseek-r1-250528`
  - official run:
    - `fair-deepseek-r1-250528-nosafety-20260524-v1`
  - probe run:
    - `probe-deepseek-r1-250528-nosafety-20260525-c2t480-v1`
  - key findings:
    - high `ReadTimeout` rate under prior settings
    - later probe failures were caused by broken `qwen-test` judge endpoint `404`
  - conclusion:
    - not usable as final result yet

## Remaining Models To Finish Cleanly

- `doubao-seed-2-0-pro-260215`
- `kimi-k2.6`
- `deepseek-r1-250528`
- `gemini-3.1-pro-preview`
- optional later if quota allows:
  - `gpt-5.2-pro`

## Important Technical Findings

- `kimi-k2.6` needs special low-rate execution
- `deepseek-r1-250528` likely needs lower concurrency and longer timeout than the default cheap-model settings
- judge endpoint health must be verified before any further official judged run
- for any future run, do not trust checkpoint alone
- future official runs must confirm all of these are being written:
  - `outputs/raw_responses/...`
  - `outputs/judged_results/...`
  - `outputs/logs/...`
  - `outputs/summaries/...`

## Next Workstream

- Repair or replace the `qwen-test` judge endpoint
- Run a full-chain smoke test
- Start organizing completed clean results into:
  - standardized result folders
  - report tables
  - leaderboard
