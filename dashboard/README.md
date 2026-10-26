# Hosted ELBench runner

Visitors bring an OpenRouter key, pick a model, and run a subset of ELBench.
The Space process calls `BenchmarkRunner`. Prompts, retries, and scores are the
ones that runner already uses.

The Hugging Face config is the YAML block at the top of the repository
[README.md](../README.md), not this file. Spaces read `README.md` at the
repository root. `app_file` is `dashboard/app.py`. `python_version` is `3.12`
because the package requires Python 3.11 or newer and the Space default is
3.10. `sdk_version` is `6.4.0`, the Gradio version this app was run against.
`suggested_hardware: cpu-basic` is only a suggestion for someone duplicating
the Space. It does not assign hardware.

## Host

Checked against the Spaces overview, configuration reference, and GPU docs:

- CPU Basic is 2 vCPU, 16 GB RAM, and 50 GB of disk, and its hourly price is free.
- Creating a Gradio or Docker Space requires a paid plan: PRO for a personal account, Team or Enterprise for an organization. Static Spaces are the ones that are free for every account. A free personal account can host up to two Gradio Spaces on ZeroGPU. ZeroGPU is not this app.
- CPU Basic sleeps after 48 hours without a visit. A visit starts it again. Custom sleep times require paid hardware.
- The disk is not persistent. A restart drops `outputs/`.
- Dependencies come from the root `requirements.txt`. Judge credentials belong in Space Settings as secrets. Gradio Spaces expose those secrets as environment variables. Set `QWEN_TEST_API_KEY` there for the official judge. Do not point judged tasks at a different model.
- `demo.queue(default_concurrency_limit=1)` runs at import. The Space runtime launches the imported demo and does not execute the `if __name__` block, so the one-run limit has to be set on the object itself.

`高阶育人-edu.jsonl` is scored by the rule judge, and preflight still requires the judge key. That hold is kept.

## OpenRouter

Checked against the quickstart, chat-completions reference, models list, provider routing, FAQ, and limits pages, plus a live `GET /api/v1/models`:

- Requests are `POST https://openrouter.ai/api/v1/chat/completions` with `Authorization: Bearer`. That matches `configs/providers.yaml` `endpoint_path: /chat/completions`.
- The model field is the slug the visitor selected, including a `:nitro` or `:floor` suffix. Those two suffixes are routing variants, not catalog rows. `:free` and `:batch` are catalog rows and are not rewritten.
- `pricing.prompt` and `pricing.completion` are USD per token, returned as strings. The live catalog had no `pricing.request` field. A negative price is treated as unknown.
- The model list answers without a key. If the typed key is rejected, the loader retries the public list.
- Default routing load-balances providers and falls back to another provider of the same slug. Unsupported sampling parameters are ignored unless `require_parameters` is set. This app does not send a `provider` object, and the estimate says so.
- A `:free` slug is capped at 20 requests per minute, and at 50 per day until the account has purchased at least 10 credits, then 1,000 per day. Paid slugs are limited by credits, the in-flight spending budget, the upstream provider, and DDoS protection.
- `HTTP-Referer` and `X-OpenRouter-Title` are optional. They attribute the app on OpenRouter rankings. They are not required for the call, and the client does not add them.

## What the visitor can change

- OpenRouter key and model id
- modules, source files, dimensions, and `max_samples`
- concurrency from 1 to 100 (the hosted default is 8; `configs/app.yaml` is 100)

Temperature, max tokens, streaming, provider kwargs, prompts, and judges are not form fields.

## Key handling

The key is held in `ELBENCH_VISITOR_OPENROUTER_API_KEY` for the duration of the estimate or run, then removed. It is not written to `configs/` or `key.txt`. The basic-education executor does write it into `outputs/basic_education/<run_id>/.../basic_education_runtime.yaml`, because that runtime reads the key from the rendered config. The app deletes that run directory when the run returns, including on failure. A killed process can leave the file on disk.

## Local use

```bash
pip install -e ".[basic-education,dashboard]"
python dashboard/app.py
```
