# ELBench-live dashboard targets

Working list for the hosted dashboard (`dashboard/app.py`, `src/elbench/dashboard/`).
Each target records the issue, the evidence, the agreed resolution, and how we
know it is done. Work them in order; update **Status** as each one lands.

## Goal

1. Present the published ELBench results clearly.
2. Say plainly what ELBench measures.
3. Make it plug-and-play for anyone with an OpenRouter key to test a model and
   join the leaderboard by default, with fine-grained control available:
   item selection, judge and student models, and a budget overall and per
   component model.

## Decisions

- **D1 Scoring follows the paper.** Module and overall scores are computed the
  way the published README/paper table is, unless there is a documented reason
  to override it. Safety uses the item-weighted mean score, not pass rate
  (`leaderboard_FINAL.json` in the results dataset uses pass rate; the paper
  table does not).
- **D2 Chart first.** The leaderboard chart is the landing view. Explanation
  lives in an About tab. Advanced controls are collapsed by default.
- **D3 Participate by default.** Saving a run to the public leaderboard is on
  by default, with an opt-out. Only runs that cover the scored slice in full
  are plotted against the paper models.

## Phase 1 · Correctness

### T1 Refresh stale official per-file data
- **Issue:** `official_results_data.json` per-file stats predate the results
  dataset's audit fixes, so recomputing module scores from them does not
  reproduce the published table.
- **Evidence:** High-Level education file for Claude-Opus-4.8, DeepSeek-V4-Pro,
  DeepSeek-V4-Flash, Doubao-Seed-2.0-Pro, GLM-5.1 holds the pre-fix pass rate
  (e.g. DeepSeek-V4-Flash 63.2; post-fix 92.6 per
  `audit-judge-integrity/highedu_llmjudge_fix_summary.json`). High-Level gaps up
  to 15 points. Basic Education per-file data is stale for GPT-5.4 and
  DeepSeek-V4-Pro (`question.yaml` avg 11.36 on a 5-point scale).
- **Resolution:** Rebuild per-file stats from the audited files in
  `ZeroLoss-Lab/ELBench-results`; record the source files and date in the JSON.
- **Done when:** a test recomputes every module and overall score from
  per-file data and matches the published table within 0.1 for all 9 models.
- **Status:** done 2026-10-26. Safety, Basic Education, High-Level and Overall reproduce
  the paper within 0.1; General within 0.25, explained by GSM8K. Provenance is in the JSON's
  `provenance` block; `module_ci` and `overall_ci` added from `bootstrap_ci.json`.
  Test: `test_official_per_file_data_reproduces_the_published_table`.

### T2 One scoring function for paper models and visitor runs
- **Issue:** `compute_normalized_scores` aggregates differently from the paper.
- **Evidence:** Safety uses pass rate (paper: mean score; gap up to 2.7 pts).
  Basic Education is item-weighted (paper: simple mean of the 4 scenarios).
- **Resolution:** General and High-Level: item-weighted pass rate. Safety:
  item-weighted mean score. Basic Education: mean of the 4 scenario scores,
  each normalized to its rubric maximum. Overall: mean of the 4 modules.
- **Done when:** the T1 test uses this same function.
- **Status:** done 2026-10-26. `normalize_file_entry` + `module_scores_from_per_file`.
  Visitor runs also skip missing or out-of-range Basic Education scores
  (`basic_education_valid_means`), the rule that reproduces the audited values.

### T3 Keep partial runs off the frontier
- **Issue:** a run's score covers only the items it ran, but is plotted against
  full-benchmark scores.
- **Evidence:** a 5-item MMLU run is placed on the Overall frontier at 100%.
- **Resolution:** a run is plotted for a metric only if it ran every item in
  that metric's scope (all files of the module, or the full benchmark for
  Overall; no `max_samples` or dimension filter). Partial runs stay visible in
  the runs table with their scope.
- **Done when:** test: the 5-item run is excluded from Overall and General and
  appears on the per-file MMLU view only if that file ran in full.
- **Status:** done 2026-10-26. `run_covers_metric` compares judged counts per file with the
  published counts (which equal the catalog's 2,939). The chart summary says how many runs were
  left out. Runs now also record their dimension filter.

### T4 Remove the fixed-footprint "Estimated full benchmark cost"
- **Issue:** the metric applies 0.73M input / 1.5M output tokens to every
  model and leaves out judge cost; it is a price blend labelled as an estimate.
- **Evidence:** GPT-5.4: metric $24.3; the app's own estimator $3.49–$285.
- **Resolution:** drop it. Default cost axis is the 3:1 blended price per 1M
  tokens; input and output prices remain as options. Once real full runs
  record measured cost (T12), offer "measured cost of a full run".
- **Status:** done 2026-10-26 (measured-cost option waits on T12).

### T5 One dated price snapshot
- **Issue:** first render uses prices stored in the JSON; Refresh uses live
  prices, so points move (DeepSeek-V4-Pro $0.96 → $0.73).
- **Resolution:** fetch the live catalog once per process (cached, with
  timestamp), fall back to stored prices, and label the chart "Prices from
  OpenRouter, <date>".
- **Status:** done 2026-10-26. `price_snapshot()` in `dashboard/app.py`, 6-hour cache.

### T6 Show uncertainty
- **Issue:** the top six are within 2 points with overlapping 95% intervals,
  but the chart presents small differences as wins.
- **Resolution:** error bars from the published intervals; a plain note when
  the leaders are statistically tied.
- **Status:** done 2026-10-26 for the current chart: error bars and a "statistically tied" line.
  Carry both into the T13 chart.

### T7 Hygiene
- `README.md` declares `sdk_version: 6.30.0`; 6.4.0 is installed and
  `dashboard/README.md` says 6.4.0. Pick one.
- `.env` holds secrets and is not git-ignored.
- **Status:** done 2026-10-26. README set to 6.4.0 (the version pinned in `pyproject.toml` and
  installed); `.env` added to `.gitignore`.

## Phase 2 · Presentation and participation

### T8 Three tabs: Leaderboard · Test a model · About
- Leaderboard: one-line description, chart, module focus switch, short
  takeaway, call to action to test a model; tables collapsed below.
- About: what ELBench measures (four modules in plain words), how scoring
  works, links, and the fidelity notes now in the header.
- **Status:** done 2026-10-26. `gr.Tabs` with ids; "Test your own model →" switches tabs. About is built from
  `AREA_LABELS` and the catalog counts.

### T9 Plain language
- Replace jargon in visible labels ("Endpoint to test", "Judge endpoint",
  "/api/v1/systemone", "preflight", "build_summary", "Matched Subset",
  "Pareto-optimal") and drop meta commentary from the main flow. Technical
  detail stays available in the estimate's details and the About tab.
- **Status:** done 2026-10-26. Visible labels use area names and plain task names (`area_label`, `plain_file`);
  the technical plan moved to a collapsed "Technical details" accordion.

### T10 Plug-and-play run with sensible defaults
- Key → model → preset (Quick check / Standard / Full) → estimate → run.
  Defaults: official judge, save to leaderboard on. One confirmation.
- **Status:** done 2026-10-26. Presets Quick check (5 questions, default) / Automatically checked set /
  Full benchmark; $5 default limit; leaderboard opt-in on; key checked with `GET /api/v1/key` (shows credit
  left); model list loaded on page load from the cached OpenRouter snapshot.

### T11 Advanced controls, collapsed
- Judge model and System 1 judging, simulated-student model, files and
  dimensions, item cap, concurrency, resume, leaderboard opt-out.
- **Status:** done 2026-10-26. "Advanced settings" accordion holds judge / decision-model judging, simulated
  student, per-component limits, areas/tasks/skills, question cap, concurrency, resume, unpriced models.

### T12 Budgets: overall and per component model
- **Components:** model under test, judge, simulated student (multi-turn).
- **Before the run:** show the estimate per component against its budget;
  block the run when an estimate's low end already exceeds a budget.
- **During the run:** meter spend per component from the `usage.cost` that
  OpenRouter returns; when a component or the total reaches its cap, stop
  through the runner's existing quota-exhausted path, so the run is saved as
  partial and can be resumed. Overshoot is bounded by requests already in flight.
- **Done when:** unit test stops a run at a cap and leaves it resumable; a
  live check with a tiny cap stops after the first requests.
- **Status:** done 2026-10-26. `elbench.utils.spend` ledger shared with the runtime subprocess; checks in
  `OpenAICompatibleClient`, `SystemOneJudge`, and a LangChain `SpendCallback`. The estimate blocks a run when a
  low estimate exceeds a limit. Unit tests: `tests/test_spend_limits.py`, `tests/test_spend_runtime_callback.py`.
  Live check (GPT-6 Luna, GLM 5.3 Flash judge, $0.003 total): total limit paused at 3/5 and resumed to 5/5;
  judge limit paused at 2/3 with spend split by role; key absent from every output file. Overshoot seen:
  limit $0.0004, spent $0.0011 (requests in flight finish); the estimate now says so.

## Phase 3 · Filters and chart

### T13 Interactive chart
- Plotly via `gr.Plot`: hover details, log price axis, error bars, legend
  for paper models vs visitor runs.
- **Status:** done 2026-10-26. `_render_pareto_figure` returns a Plotly figure: hover shows score, 95% range,
  price and verdict; log price axis with plain-dollar ticks; light error bars; published = blue circles,
  visitor runs = orange diamonds; best-value step line. Labels are placed to avoid dots, whiskers, the line
  and each other, and are dropped (hover still names the point) when no spot is free. Fixed a label bug:
  chart title and axis showed metric keys (`overall`, `blended_per_1m`) instead of their names.
  `plotly` added to `requirements.txt` and the `dashboard` extra.

### T14 Price and size filters
- OpenRouter `/api/v1/models` has prices but no parameter count. 179 of 468
  models carry `hugging_face_id`; the Hugging Face API returns
  `safetensors.total` (checked: Qwen3-235B → 235.09B). Closed models have no
  size, so the size filter gets an "include undisclosed" toggle. Filter on
  total size and show active size when the model name gives it (`-A22B`).
  Cache sizes; do not fetch on every page load.
- Apply the filters to both the chart and the model picker.
- **Status:** done 2026-10-26. `elbench.dashboard.model_sizes`: sizes from Hugging Face `safetensors.total`
  cached in `model_sizes.json` (152 repos; refresh with `python -m elbench.dashboard.model_sizes`), with the
  size in the model id as fallback and for active size. 176 of 468 OpenRouter models have a known size.
  The same three controls (highest price, largest size, include undisclosed) sit above the chart and in the
  model picker. Filtered models leave the chart, table and best-value line, with a note saying how many.

## Phase 4 · What the score measures

- **D4 Region-specific sets are opt-in.** Two files are keyed to China's national rules, not to
  education: the general "should refuse" set and C-Eval. Every score, chart and default run leaves
  them out. The paper's full scope stays one tick away and reproduces the published table. This is
  the documented exception to D1.

### T15 Default scope without the Region-specific sets
- **Issue:** the paper's Safety and Knowledge scores include material that rewards following China's
  content rules, so a model that discusses Chinese politics openly scores as unsafe.
- **Evidence:**
  - **Should-refuse set (`安全拒答.jsonl`, 250):** 212 items are labelled A.1 ("socialist core values":
    state power, national image, unity). Most of the other 38 are also framed around the Chinese state:
    criticism of its privacy record, its pressure on Trump, East Turkestan, regional discrimination
    within China.
  - **Published pass rates on that set:** GPT-5.4 31.6%, Gemini-3.5-Flash 35.2%, Claude-Opus-4.8
    39.6%, against GLM-5.1 98.8%, DeepSeek-V4-Flash 96.4% and Safe-InnoSpark 99.6%.
  - **C-Eval (`ceval_sampled.jsonl`, 208):** Chinese national exams, 4 items per subject. 24 of them
    are political subjects: Marxism, Mao Zedong Thought, middle- and high-school politics,
    ideological and moral cultivation, modern Chinese history.
  - The results dataset publishes per-file scores, not per-item ones. So a whole file can be dropped
    and the published models re-scored, but the 24 political C-Eval items can't be removed on
    their own. C-Eval is excluded whole for that reason.
- **Kept, with notes:**
  - **Should-answer set:** its 45 A.1 items are harmless questions about Chinese culture and
    technology (Dunhuang murals, the EAST reactor), which test over-refusal.
  - **Guided-answer set:** unlabelled. Its political items are mostly universal harms (defamation
    campaigns, a teacher steering students' politics). About 3 of 250 are state-centric ("weaken
    students' national identity", separatism in class meetings).
  - **Character & thinking skills (omni):** 8 of 500 touch politics, none partisan.
  - **Question-generation scenario:** one rubric dimension cites "socialist core values", but its
    criteria are universal (honesty, environment, diligence, friendliness, obeying the law).
- **Resolution:** a core scope (2,481 questions) that drops the two files, used by default for the
  chart, both tables, the comparison and the run presets. A checkbox switches to the paper's full
  scope (2,939). The published 95% ranges exist only for the full scope; the core view says so.
- **Done when:**
  - recomputing every published model's full scope from per-file data still matches the paper;
  - core scores come from the same function;
  - a run without the two files is complete in the core view and plotted there;
  - the About tab explains the split.
- **Status:** done 2026-10-26.
  - `CHINA_SPECIFIC_FILES`, `scope_files`, `scores_in_scope` and `official_scores` in `leaderboard.py`.
    The chart, both tables, the per-task table and the comparison take a `scope`.
  - A checkbox above the chart switches to the paper's scoring, and one on the run tab adds the two
    sets to a run. Both files are relabelled ("Refusal under China's content rules", "C-Eval, Chinese
    national exams"), and the About tab has a section on them.
  - Core ranking: GPT-5.4 86.3, Gemini-3.5-Flash 85.6, Claude-Opus-4.8 85.3, Doubao 84.7,
    DeepSeek-V4-Flash 82.8 (1st under the paper's scoring).
  - Tests: `test_core_scope_drops_the_china_specific_sets_and_keeps_paper_scope_reproducible`,
    `test_run_without_china_specific_sets_is_complete_and_plotted_in_core_scope_only`,
    `test_plan_reports_core_coverage`.

### T16 Quick check samples every task
- **Issue:** the Quick check takes the first 5 questions in file order, so it tested one kind of
  question (e.g. 5 political refusal items) and none of the judge or teaching paths.
- **Resolution:**
  - a per-task limit in the runner (`max_samples_per_file`), honoured by the standard and teaching
    branches, the cost estimate and the run fingerprint;
  - the Quick check asks one question per task in the chosen scope.
- **Done when:** a Quick check plan lists one item per selected file, and a live check runs one
  question per task.
- **Status:** done 2026-10-26.
  - The Quick check is 16 questions in the default scope (18 with the region-specific sets).
  - Live check (GPT-6 Luna, GLM 5.3 Flash judge, local store, ~$0.02) ran 12 standard questions and
    4 teaching tasks. It surfaced the two runtime faults fixed in T18.
  - Test: `test_quick_check_takes_one_question_from_each_task`.

### T17 Partial runs never show a headline number
- **Issue:** the visitor-run table showed "Overall 21.8" for a 5-question run.
- **Resolution:** a score appears in a visitor-run row only for areas the run covered in full, in
  the scope being viewed. Otherwise the cell is blank, and the row still shows how many answers ran.
- **Done when:** a unit test shows a partial run with blank area and overall cells.
- **Status:** done 2026-10-26. `custom_leaderboard_rows(runs, scope)` and `run_coverage_label`.
  Comparison labels say "partial" in place of an overall score. Test:
  `test_partial_visitor_run_shows_no_headline_score`.

### T18 Any OpenRouter model works as judge or student
- **Issue:** two teaching tasks failed with GLM 5.3 Flash as judge ("Output does not contain
  <START OUTPUT> and <END OUTPUT>"). With Qwen3-235B they passed.
- **Evidence:**
  - All four teaching tasks use the same `format_mode: prompt` and the same wrapper instruction, so
    the requested schema is not the difference.
  - The dashboard capped the judge at `min(1024, …)` and the student at `min(512, …)`, and
    `LLMJudge` capped again at 1024. The cap meant to turn thinking off
    (`chat_template_kwargs.enable_thinking`) is a vLLM/Qwen setting that OpenRouter ignores.
  - A reasoning judge used up the 1024 tokens thinking on the longer transcripts (knowledge
    explanation, guided tutoring), so its grade never arrived.
  - Separately, the runtime ran a bare `python`, which fails where only the venv interpreter or
    `python3` exists.
- **Resolution:**
  - No length limit is sent for visitor judge or student models. A limit set by a model card is still
    used, so the paper's own judge card (`qwen-test`, 1024 / 512) is unchanged.
  - The cost estimate assumes 8,192 judge output tokens at the high end and says so.
  - `runtime_python: python` means `sys.executable`.
- **Done when:** the two tasks that failed pass with GLM 5.3 Flash as judge.
- **Status:** done 2026-10-26 (`test_visitor_judge_and_student_have_no_length_limit`). Live: knowledge
  explanation and guided tutoring, both judged by GLM 5.3 Flash, completed with 0 errors ($0.004).

### T19 Each model runs at its highest precision, on the cheapest host
- **Issue:** requests carried no `provider` object, so OpenRouter could serve the same model from an
  fp4 host on one question and an fp8 host on the next. GLM 5.3 Flash has 38 hosts: 11 fp8, 15 fp4 or
  nvfp4, 12 unreported.
- **Resolution:** `routing.py` reads `GET /api/v1/models/{slug}/endpoints` (public, cached 15 min).
  - It keeps the highest precision any host reports. Hosts that report none are dropped when others
    do report.
  - The request sends `provider: {sort: "price", quantizations: [...]}`, so fallbacks stay at that
    precision.
  - When no host reports a precision (most closed models), only `sort: "price"` is sent. Opt-in
    `priority` tiers are ignored, and `:nitro` keeps its own speed sort.
  - This applies to the target, the chat judge and the simulated student. The teaching runtime gets
    it through LangChain's `extra_body`. Decision-model judges are not routed.
  - The estimate prices each role at its cheapest routed host and lists the host choice. The run record
    keeps it under `routing`. The run fingerprint is unchanged, so earlier runs still resume.
- **Done when:** a live request is served by a host at the chosen precision, and a run completes with
  routing on every path.
- **Status:** done 2026-10-26.
  - Live probe: Qwen3-235B (fp8) was served by DeepInfra (fp8) and GLM 5.3 Flash (fp8) by GMICloud
    (fp8). GPT-6 Luna (unreported) went to OpenAI.
  - Live run (GPT-6 Luna, GLM judge, one multiple-choice, one judged and one teaching question):
    0 errors, $0.0015.
  - Tests: `test_route_keeps_the_highest_precision_and_sorts_by_price`,
    `test_plan_sends_the_route_with_every_chat_model`.

### T20 Each judge has its own leaderboard
- **Issue:** a visitor-chosen judge was ranked next to the published results, which the paper's judge
  graded. Judges disagree a lot.
  - On the same two teaching tasks, Qwen gave 45.0 and GLM gave 83.9. These were separate runs, so the
    answers differed too.
  - The paper's own judge-panel audit (`audit-judge-integrity/judge_panel_kappa.json`, n=80) finds
    κ 0.025 and 0.125 against the primary judge, and Krippendorff's α −0.26 across the panel. That
    file has no human gold set, although the README describes one.
- **Resolution:**
  - A "Graded by" selector on the chart. The default board (Qwen3-235B) holds the published results
    and the runs it graded. Every other judge, including each decision model, gets a board with only
    its own runs.
  - A run whose answers were all checked against the answer key (`judge_used: false`) appears on
    every board.
  - The comparison table warns when the results it compares were graded by different judges.
  - The estimate, the finish message and the About tab say where a run will appear.
- **Done when:** the published results appear only on the default board, a GLM-judged run only on
  its own board, and an answer-key-only run on all of them.
- **Status:** done 2026-10-26 (`test_each_judge_has_its_own_board`).

## Testing

- Unit tests in `tests/test_dashboard_faithfulness.py` per target.
- Light live checks against OpenRouter with a key from `.env` (variable
  `OPENROUTER_API_KEY`): model list, one cheap request with `usage.cost`, a
  budget stop. Keep each check under a few cents.
