# Free Model Tester — 11 providers / 101 models

Live endpoint checker with Vanessa-style cooldown + probe-cache. No API keys required for Kilo/LLM7/Pollinations.

## Quick start
```bash
cd free-model-tester
pip install aiohttp rich
python free_model_tester.py               # live refresh + built-in dashboard on :8765
python free_model_tester.py --once        # single round, no server (use --no-serve to be explicit)
python free_model_tester.py --interval 60 --timeout 10
```

Starting normally prints:

```
Dashboard live at http://127.0.0.1:8765/ (Ctrl+C stops both)
```

Open that URL. The page is served by the tester itself, so it is genuinely live: it polls `/results.json`
every 5s and repaints whenever `generated_at` changes. Ctrl+C stops the tester and the server together.

## Flags
| Flag | Meaning |
|---|---|
| `--once` | one round, then exit |
| `--interval N` | seconds between rounds (default 120) |
| `--timeout N` | per-request timeout seconds (default 15) |
| `--concurrency N` | global concurrent requests (default 40) |
| `--providers a,b` | only test these provider keys |
| `--output PREFIX` | results file prefix (default `results`) |
| `--port N` | dashboard port (default 8765) |
| `--no-serve` | don't start the dashboard server |
| `--no-jitter` | disable per-request startup stagger |
| `--list-providers` | print provider keys and exit |
| `--deep` | require a real answer ("OK") instead of just a 200 handshake |
| `--stats` | print uptime table from history and exit |
| `--days N` | history window for `--stats` (default 7) |

## Server routes
| Route | Serves |
|---|---|
| `/` | `web/index.html` |
| `/results.json` | current results file for the run |
| `/api/status` | JSON health + last `generated_at`, count, providers |
| `/web/*` | static assets under `web/` |

Paths are confined to `web/`; traversal returns 403.

## Folder
```
free-model-tester/
  free_model_tester.py   # tester + dashboard server (11 providers / 101 models)
  .env                   # pooled keys (31 Mistral + others)
  results.json           # written every round
  results_summary.txt
  results_history.jsonl  # append-only: one line per model per round (uptime data)
  models_cache.json      # discovery cache (catalog size per provider, 1h TTL)
  data/                  # optional copy location
  web/index.html         # dashboard UI (Tailwind, polls /results.json, share buttons)
```

## Model discovery
Each round fetches `/models` from nvidia, groq, openrouter, mistral, googleai and cohere
(cached for an hour in `models_cache.json`). New ids are only **merged** into the test list
for providers in `DISCOVERY_MERGE_PROVIDERS` (default: `openrouter`, override with
`FREE_MODEL_TESTER_DISCOVERY_MERGE`). Others just record catalog size, because their `/models`
lists mix paid and non-chat models that 404 on `/chat/completions`.

## Uptime history
Every round appends one JSON line per model to `results_history.jsonl` (rotates to `.old` at
100 MB). Summarize it with:

```bash
python free_model_tester.py --stats --days 7
```

Columns: uptime %, provider, model, samples, OK count, deep-verified count, p50 latency.

## Website
Serve it with the tester itself (recommended):
```bash
python free_model_tester.py      # then http://127.0.0.1:8765/
```
Static hosting also works — open `web/index.html` directly or use `python -m http.server 8000`
and hit `http://localhost:8000/web/`; the page falls back to `../results.json` /
`../data/results.json`. Without a backend it shows a "backend offline" banner instead of
pretending the data is fresh.

## Providers (11)
- **Have key:** nvidia (13), groq (5), openrouter (11 free), mistral (5), codestral (1), googleai (6), cohere (9), xkiro (24 free)
- **No key:** kilo (10), llm7 (4), pollinations (13)
- Removed per health audits: deepinfra (402), aihorde (400), 17 dead 404/410 IDs