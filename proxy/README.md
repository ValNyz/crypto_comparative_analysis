# Hyperliquid /info caching proxy

Local dev-side proxy for `https://api.hyperliquid.xyz`. Caches identical
queries to `/info` (LRU + TTL) and serialises outbound calls (token-bucket
rate-limit) so a parallel grid of freqtrade workers stops hammering HL on
startup checks.

This is the same `hl_proxy.py` used in the production sister-repo
`../crypto/` — port 8888 with battle-tested aiohttp single-file server.

## Run

```bash
cd proxy
docker compose up -d --build

# verify
curl -s http://localhost:8888/health
# → ok tokens=15.00/15 refill=1.0s cache=0/200 hit_rate=0.0% (hits=0 misses=0)
```

## Wire freqtrade to the proxy

`user_data/config_analysis.json` already has the override baked in
(`exchange.ccxt_config.urls.api.public = "http://localhost:8888"`). CCXT
picks this up at exchange-instance creation time, so all `/info` POSTs
from freqtrade workers go through the proxy.

If you ever use a different config, add this snippet to its `exchange`
block:

```json
"ccxt_config": {
  "urls": {"api": {"public": "http://localhost:8888", "private": "http://localhost:8888"}}
},
"ccxt_async_config": {
  "urls": {"api": {"public": "http://localhost:8888", "private": "http://localhost:8888"}}
}
```

## Knobs (env vars in `docker-compose.yml`)

| Var | Default | Effect |
|---|---|---|
| `HL_PROXY_REFILL_SEC` | `1.0` | One token refilled every N seconds |
| `HL_PROXY_BUCKET_CAPACITY` | `15` | Max burst (tokens at full bucket) |
| `HL_PROXY_TIMEOUT_SEC` | `30` | Upstream HTTP timeout |
| `HL_PROXY_CACHE_TTL` | `3600` | Cache freshness for `/info` (1 hour) |
| `HL_PROXY_CACHE_MAX_ENTRIES` | `200` | LRU cache capacity |

Token bucket math: with `refill=1.0s capacity=15`, sustained rate is **1 req/s**
to HL with up to 15 stacked bursts. Tune `REFILL_SEC` lower (e.g. `0.5`) if HL
tolerates more aggressive sustained rates.

## What's cached

Only the `/info` path. All other paths (e.g. `/exchange` for trading) are
proxied with rate-limit but **not** cached — those are stateful or
user-specific.

The cache key is `(method, path_qs, md5(body))` so different POST bodies to
`/info` (e.g. different `type` queries — `meta`, `candleSnapshot`,
`fundingHistory`, …) get distinct cache entries.

## Observability

`GET /health` returns plain text:

```
ok tokens=14.83/15 refill=1.0s cache=47/200 hit_rate=86.9% (hits=312 misses=47)
```

So you can watch hit-rate climb during a grid run:

```bash
watch -n 2 'curl -s http://localhost:8888/health'
```

## Stop / clean up

```bash
docker compose down            # stop container, keep image
docker compose down --rmi all  # also remove the built image
```
