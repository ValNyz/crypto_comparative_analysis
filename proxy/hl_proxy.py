import asyncio
import hashlib
import logging
import os
import time

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger("hl_proxy")

HL_BASE = "https://api.hyperliquid.xyz"
REFILL_SEC = float(os.environ.get("HL_PROXY_REFILL_SEC", "1.0"))
BUCKET_CAPACITY = int(os.environ.get("HL_PROXY_BUCKET_CAPACITY", "15"))
REQUEST_TIMEOUT = float(os.environ.get("HL_PROXY_TIMEOUT_SEC", "30"))
CACHE_TTL = float(os.environ.get("HL_PROXY_CACHE_TTL", "5.0"))
CACHE_MAX_ENTRIES = int(os.environ.get("HL_PROXY_CACHE_MAX_ENTRIES", "200"))
CACHE_PATHS = {"/info"}


class TokenBucket:
    def __init__(self, capacity: int, refill_seconds: float) -> None:
        self.capacity = capacity
        self.refill = refill_seconds
        self.tokens = float(capacity)
        self.last = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = asyncio.get_event_loop().time()
                if self.last == 0.0:
                    self.last = now
                self.tokens = min(
                    self.capacity,
                    self.tokens + (now - self.last) / self.refill,
                )
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) * self.refill
            await asyncio.sleep(wait)


bucket = TokenBucket(BUCKET_CAPACITY, REFILL_SEC)
# cache key (str) → (timestamp, status, headers_dict, body_bytes)
cache: dict = {}
cache_lock = asyncio.Lock()
cache_hits = 0
cache_misses = 0


def _cache_key(method: str, path_qs: str, body: bytes) -> str:
    h = hashlib.md5(body).hexdigest()
    return f"{method}:{path_qs}:{h}"


async def _cache_get(key: str):
    async with cache_lock:
        entry = cache.get(key)
        if entry is None:
            return None
        ts, status, headers, body = entry
        if time.time() - ts > CACHE_TTL:
            del cache[key]
            return None
        return status, headers, body


async def _cache_set(key: str, status: int, headers: dict, body: bytes) -> None:
    async with cache_lock:
        if len(cache) >= CACHE_MAX_ENTRIES:
            oldest = min(cache.keys(), key=lambda k: cache[k][0])
            del cache[oldest]
        cache[key] = (time.time(), status, headers, body)


async def health(_: web.Request) -> web.Response:
    total = cache_hits + cache_misses
    hit_rate = (cache_hits / total * 100) if total > 0 else 0
    return web.Response(
        text=(
            f"ok tokens={bucket.tokens:.2f}/{BUCKET_CAPACITY} "
            f"refill={REFILL_SEC}s cache={len(cache)}/{CACHE_MAX_ENTRIES} "
            f"hit_rate={hit_rate:.1f}% (hits={cache_hits} misses={cache_misses})\n"
        )
    )


async def handle(request: web.Request) -> web.Response:
    global cache_hits, cache_misses
    body = await request.read()
    cacheable = request.path in CACHE_PATHS
    cache_key = _cache_key(request.method, request.path_qs, body) if cacheable else ""

    if cacheable:
        cached = await _cache_get(cache_key)
        if cached is not None:
            cache_hits += 1
            status, headers, resp_body = cached
            return web.Response(body=resp_body, status=status, headers=headers)
        cache_misses += 1

    await bucket.acquire()
    url = HL_BASE + request.path_qs
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "connection", "content-length")
    }
    session: ClientSession = request.app["session"]
    try:
        async with session.request(
            method=request.method,
            url=url,
            headers=headers,
            data=body,
            allow_redirects=False,
        ) as resp:
            resp_body = await resp.read()
            resp_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in (
                    "content-encoding", "transfer-encoding",
                    "content-length", "connection",
                )
            }
            if cacheable and resp.status == 200:
                await _cache_set(cache_key, resp.status, resp_headers, resp_body)
            return web.Response(
                body=resp_body, status=resp.status, headers=resp_headers,
            )
    except Exception as e:
        log.warning(f"upstream error {request.method} {url}: {e}")
        return web.Response(status=502, text=f"proxy upstream error: {e}\n")


async def init_session(app: web.Application) -> None:
    timeout = ClientTimeout(total=REQUEST_TIMEOUT)
    connector = TCPConnector(limit=32, limit_per_host=32, keepalive_timeout=60)
    app["session"] = ClientSession(timeout=timeout, connector=connector)
    log.info("aiohttp ClientSession ready (keepalive)")


async def cleanup_session(app: web.Application) -> None:
    await app["session"].close()


def main() -> None:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_route("*", "/{path:.*}", handle)
    app.on_startup.append(init_session)
    app.on_cleanup.append(cleanup_session)
    log.info(
        f"starting hl_proxy on :8888 → {HL_BASE} | "
        f"refill={REFILL_SEC}s capacity={BUCKET_CAPACITY} "
        f"cache_ttl={CACHE_TTL}s cache_paths={CACHE_PATHS}"
    )
    web.run_app(app, host="0.0.0.0", port=8888, print=None)


if __name__ == "__main__":
    main()
