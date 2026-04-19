# V4EMA Regime Classifier — Framework Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `REGIME_CLASSIFIER=v4ema` env-var hack in `lib/generation/generator.py` into a proper per-signal YAML field so rolling backtests can compare V3 and V4EMA classifiers cleanly, with validated backward compatibility.

**Architecture:** Add a `regime_classifier` field (default `"v3"`) to `SignalConfig`, propagate it through the YAML loader (`expand_signal_template`), and let the generator dispatch on the per-signal value instead of a process-wide env var. Keep existing YAMLs unchanged — they inherit `"v3"` automatically.

**Tech Stack:** Python 3.13, dataclasses, pytest (new), PyYAML (already used). No external deps added.

---

## Current state (before implementation)

- `lib/generation/templates/base.py` — contains both `REGIME_DETECTION_BLOCK` (v3 ADX) and `REGIME_DETECTION_BLOCK_V4EMA` (already added).
- `lib/generation/generator.py` — has `_select_regime_block()` helper that reads `REGIME_CLASSIFIER` env var (**the hack to replace**).
- `lib/signals/base.py:11-57` — `SignalConfig` dataclass, no `regime_classifier` field.
- `lib/signals/registry.py:91-226` — `expand_signal_template` builds `SignalConfig` from YAML dict but ignores any `regime_classifier` key.
- No `tests/` directory in the project. We will create a minimal one for this feature.

## File Structure

**Modify:**
- `lib/signals/base.py` — add field, update `from_dict`/`to_dict`
- `lib/signals/registry.py` — propagate field in `expand_signal_template` (both branches)
- `lib/generation/generator.py` — dispatch on `signal.regime_classifier`, delete env-var helper

**Create:**
- `tests/__init__.py`
- `tests/test_regime_classifier.py` — unit + integration tests
- `configs/signals_v4ema_example.yaml` — YAML demonstrating the new field
- `docs/regime_classifier.md` — feature doc (short)

**Unchanged (backward-compat target):**
- `configs/signals.yaml`, `configs/signals_btc_run*.yaml`, `configs/signals_hype_*.yaml` — all must still load correctly without modification.

---

## Task 1: Add `regime_classifier` field to `SignalConfig`

**Files:**
- Modify: `lib/signals/base.py:11-57` (dataclass), `:79-99` (`from_dict`), `:101-113` (`to_dict`)
- Test: `tests/test_regime_classifier.py`

- [ ] **Step 1: Create the tests/ package**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Write failing tests for SignalConfig regime_classifier**

Create `tests/test_regime_classifier.py`:

```python
"""Tests for regime_classifier integration."""
from lib.signals.base import SignalConfig


def test_signalconfig_defaults_to_v3():
    sc = SignalConfig(name="s1", signal_type="funding", direction="both")
    assert sc.regime_classifier == "v3"


def test_signalconfig_accepts_v4ema():
    sc = SignalConfig(
        name="s1",
        signal_type="funding",
        direction="both",
        regime_classifier="v4ema",
    )
    assert sc.regime_classifier == "v4ema"


def test_signalconfig_from_dict_reads_regime_classifier():
    data = {
        "name": "s1",
        "signal_type": "funding",
        "direction": "both",
        "regime_classifier": "v4ema",
    }
    sc = SignalConfig.from_dict(data)
    assert sc.regime_classifier == "v4ema"


def test_signalconfig_from_dict_defaults_when_missing():
    data = {"name": "s1", "signal_type": "funding", "direction": "both"}
    sc = SignalConfig.from_dict(data)
    assert sc.regime_classifier == "v3"


def test_signalconfig_to_dict_includes_regime_classifier():
    sc = SignalConfig(
        name="s1",
        signal_type="funding",
        direction="both",
        regime_classifier="v4ema",
    )
    d = sc.to_dict()
    assert d["regime_classifier"] == "v4ema"
```

- [ ] **Step 3: Run tests to confirm they fail**

Run from project root:
```bash
cd /home/nyzam/Documents/Valentin/crypto_comparative_analysis
python -m pytest tests/test_regime_classifier.py -v
```
Expected: 5 failures, all because `regime_classifier` isn't a field yet (AttributeError or TypeError on the constructor).

- [ ] **Step 4: Add the field to the dataclass**

In `lib/signals/base.py`, change lines 66-70 from:

```python
    # Allowed regimes (None = auto-detect based on name)
    allowed_regimes: Optional[List[str]] = None

    # Exit configuration name
    exit_config: str = "none"
```

to:

```python
    # Allowed regimes (None = auto-detect based on name)
    allowed_regimes: Optional[List[str]] = None

    # Exit configuration name
    exit_config: str = "none"

    # Regime detection implementation: "v3" (ADX/ATR — default) or "v4ema" (EMA alignment)
    regime_classifier: str = "v3"
```

- [ ] **Step 5: Update `from_dict` to read the field**

In `lib/signals/base.py:79-99`, replace the `from_dict` method with:

```python
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalConfig":
        """Create SignalConfig from dictionary."""
        params = data.get("params", {})
        default_params = cls.__dataclass_fields__["params"].default_factory()
        merged_params = {**default_params, **params}

        return cls(
            name=data["name"],
            signal_type=data["signal_type"],
            direction=data.get("direction", "both"),
            params=merged_params,
            roi=data.get("roi", {"0": 0.02}),
            stoploss=data.get("stoploss", -0.03),
            timeframe_override=data.get("timeframe_override"),
            allowed_regimes=data.get("allowed_regimes"),
            exit_config=data.get("exit_config", "none"),
            regime_classifier=data.get("regime_classifier", "v3"),
        )
```

- [ ] **Step 6: Update `to_dict` to include the field**

In `lib/signals/base.py:101-113`, replace `to_dict` with:

```python
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "params": self.params,
            "roi": self.roi,
            "stoploss": self.stoploss,
            "timeframe_override": self.timeframe_override,
            "allowed_regimes": self.allowed_regimes,
            "exit_config": self.exit_config,
            "regime_classifier": self.regime_classifier,
        }
```

- [ ] **Step 7: Update `with_exit` helper to preserve the field**

In `lib/signals/base.py:119-131`, replace `with_exit` with:

```python
    def with_exit(self, exit_name: str) -> "SignalConfig":
        """Create a copy with different exit config."""
        return SignalConfig(
            name=f"{self.name}_x{exit_name[:8]}",
            signal_type=self.signal_type,
            direction=self.direction,
            params=self.params.copy(),
            roi={"0": 0.2},
            stoploss=self.stoploss,
            timeframe_override=self.timeframe_override,
            allowed_regimes=self.allowed_regimes,
            exit_config=exit_name,
            regime_classifier=self.regime_classifier,
        )
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
python -m pytest tests/test_regime_classifier.py -v
```
Expected: all 5 pass.

- [ ] **Step 9: Commit**

```bash
git add lib/signals/base.py tests/__init__.py tests/test_regime_classifier.py
git commit -m "feat(signals): add regime_classifier field to SignalConfig

Default 'v3' preserves current behavior. Enables per-signal classifier
selection as part of YAML-driven config.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Propagate `regime_classifier` through the YAML loader

**Files:**
- Modify: `lib/signals/registry.py:131-226` (`expand_signal_template`)
- Test: `tests/test_regime_classifier.py`

- [ ] **Step 1: Add failing integration test for YAML → SignalConfig**

Append to `tests/test_regime_classifier.py`:

```python
from lib.signals.registry import expand_signal_template


def test_expand_template_reads_regime_classifier():
    template = {
        "name": "f_z{zscore}_v4ema",
        "direction": "both",
        "params": {"zscore": [1.0, 1.5]},
        "roi": {"0": 0.02},
        "stoploss": -0.03,
        "exit_config": "none",
        "regime_classifier": "v4ema",
    }
    signals = expand_signal_template(template, "funding", available_exits=["none"])
    assert len(signals) == 2
    assert all(s.regime_classifier == "v4ema" for s in signals)


def test_expand_template_defaults_when_classifier_absent():
    template = {
        "name": "f_z{zscore}_legacy",
        "direction": "both",
        "params": {"zscore": [1.0]},
        "roi": {"0": 0.02},
        "stoploss": -0.03,
        "exit_config": "none",
    }
    signals = expand_signal_template(template, "funding", available_exits=["none"])
    assert len(signals) == 1
    assert signals[0].regime_classifier == "v3"
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_regime_classifier.py::test_expand_template_reads_regime_classifier tests/test_regime_classifier.py::test_expand_template_defaults_when_classifier_absent -v
```
Expected: both fail (regime_classifier will be "v3" in both cases because registry doesn't read it yet).

- [ ] **Step 3: Extract `regime_classifier` from template in `expand_signal_template`**

In `lib/signals/registry.py`, find the block at line 131-141 (inside `expand_signal_template`) that reads:

```python
    # Extract template fields
    name_template = template.get("name", "unnamed")
    direction = template.get("direction", "both")
    params = template.get("params", {})
    roi = template.get("roi", {"0": 0.02})
    stoploss = template.get("stoploss", -0.03)
    timeframe_override = template.get("timeframe_override")
    allowed_regimes = template.get("allowed_regimes")
    exit_config = template.get("exit_config", "none")
```

Replace with:

```python
    # Extract template fields
    name_template = template.get("name", "unnamed")
    direction = template.get("direction", "both")
    params = template.get("params", {})
    roi = template.get("roi", {"0": 0.02})
    stoploss = template.get("stoploss", -0.03)
    timeframe_override = template.get("timeframe_override")
    allowed_regimes = template.get("allowed_regimes")
    exit_config = template.get("exit_config", "none")
    regime_classifier = template.get("regime_classifier", "v3")
```

- [ ] **Step 4: Pass it to SignalConfig in the cartesian-product branch**

In `lib/signals/registry.py:197-207`, the `SignalConfig(...)` call. Replace with:

```python
            signal = SignalConfig(
                name=name,
                signal_type=signal_type,
                direction=direction,
                params=combo_params,
                roi=roi if isinstance(roi, dict) else {"0": roi},
                stoploss=stoploss,
                timeframe_override=timeframe_override,
                allowed_regimes=allowed_regimes,
                exit_config=exit_config,
                regime_classifier=regime_classifier,
            )
```

- [ ] **Step 5: Pass it in the single-config branch**

In `lib/signals/registry.py:213-223` (the `else` branch after `if expansion_items:`). Replace with:

```python
        signal = SignalConfig(
            name=name,
            signal_type=signal_type,
            direction=direction,
            params=fixed_params,
            roi=roi_list[0] if isinstance(roi_list[0], dict) else {"0": roi_list[0]},
            stoploss=stoploss_list[0],
            timeframe_override=timeframe_override,
            allowed_regimes=allowed_regimes,
            exit_config=exit_list[0],
            regime_classifier=regime_classifier,
        )
```

- [ ] **Step 6: Run tests, all should pass**

```bash
python -m pytest tests/test_regime_classifier.py -v
```
Expected: all 7 pass.

- [ ] **Step 7: Commit**

```bash
git add lib/signals/registry.py tests/test_regime_classifier.py
git commit -m "feat(signals): propagate regime_classifier from YAML template

expand_signal_template reads the optional 'regime_classifier' key and
threads it into every generated SignalConfig (both cartesian-product
and single-config branches).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Generator dispatches on `signal.regime_classifier` (remove env-var hack)

**Files:**
- Modify: `lib/generation/generator.py:16-30` (imports + `_select_regime_block`), `:135-140` (call site)
- Test: `tests/test_regime_classifier.py`

- [ ] **Step 1: Add failing test for generator dispatch**

Append to `tests/test_regime_classifier.py`:

```python
from lib.generation.templates.base import (
    REGIME_DETECTION_BLOCK,
    REGIME_DETECTION_BLOCK_V4EMA,
)
from lib.generation.generator import _regime_block_for


def test_regime_block_for_v3():
    block = _regime_block_for("v3")
    assert block is REGIME_DETECTION_BLOCK


def test_regime_block_for_v4ema():
    block = _regime_block_for("v4ema")
    assert block is REGIME_DETECTION_BLOCK_V4EMA


def test_regime_block_for_unknown_falls_back_to_v3():
    block = _regime_block_for("nonexistent")
    assert block is REGIME_DETECTION_BLOCK
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_regime_classifier.py -v
```
Expected: 3 new failures on `_regime_block_for` (doesn't exist yet — the current function is `_select_regime_block`).

- [ ] **Step 3: Replace `_select_regime_block` with `_regime_block_for`**

In `lib/generation/generator.py`, find the block at lines 16-30 currently reading:

```python
import os as _os
from .templates.base import (
    INDICATORS_BLOCK,
    REGIME_DETECTION_BLOCK,
    REGIME_DETECTION_BLOCK_V4EMA,
)


def _select_regime_block():
    """Pick regime block based on REGIME_CLASSIFIER env var.
    - v4ema  -> EMA-alignment classifier
    - default -> original ADX/ATR classifier
    """
    if _os.environ.get("REGIME_CLASSIFIER", "").lower() == "v4ema":
        return REGIME_DETECTION_BLOCK_V4EMA
    return REGIME_DETECTION_BLOCK
```

Replace with:

```python
from .templates.base import (
    INDICATORS_BLOCK,
    REGIME_DETECTION_BLOCK,
    REGIME_DETECTION_BLOCK_V4EMA,
)


_REGIME_BLOCKS = {
    "v3": REGIME_DETECTION_BLOCK,
    "v4ema": REGIME_DETECTION_BLOCK_V4EMA,
}


def _regime_block_for(classifier: str) -> str:
    """Return the regime-detection code block matching a classifier name.

    Unknown names fall back to 'v3' (ADX/ATR) to keep the generator
    strictly backward-compatible.
    """
    return _REGIME_BLOCKS.get(classifier, REGIME_DETECTION_BLOCK)
```

- [ ] **Step 4: Update the call site inside `_generate_funding_strategy`**

In `lib/generation/generator.py`, the call site currently reads:

```python
            regime_detection_block=_select_regime_block(),
```

Replace with:

```python
            regime_detection_block=_regime_block_for(signal.regime_classifier),
```

- [ ] **Step 5: Update call site in `_generate_standard_strategy` similarly**

Same replacement as Step 4 — every `_select_regime_block()` invocation becomes `_regime_block_for(signal.regime_classifier)`.

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_regime_classifier.py -v
```
Expected: all 10 pass.

- [ ] **Step 7: Add a "strategy code differs" integration test**

Append to `tests/test_regime_classifier.py`:

```python
from pathlib import Path
from lib.config.base import Config
from lib.generation.generator import StrategyGenerator
from lib.signals.base import SignalConfig


def test_generated_strategies_differ_by_classifier(tmp_path):
    cfg = Config()
    cfg.strategies_dir = str(tmp_path)

    signal_v3 = SignalConfig(
        name="ftest_v3", signal_type="funding", direction="both",
        params={"zscore": 1.0, "lookback": 168},
        regime_classifier="v3",
    )
    signal_v4 = SignalConfig(
        name="ftest_v4ema", signal_type="funding", direction="both",
        params={"zscore": 1.0, "lookback": 168},
        regime_classifier="v4ema",
    )

    gen = StrategyGenerator(cfg)
    _, path_v3 = gen.generate(signal_v3, "30m")
    _, path_v4 = gen.generate(signal_v4, "30m")

    code_v3 = Path(path_v3).read_text()
    code_v4 = Path(path_v4).read_text()

    # V3 uses ADX thresholds; V4EMA uses EMA alignment.
    assert "REGIME_ADX_THRESHOLD" in code_v3
    assert "EMA-alignment regime classifier" in code_v4
    assert "REGIME_ADX_THRESHOLD" not in code_v4
```

- [ ] **Step 8: Run full test suite, all pass**

```bash
python -m pytest tests/test_regime_classifier.py -v
```
Expected: all 11 pass.

- [ ] **Step 9: Commit**

```bash
git add lib/generation/generator.py tests/test_regime_classifier.py
git commit -m "feat(gen): dispatch regime block on signal.regime_classifier

Replaces the REGIME_CLASSIFIER env-var hack with a proper per-signal
field. Unknown names fall back to v3 to preserve backward compat.
The generator now emits different regime code per signal in a single
run, which lets rolling backtests compare classifiers directly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Example YAML using V4EMA

**Files:**
- Create: `configs/signals_v4ema_example.yaml`

- [ ] **Step 1: Write the example YAML**

Create `configs/signals_v4ema_example.yaml`:

```yaml
# =============================================================================
# Example: opt-in V4EMA regime classifier for funding-contrarian signals.
#
# Any signal block may carry `regime_classifier: v4ema` at its top level
# to switch that block's generated strategies to the EMA-alignment
# classifier (close > EMA50 > EMA200 → bull, mirror for bear).
#
# When the key is omitted, the default "v3" (ADX/ATR) classifier is used
# exactly as before. Existing YAMLs need no changes.
# =============================================================================

funding_signals:
  - name: "f_z{zscore}_lb{lookback}_r35_v4ema"
    signal_type: funding
    direction: both
    regime_classifier: v4ema
    params:
      zscore: [1.0, 1.5, 2.0]
      lookback: [336, 504, 720]
      use_rsi: true
      rsi_min: 35
      rsi_max: 65
    allowed_regimes: [bull, bear, range, volatile]
    roi: { "0": 0.02 }
    stoploss: -0.03
    exit_config: none
```

- [ ] **Step 2: Smoke-test the loader**

```bash
python -c "
from lib.config.base import Config
from lib.signals.registry import load_signals_from_yaml
cfg = Config()
signals = load_signals_from_yaml(cfg, 'configs/signals_v4ema_example.yaml')
print(f'Loaded {len(signals)} signals')
for s in signals[:3]:
    print(f'  {s.name}: classifier={s.regime_classifier}')
assert all(s.regime_classifier == 'v4ema' for s in signals), 'All signals should be v4ema'
print('OK')
"
```

Expected output:
```
Loaded 9 signals
  f_z1_lb336_r35_v4ema: classifier=v4ema
  f_z1_5_lb336_r35_v4ema: classifier=v4ema
  f_z2_lb336_r35_v4ema: classifier=v4ema
OK
```

- [ ] **Step 3: Commit**

```bash
git add configs/signals_v4ema_example.yaml
git commit -m "docs(configs): add v4ema example signal config

Minimal YAML showing how to opt a signal block into the V4EMA
classifier via the new regime_classifier field.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Backward-compatibility smoke test

**Files:**
- Test: `tests/test_regime_classifier.py`

- [ ] **Step 1: Add end-to-end test against existing YAML**

Append to `tests/test_regime_classifier.py`:

```python
def test_existing_signals_yaml_still_loads_as_v3():
    """Existing YAMLs must produce v3-classified signals with zero changes."""
    cfg = Config()
    cfg.signals = "configs/signals.yaml"

    from lib.signals.registry import load_signals_from_yaml
    signals = load_signals_from_yaml(cfg, cfg.signals)
    assert len(signals) > 0, "Should load at least one signal"
    assert all(
        s.regime_classifier == "v3" for s in signals
    ), f"Expected all v3, got: {set(s.regime_classifier for s in signals)}"
```

- [ ] **Step 2: Run it**

```bash
python -m pytest tests/test_regime_classifier.py::test_existing_signals_yaml_still_loads_as_v3 -v
```
Expected: pass. Confirms existing configs are untouched.

- [ ] **Step 3: Commit**

```bash
git add tests/test_regime_classifier.py
git commit -m "test: verify existing YAMLs still load as v3 after refactor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Short feature doc

**Files:**
- Create: `docs/regime_classifier.md`

- [ ] **Step 1: Write the doc**

Create `docs/regime_classifier.md`:

```markdown
# Regime classifier selection

Every funding / technical signal in a YAML config can opt into a
specific regime-detection implementation via the `regime_classifier`
top-level key:

```yaml
funding_signals:
  - name: "f_z1_lb336_v4ema"
    signal_type: funding
    regime_classifier: v4ema     # opt-in; default is "v3"
    # ...rest of the signal definition
```

## Supported values

| Value     | Detection logic                                                |
|-----------|----------------------------------------------------------------|
| `v3`      | ADX + ATR (default — ADX ≥ 30 with DI alignment = bull/bear)   |
| `v4ema`   | EMA alignment (close > EMA50 > EMA200 = bull, mirror for bear) |

Unknown values fall back to `v3` at generation time.

## Why it matters

Regime labels gate the strategy's entry logic (`allowed_regimes*`
filters). A classifier that mislabels a trending market as "range"
can cause contrarian longs to fire during trends and get stopped out.
V4EMA was introduced to address an observed March 2026 HYPE case where
ADX underlabeled a +19% month as "range".

Running rolling backtests with both classifiers side by side is now
possible: put half the signals in one YAML block (v3), half in another
(v4ema), and let the framework generate and compare.
```

- [ ] **Step 2: Commit**

```bash
git add docs/regime_classifier.md
git commit -m "docs: regime_classifier field reference

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Full integration validation — rolling backtest

**Files:**
- None created; runs existing signal configs end-to-end.

- [ ] **Step 1: Run the framework with the example v4ema YAML**

```bash
docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/home/ftuser \
  -e PYTHONUNBUFFERED=1 \
  -e PATH=/home/ftuser/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -v /home/nyzam/Documents/Valentin/crypto_comparative_analysis/configs:/freqtrade/configs \
  -v /home/nyzam/Documents/Valentin/crypto_comparative_analysis/lib:/freqtrade/lib \
  -v /home/nyzam/Documents/Valentin/crypto_comparative_analysis/scripts:/freqtrade/scripts \
  -v /home/nyzam/Documents/Valentin/crypto_comparative_analysis/user_data:/freqtrade/user_data \
  -v /home/nyzam/Documents/Valentin/hyperliquid_data/user_data/data/hyperliquid:/freqtrade/data/hyperliquid:ro \
  -w /freqtrade \
  --entrypoint python \
  freqtradeorg/freqtrade:stable_freqai \
  -u scripts/comparative_analysis_v3.py \
  --pairs HYPE/USDC:USDC \
  --timeframes 30m \
  --timerange 20250108-20260408 \
  --rolling --window 3 --step 3 --min-windows 4 \
  --enable-filter \
  --signals configs/signals_v4ema_example.yaml > /tmp/bt_v4ema_yaml_verify.log 2>&1
```

Expected: process exits 0; `/tmp/bt_v4ema_yaml_verify.log` contains a robustness report (non-empty top signals table).

- [ ] **Step 2: Verify generated strategies use V4EMA**

```bash
grep -l "EMA-alignment regime classifier" \
  user_data/strategies/generated_v3/S_f_z*_v4ema*.py
```

Expected: at least one filename listed.

- [ ] **Step 3: Confirm other (non-v4ema) strategies still use V3**

```bash
grep -l "REGIME_ADX_THRESHOLD" \
  user_data/strategies/generated_v3/S_f_z*_base*.py | head -3
```

Expected: at least one filename listed (confirms legacy signals still emit ADX logic).

- [ ] **Step 4: Confirm no `REGIME_CLASSIFIER` env var left in code**

```bash
grep -r "REGIME_CLASSIFIER" lib/ scripts/
```

Expected: no output. The env-var hack is fully gone.

- [ ] **Step 5: Commit an integration-validated tag (optional)**

```bash
git tag -a v4ema-classifier-yaml -m "Feature landed: regime_classifier YAML field"
```

---

## Self-review notes

- **Spec coverage**: Every element of the stated goal (remove env-var hack, per-signal YAML field, backward-compat, integration-verified) maps to a task (1→2→3, 4, 5, 7).
- **Placeholders**: None — every code block shows actual content.
- **Type consistency**: `regime_classifier: str = "v3"` is used everywhere. Helper is named `_regime_block_for` consistently in both definition (Task 3 Step 3) and callers (Task 3 Steps 4-5) and tests (Task 3 Step 1).
- **Deletion**: The old `_select_regime_block` is removed as part of Task 3 Step 3 (replaced wholesale). No stray references remain (verified in Task 7 Step 4).
