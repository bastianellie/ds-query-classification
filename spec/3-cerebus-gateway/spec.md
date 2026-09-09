# Spec 3: Cerebus/Portkey Gateway Support

> **Status: CLOSED** — Implemented and verified on 2026-09-09.
> Implementation summary: `spec/3-cerebus-gateway/implementation-summary.md`

## Overview

Adds an opt-in `--cerebus` flag to both entry points (`classify.py`/`cli.py` and
`experiment.py`) that routes every LLM call through Cerebus — an internal LLM gateway
built on [Portkey](https://portkey.ai/) — instead of a direct provider. Credentials
resolve from an env var first, falling back to AWS Secrets Manager. This spec is written
**retroactively**: the feature was already implemented and tested in this session before
being formalized here, so every requirement below describes verified, shipped behavior in
`src/query_classification/classifier.py`, `cli.py`, and `experiment.py`, plus
`tests/test_cerebus.py` (32 tests, all passing) — not a greenfield design. A first draft
of this spec was critiqued by Codex (`spec/3-cerebus-gateway/critique-v-1-codex.md`),
which found several places where the spec's prose overstated what the code did, plus three
real gaps (a security issue in the `--api-base` override, unsanitized error text, and a
test-import-boundary conflict with `INV-7`) — all fixed in the code before this version of
the spec was written, so the text below matches the corrected implementation.

## Goals

- Let a run reach Azure-backed or direct-provider (OpenAI, Gemini, ...) models through
  one internal gateway, without hand-rolling gateway auth headers per invocation.
- Match this repo's established `.env`-via-`python-dotenv` credential convention (already
  used for `AZURE_OPENAI_API_KEY` etc.) rather than introducing a new configuration
  mechanism.
- Resolve the gateway service key with minimal manual setup where possible (an explicit
  env var, or automatic AWS Secrets Manager lookup for developers with AWS SSO access),
  while failing clearly and actionably when neither source yields a key.
- Keep this fully opt-in: a run that never passes `--cerebus` produces the same
  `litellm.completion(...)` kwargs as before this feature existed — no new required
  dependency, no changed default behavior.

---

## Feature 1: Gateway Routing

**Who & why:** Whoever runs `classify.py` or `experiment.py` today points `--model` at a
LiteLLM model id and relies on LiteLLM's own per-provider env-var credential resolution
(`AZURE_OPENAI_API_KEY`, etc.) or a plain `--api-base` override. Some environments instead
require *all* LLM traffic to route through Cerebus, an internal gateway that fronts
multiple providers behind one OpenAI-compatible endpoint, authenticated by custom HTTP
headers rather than a provider's native scheme. Without this feature, using such an
environment means each `Classifier` construction site would need hand-written gateway
wiring duplicated across `cli.py` and `experiment.py`.

### Functional Requirements

#### FR-1.1: `--cerebus` is an opt-in boolean flag, available everywhere a `Classifier` is constructed
`classify.py`/`cli.py` gets one `--cerebus` flag (this entry point has no subcommands).
`experiment.py` gets the same flag in its **common** argument group, so it's available on
`induce`, `classify`, and `run` alike — induction calls need gateway routing exactly as
much as classification calls do. Argparse defaults it to `False`. When it's not passed,
the `Classifier` instances constructed are given `api_key=None, extra_headers=None` (the
same as before this feature existed), so the resulting `litellm.completion(...)` kwargs
are unchanged — omitting `--cerebus` is a no-op, not merely "close to" one.
**Verify:** `build_parser()` in both `cli.py` and `experiment.py` accepts `--cerebus` and
defaults it to `False`; `experiment.py`'s parser accepts it on all three subcommands.

#### FR-1.2: `CEREBUS_MODE` selects Azure-config-mode or direct-slug-mode gateway addressing
Cerebus/Portkey supports two addressing styles, selected by the exact (case-sensitive,
untrimmed) string `CEREBUS_MODE=azure` or `CEREBUS_MODE=direct` — any other value,
including unset, whitespace-only, or differently-cased, is a configuration error:
- **`azure`**: an Azure-backed model, addressed by a Portkey **Config ID** sent as the
  `x-portkey-config` header, reading the gateway URL from `CEREBUS_GATEWAY_AZURE_URL`.
- **`direct`**: a directly-hosted model (OpenAI, Gemini, ...), addressed by whatever value
  `--model` (or the relevant per-role flag) already holds, sent with an
  `x-portkey-provider: openai` header, reading the gateway URL from
  `CEREBUS_GATEWAY_DIRECT_URL`.

Mode-specific configuration (`CEREBUS_GATEWAY_AZURE_URL`/`CEREBUS_CONFIG_ID` for `azure`;
`CEREBUS_GATEWAY_DIRECT_URL` for `direct`) is validated for presence **before** the API key
is resolved (Feature 2) — so a configuration mistake surfaces immediately rather than
being masked behind, or delayed by, a slow/failing AWS Secrets Manager call. All values
(mode, URLs, config ID) are checked for truthiness only: whitespace-only strings pass, and
URLs are not parsed or scheme-validated (beyond the separate HTTPS requirement on an
`--api-base` override, FR-1.4) — this permissiveness is deliberate scope-limiting, not an
oversight (see Constraints).
**Verify:** `CEREBUS_MODE` unset or set to a value other than exactly `azure`/`direct`
raises `ValueError` naming the invalid value, without attempting key resolution; `azure`
mode without `CEREBUS_GATEWAY_AZURE_URL` or without `CEREBUS_CONFIG_ID` each raise
`ValueError` naming the missing variable, before any AWS call; `direct` mode without
`CEREBUS_GATEWAY_DIRECT_URL` raises `ValueError`; each mode's success path returns the
documented `api_base` + header shape (AR-1.2).

#### FR-1.3: The model id is auto-prefixed for LiteLLM's OpenAI-compatible routing
LiteLLM only treats a custom `api_base` as a generic OpenAI-compatible chat-completions
endpoint when the model id carries its `openai/` custom-provider prefix. When `--cerebus`
is set, every model id passed to a `Classifier` (the base `--model`, and any per-role
override: `--critic-model`, `--reconciler-model`, `--induction-model`) is prefixed with
`openai/` if not already present, so a user only ever types the bare model name/slug (e.g.
`gpt-4o-mini`) regardless of gateway mode. Users must supply a bare gateway model name or
slug when `--cerebus` is set — an already provider-qualified id (e.g. `azure/gpt-4o`) is
**not** detected or stripped, and prefixing it anyway (`openai/azure/gpt-4o`) will not
resolve correctly; this is an accepted limitation, not a validated error (see Constraints).
**Verify:** `cerebus_model_id("gpt-4o-mini")` returns `"openai/gpt-4o-mini"`;
`cerebus_model_id("openai/gpt-4o-mini")` returns it unchanged (idempotent); a `--cerebus
--critics` run over `cli.py` constructs the classification, critic, and reconciler
`Classifier` instances with distinctly-prefixed model ids matching `--model`/
`--critic-model`/`--reconciler-model` respectively (not just the base model); a `--cerebus
run` invocation over `experiment.py` does the same for the induction and classification
roles.

#### FR-1.4: An explicit `--api-base` still overrides the resolved gateway URL, but must be HTTPS
If the user passes `--api-base` explicitly alongside `--cerebus`, that value is used as the
endpoint instead of `CEREBUS_GATEWAY_AZURE_URL`/`CEREBUS_GATEWAY_DIRECT_URL` — an explicit
override always wins over an automatic resolution, matching this repo's existing
`--api-base` precedence over env-var-based endpoint resolution (`classifier.py`'s
`resolve_api_base`). The resolved API key and Portkey headers are still attached
regardless (the endpoint may change; the gateway's auth mechanism doesn't) — which means
an overridden endpoint receives the same gateway credential a misconfigured or mistyped
`--api-base` would otherwise leak in the clear. To bound this risk without a full
allowlist (out of scope — the value is the user's own CLI flag, not externally
attacker-controlled input), an `--api-base` override is rejected outright unless it starts
with `https://` when `--cerebus` is set; this check runs during the same fail-fast
validation phase as FR-1.6, before any file or run-directory work.
**Verify:** `--cerebus --api-base https://custom/v1` produces a `Classifier` whose
`api_base` is `https://custom/v1`, while `api_key`/`extra_headers` are still populated
from the Cerebus resolution; `--cerebus --api-base http://insecure/v1` exits non-zero with
an error naming the `https://` requirement, on both entry points, before any output file
or run directory is touched.

#### FR-1.5: Every classifier role in one invocation shares one, single resolution of the gateway configuration
`cli.py` constructs up to three `Classifier` instances (classification, critic,
reconciler); `experiment.py` constructs up to four (those three, plus induction, across
two separate calls to its internal `_construct_classifiers` helper for the `run`
subcommand). When `--cerebus` is set, gateway configuration (`api_base`/`api_key`/
`extra_headers`) is resolved **exactly once** per invocation — in `cli.py`'s `main()`
directly, and in `experiment.py`'s `main()` via a dedicated `_resolve_gateway_kwargs`
helper called once and threaded through both `_construct_classifiers` calls as an explicit
parameter — and the identical resulting dict is applied to every classifier role. There is
no per-role Cerebus toggle in this spec (see Out of Scope). Every role receives the *same*
`extra_headers` dict object (not a copy); this module never mutates it after construction,
but a caller reaching into a constructed `Classifier.extra_headers` and mutating it in
place would affect every sibling role sharing that invocation — an accepted, undocumented
edge case rather than a validated immutability contract.
**Verify:** a `--critics` run with `--cerebus` produces matching `extra_headers` on the
classification, critic, and reconciler `Classifier` instances, with each role's model id
distinctly prefixed per its own `--model`/`--critic-model`/`--reconciler-model` value; a
`run` subcommand invocation with `--cerebus` produces matching `extra_headers` on the
induction and classification `Classifier` instances, with distinctly prefixed model ids
per `--induction-model`/`--model`.

#### FR-1.6: Gateway misconfiguration fails before any output file or run-directory work, with a clear error
`cli.py` resolves and validates gateway configuration (FR-1.2's mode/URL/config-id checks,
FR-1.4's HTTPS check, and Feature 2's key resolution) before its one and only output
write (`classify_csv`) — categories/prompt files may already have been read by that point
(unrelated to this feature; that read order is unchanged), but nothing is written. In
`experiment.py`, the same resolution happens once, early in `main()`, before `--run-dir` is
created, emptied, or written to in any way. Any failure prints `Error: <message>` to
stdout and exits with status 1; this "clean error, not a bare traceback" guarantee holds
for both `main()` entry points, not for `classifier.py`'s functions called directly as a
library (a caller importing `build_cerebus_completion_kwargs` directly gets the raw
`ValueError`/`RuntimeError`, by design — `main()` is what adds the catch-and-print layer).
**Verify:** `experiment.py classify --cerebus` with `CEREBUS_MODE` unset exits 1, prints an
error (to stdout) naming `CEREBUS_MODE`, and creates no `--run-dir` at all; the equivalent
`cli.py` invocation exits 1 before writing any output file.

### Architectural Requirements

#### AR-1.1: `Classifier` gains two new, additive constructor parameters
`Classifier.__init__` gains `api_key: str | None = None` and
`extra_headers: dict[str, str] | None = None`. `_completion_kwargs()` includes each only
when it is *truthy* (`if self.api_key:` / `if self.extra_headers:`) — matching `api_base`'s
existing truthy check in the same method, not `temperature`'s `is not None` check (the two
existing params already differ from each other here; this feature follows `api_base`'s
convention, not `temperature`'s). A practical consequence: passing `api_key=""` or
`extra_headers={}` explicitly is indistinguishable from passing `None` — both are omitted
from the LiteLLM call. Every existing call site that doesn't pass either new parameter
constructs the exact same `_completion_kwargs()` output as before this feature.
**Verify:** `Classifier(...)._completion_kwargs([])` omits `api_key`/`extra_headers` when
both are left at their `None` default (or set to `""`/`{}`), and includes them verbatim
when set to a truthy value.

#### AR-1.2: `build_cerebus_completion_kwargs()` centralizes gateway resolution in `classifier.py`
One function, `build_cerebus_completion_kwargs() -> dict`, validates the `CEREBUS_MODE`
routing configuration described in FR-1.2, resolves the API key (Feature 2), and returns
`{"api_base": ..., "api_key": ..., "extra_headers": {...}}` — the exact kwargs `cli.py`/
`experiment.py` merge into every `Classifier(...)` call under `--cerebus`. Applying the
`--api-base` override (FR-1.4, including its HTTPS check) and the `openai/` model-id
prefix (FR-1.3) are call-site responsibilities, not something this function does itself —
`experiment.py` centralizes both into its own `_resolve_gateway_kwargs` helper (FR-1.5);
`cli.py` inlines the equivalent logic directly in `main()`.
**Verify:** the function's return value's three keys map 1:1 onto `Classifier`'s
`api_base`/`api_key`/`extra_headers` constructor parameters.

#### AR-1.3: This lives inside `classifier.py`, not a new module — `INV-1` is unaffected
`spec/ARCHITECTURE.md`'s INV-1 states `classifier.py` never imports any sibling
`query_classification` module (it's a leaf). This feature adds functions to `classifier.py`
itself rather than a new module, and its one new dependency (`boto3`) is external and
lazily imported inside `_resolve_cerebus_api_key()` — so `classifier.py` remains a
zero-internal-import leaf exactly as INV-1 requires. No ADR or INV-1 amendment is needed
for production code.

#### AR-1.4: `classifier.py`'s Cerebus functions and `cli.py` join INV-7's accepted test-import exceptions
`tests/test_cerebus.py` imports `query_classification.classifier` (for
`_resolve_cerebus_api_key`, `build_cerebus_completion_kwargs`, `cerebus_model_id`,
`reject_insecure_cerebus_endpoint`, and the `_cerebus_api_key_cache` module attribute for
test-isolation resets) and `query_classification.cli` (`build_parser`, `main`) directly —
both forbidden by INV-7's current wording ("Do not import
`categories.py`/`classifier.py`/`pipeline.py`/`cli.py` directly"), a conflict caught during
this spec's own critique. Neither module's Cerebus-relevant surface is part of
`__init__.py`'s public re-exports, and testing `cli.py`'s actual CLI wiring (flag
defaults, fail-fast validation order, per-role model-id prefixing) has no practical
public-API equivalent — mirroring the exact rationale Spec 1 and Spec 2 already used for
`debate.py`/`dataset_io.py`/`induction.py`/`experiment.py`. `spec/3-cerebus-gateway/ADR.md`
records the precise INV-7 amendment (scoped to the named functions above, not the whole
of either module), to be folded into `spec/ARCHITECTURE.md` by `/spec-close`.
**Verify:** `spec/3-cerebus-gateway/ADR.md` exists and names exactly the functions
`tests/test_cerebus.py` imports directly.

---

## Feature 2: Credential Resolution

**Who & why:** Developers with AWS access already use an `aws sso login` + profile-based
workflow for other internal services (mirrored in sibling Elsevier repos this spec's
design was modeled on). They want the Cerebus service key to resolve automatically from
that same mechanism without needing to manually copy a key into `.env` — while an explicit
`CEREBUS_API_KEY` should still work standalone for anyone without AWS access, or who
already has the key from a teammate.

### Functional Requirements

#### FR-2.1: `CEREBUS_API_KEY` is checked first
If the `CEREBUS_API_KEY` environment variable is set (non-empty, including whitespace-only
values — no trimming or format validation is applied), it is used directly and AWS Secrets
Manager is never contacted.
**Verify:** with `CEREBUS_API_KEY` set, `_resolve_cerebus_api_key()` returns it and makes
zero calls into the (fake, injected) `boto3` module.

#### FR-2.2: AWS Secrets Manager is the fallback when the env var is unset
When `CEREBUS_API_KEY` is unset, the key is fetched from AWS Secrets Manager:
`boto3.Session(profile_name=CEREBUS_AWS_PROFILE).client("secretsmanager",
region_name=CEREBUS_AWS_REGION).get_secret_value(SecretId=CEREBUS_SECRET_ID)`, then the
`SecretString` field (JSON-decoded; `SecretBinary` secrets are not supported) must decode
to a JSON object whose `CEREBUS_SECRET_KEY` field is a non-empty string — that string is
the resolved key. `CEREBUS_AWS_PROFILE`/`CEREBUS_AWS_REGION`/`CEREBUS_SECRET_ID`/
`CEREBUS_SECRET_KEY` each have working defaults (`kd-nonprod`/`us-east-1`/
`shared_genai/portkey-nonprod`/`sciencedirect_portkey_api_key` respectively — the same
secret coordinates used elsewhere internally) and are individually overridable via their
own env vars.
**Verify:** with `CEREBUS_API_KEY` unset and a fake `boto3.Session`/client returning a
valid `SecretString`, `_resolve_cerebus_api_key()` returns the decoded key and the exact
default profile/region/secret-id values are what's passed to the fake `Session`/client;
overriding `CEREBUS_AWS_PROFILE`/`CEREBUS_AWS_REGION`/`CEREBUS_SECRET_ID`/
`CEREBUS_SECRET_KEY` changes what's passed through / which JSON field is read; a
`SecretString` whose decoded field is missing, empty, or not a string raises the same
clear `RuntimeError` as any other resolution failure (FR-2.4).

#### FR-2.3: `boto3` is an optional, lazily-imported dependency
`boto3` is declared under `[project.optional-dependencies] cerebus = ["boto3"]` in
`pyproject.toml` — not a base dependency — and imported **inside**
`_resolve_cerebus_api_key()`, never at module import time, mirroring Spec 2's `datasets`
lazy-import pattern (FR-1.5 there). A user who never sets `--cerebus`, or who sets
`CEREBUS_API_KEY` directly, never needs `boto3` installed at all. If `import boto3` itself
raises `ModuleNotFoundError` (checked via the exception's `.name` attribute equaling
exactly `"boto3"`), the error names the exact install command; a `ModuleNotFoundError` for
anything else (one of `boto3`'s own transitive dependencies) is **not** misreported as
"boto3 is missing" — it propagates as-is, since telling someone to reinstall a package
that's already present would send them in circles.
**Verify:** `import query_classification.classifier` succeeds with `boto3` absent from the
environment; calling `_resolve_cerebus_api_key()` with `CEREBUS_API_KEY` unset and `boto3`
absent raises `RuntimeError` containing `pip install '.[cerebus]'`; the same call with a
`ModuleNotFoundError` for a *different* module name propagates that original exception
unchanged, not the `boto3`-specific `RuntimeError`.

#### FR-2.4: A resolution failure produces one clear, actionable error, with the exception's type but never its message text
If the AWS Secrets Manager call itself fails (expired SSO session, wrong profile, missing
secret, malformed/missing JSON field, network error, or any other exception), it is caught
and re-raised as a single `RuntimeError` naming the likely fix (`aws sso login --profile
<profile>`) and which `CEREBUS_*` variables to check, with only the underlying exception's
**type name** appended for diagnostic context — never `str(underlying)`. This is a
deliberate correction from an earlier draft of this feature, which appended the raw
exception message text: an AWS/SDK exception's message is not a value this spec can
guarantee is free of request details, identifiers, or other content unsafe to display
verbatim, so only the type name (which carries no such content) is included, matching
`dataset_io.py`'s/`induction.py`'s existing sanitized-failure convention elsewhere in this
codebase.
**Verify:** a fake AWS client that raises on `get_secret_value` produces a `RuntimeError`
whose message contains `aws sso login` and the raising exception's type name, but not that
exception's original message text; a fake client returning a `SecretString` missing the
expected JSON key produces the same kind of clear `RuntimeError`, not a raw `KeyError`/
`TypeError`.

#### FR-2.5: The AWS-resolved key is cached in-process only, never persisted to disk
Once resolved via the AWS Secrets Manager fallback, the key is cached in a module-level
variable for the remainder of the process, avoiding a repeated AWS call within one
invocation (e.g. when `experiment.py run` constructs both an induction and a classification
`Classifier`, both draw from the same single resolution — see FR-1.5). The env-var path
(FR-2.1) is not cached — it's a single `os.getenv` call each time, cheap enough not to need
it. Nothing is written back to `.env` or any other file — this is a deliberate
simplification from the sibling-repo pattern this feature was modeled on, which does
persist a fetched key to `.env`; that write-back is explicitly not replicated here (see Out
of Scope), so every new process invocation re-resolves from the environment/AWS. Changing
`CEREBUS_AWS_PROFILE`/`CEREBUS_AWS_REGION`/`CEREBUS_SECRET_ID`/`CEREBUS_SECRET_KEY`
mid-process has no effect once an AWS-resolved key is already cached — there is no
cache-invalidation or refresh operation.
**Verify:** two calls to `_resolve_cerebus_api_key()` within the same process, with
`CEREBUS_API_KEY` unset, result in exactly one call into the (fake) AWS client; a `.env`
file's contents are byte-identical before and after a resolution that used the AWS
fallback.

### Architectural Requirements

#### AR-2.1: Tests fake `boto3` via `sys.modules` injection — no live AWS calls
`tests/test_cerebus.py` never imports the real `boto3` or contacts AWS; a minimal fake
`boto3` module (a `Session` class whose `.client(...)` returns an object exposing
`get_secret_value`, recording the exact profile/region/secret-id it was called with) is
injected via `monkeypatch.setitem(sys.modules, "boto3", ...)`, matching this repo's
established pattern for faking other optional/lazy-imported dependencies (Spec 2's fake
`datasets` module). This satisfies `AGENTS.md`'s binding "no live network/provider calls in
tests" rule.
**Verify:** `tests/test_cerebus.py` contains no import of a real `boto3`/AWS SDK and makes
no network call; the full suite passes with `boto3` absent from the environment except
where a test explicitly injects the fake.

#### AR-2.2: Ship unit tests
The implementation ships a `pytest`-runnable suite (`tests/test_cerebus.py`) covering every
FR-1.x/FR-2.x **Verify:** condition above, including the error/edge cases named in each —
with no live network, AWS, or provider calls (AR-2.1). Both `main()` entry points'
CLI-level behavior under `--cerebus` (success and failure paths, including the HTTPS
override check and the per-role model-id/header wiring FR-1.3/FR-1.5 describe) is covered
directly, not only the underlying `classifier.py` primitives in isolation.
**Verify:** `tests/test_cerebus.py` exists and `pytest` passes against the implementation.

---

## Data Requirements

`CEREBUS_MODE`/`CEREBUS_GATEWAY_AZURE_URL`/`CEREBUS_GATEWAY_DIRECT_URL`/
`CEREBUS_CONFIG_ID`/`CEREBUS_API_KEY`/`CEREBUS_AWS_PROFILE`/`CEREBUS_AWS_REGION`/
`CEREBUS_SECRET_ID`/`CEREBUS_SECRET_KEY` are read from the process environment (via `.env`
+ `python-dotenv`, already loaded by both entry points' existing
`load_dotenv(override=True)` call) — none are written anywhere by this feature itself
(operators may of course choose to persist `CEREBUS_API_KEY` in their own `.env`, which is
expected and already `.gitignore`d, same as every other credential in that file).
`experiment.py`'s `run_config.json` (Spec 2) gains one new field, `cerebus: {enabled: bool,
mode: str | None}` — recording *whether* and in *which mode* a run used the gateway, for
audit purposes, but never the gateway URL, config ID, or key, consistent with Spec 2's
existing rule that `run_config.json` never records `--api-base`'s value or any credential.

## Integration Points

- `src/query_classification/classifier.py` — gains `_resolve_cerebus_api_key`,
  `build_cerebus_completion_kwargs`, `cerebus_model_id`, `reject_insecure_cerebus_endpoint`,
  and `Classifier`'s new `api_key`/`extra_headers` params. All other existing functions
  unchanged.
- `src/query_classification/cli.py` — gains `--cerebus`; resolves gateway kwargs once in
  `main()` (including the HTTPS check) and merges them into every `Classifier`
  construction site.
- `src/query_classification/experiment.py` — gains `--cerebus` on the shared `common`
  argument group; a new `_resolve_gateway_kwargs` helper resolves gateway kwargs once in
  `main()` and threads them through both `_construct_classifiers` calls; `run_config.json`
  gains the `cerebus` field.
- `pyproject.toml` — new `[project.optional-dependencies] cerebus = ["boto3"]`.
- `.env.example` (and the developer's own `.env`) — new `CEREBUS_*` placeholder variables.
- `tests/test_cerebus.py` — new test file (32 tests).
- `README.md` — new "Cerebus / Portkey gateway" section, `--cerebus` row in the Options
  table, and a note on the experiment runner's flag availability.
- `spec/3-cerebus-gateway/ADR.md` — INV-7 amendment (AR-1.4), folded into
  `spec/ARCHITECTURE.md` by `/spec-close`.

## Related Specs

| Spec | Relationship | Affected Requirements |
|------|-------------|---------------------|
| Spec 1: Multi-Agent Debate Classification with Critics | **Extends** — `--cerebus` applies uniformly to the critic/reconciler `Classifier` instances Spec 1 introduced in `cli.py`, with no change to `debate.py` itself | FR-1.5 |
| Spec 2: Experiment Runner | **Extends** — `--cerebus` applies to every role `experiment.py` constructs, including the induction `Classifier` Spec 2 introduced; reuses Spec 2's "validate everything before touching `--run-dir`" invariant (FR-3.4) for its own fail-fast requirement, its lazy-optional-dependency pattern (FR-1.5 there) for `boto3`, and adds one field to `run_config.json` (Data Requirements) | FR-1.1, FR-1.5, FR-1.6 |

## Constraints

- **No per-role Cerebus toggle.** `--cerebus` is all-or-nothing for a given invocation —
  there is no way to route only the critic role through Cerebus while the classification
  role uses a direct provider, or vice versa.
- **No per-model deployment mapping table.** Unlike the sibling-repo pattern this feature
  was modeled on (which maps specific model names to specific Portkey Config IDs/slugs in
  a checked-in table), this spec has exactly one `CEREBUS_CONFIG_ID` (azure mode) or relies
  on `--model`'s own value as the slug (direct mode) per invocation. A practical
  consequence in `azure` mode: every role shares the *same* Config ID regardless of its own
  `--critic-model`/`--reconciler-model`/`--induction-model` value, and whether Portkey's
  config or the per-request model field actually selects the deployment on Azure's side is
  a Portkey-workspace-specific detail this spec does not define or depend on — operators
  using `azure` mode with per-role model overrides should confirm their workspace's actual
  behavior.
- **Minimal input validation, by design.** `CEREBUS_MODE`/URLs/config ID/AWS coordinates
  are checked for presence (truthiness) only — not trimmed, case-normalized, length-bounded,
  or scheme/host-validated (beyond FR-1.4's HTTPS check on an explicit `--api-base`
  override specifically). A malformed value one env var away from correct (extra
  whitespace, wrong case) surfaces as a downstream HTTP/auth failure, not a specific
  validation error naming the exact problem.
- **No `.env` auto-persistence.** As stated in FR-2.5, an AWS-resolved key is never written
  back to disk — every process re-resolves it. This trades a small repeated-latency cost
  per process (one Secrets Manager call, not per LLM request — see Feature 1's "resolved
  once per invocation") for not silently rewriting a file `python-dotenv` will later reload.
- **`.env`'s `override=True` can mask an already-set process environment variable.** Both
  entry points call `load_dotenv(override=True)` (pre-existing behavior, unchanged by this
  feature) before any Cerebus resolution — meaning a value already present in `.env`
  replaces an already-exported shell/CI environment variable of the same name, not the
  other way around. An operator who exports `CEREBUS_API_KEY` in their shell expecting it
  to take precedence over a stale/placeholder `.env` entry will be surprised; this is an
  existing repo-wide behavior this spec inherits rather than changes.
- **Gateway endpoint values are user-supplied, not defaulted.** `CEREBUS_GATEWAY_AZURE_URL`/
  `CEREBUS_GATEWAY_DIRECT_URL`/`CEREBUS_CONFIG_ID` ship as empty placeholders in
  `.env.example` — this repo does not assume access to any particular Elsevier Portkey
  workspace, unlike the AWS Secrets Manager coordinates (FR-2.2), which do ship with
  working defaults since they identify secret *coordinates*, not workspace-specific
  routing.
- **Gateway latency and availability become a shared dependency for every LLM call under
  `--cerebus`.** This adds a network hop (and, transitively, Cerebus/Portkey's own
  availability and rate limits) to every request `Classifier.classify` makes, on top of
  this repo's existing `--workers`/`--sampling-runs` concurrency (Spec 1's Constraints);
  there is no separate connection/read timeout or retry policy layered on top of
  `litellm`/`botocore`'s own defaults, and no explicit latency budget is specified. The
  operational mitigation is the same as for any other misconfiguration: omit `--cerebus`.
- **Data governance is the operator's responsibility, not validated by this feature.**
  Enabling `--cerebus` changes which system receives request text (train examples under
  `induce`, test/query text under `classify`/`run`) and how that text is logged/retained on
  the gateway side — this spec does not audit or restrict that; it only wires the routing
  mechanism. See Spec 2's existing "Data egress" README note for the equivalent
  provider-routing consideration this inherits.

## Out of Scope

- **Gemini/other-provider slug tables, `use_responses_api`, or reasoning-effort-suffix
  handling** — the sibling-repo pattern this feature was modeled on has per-model
  bookkeeping (`RESPONSES_API_MODELS`, `CEREBUS_DEPLOYMENTS`) for a much larger model
  surface than this tool exposes; none of that is replicated.
- **A `DEFAULT_LLM_PROVIDER`-style env-var toggle** — this repo uses an explicit `--cerebus`
  flag instead of an env-var-driven default provider switch.
- **A full endpoint allowlist for `--api-base` overrides** — FR-1.4's HTTPS-only check is a
  minimal guard, not a trust boundary against a malicious `--api-base` value; the value is
  the user's own CLI flag.
- **Auto-detecting or validating AWS SSO session freshness** — a stale SSO session simply
  surfaces as an AWS error, funneled through FR-2.4's generic error handling; there is no
  separate pre-check.
- **Rotating, revoking, or otherwise managing the Cerebus service key's lifecycle** —
  entirely out of scope; this spec only resolves and uses whatever key is currently valid.
- **`config_id` name to task-type mapping tables in `experiment.py`** — a single
  `CEREBUS_CONFIG_ID` applies uniformly per invocation, not per subcommand or per role.
- **Gateway compatibility/version guarantees** — whether Cerebus/Portkey supports LiteLLM's
  structured-output (`response_format`) request shape as well as its JSON-mode fallback,
  and any resulting double-call cost on an unsupported request, is not verified or bounded
  by this spec.
- **Request-level observability (correlation IDs, redaction rules, metrics)** — beyond the
  `run_config.json` `cerebus.enabled`/`cerebus.mode` fields (Data Requirements), no
  additional tracing/logging is added.

## Spec Completeness Checklist

- [x] **Scope & acceptance criteria** — every FR carries a **Verify:** condition; Out of
  Scope names eight explicitly excluded directions.
- [x] **Testing strategy** — `tests/test_cerebus.py` (32 tests) covers every FR-1.x/FR-2.x
  Verify condition, including CLI-level success-path wiring checks (AR-2.2) added after
  this spec's critique found the original draft's end-to-end claims untested.
- [x] **Existing patterns** — modeled explicitly on Spec 2's lazy-optional-dependency
  pattern (`datasets`/`hf` extra), this repo's existing `.env`/`python-dotenv` convention,
  and `dataset_io.py`'s/`induction.py`'s sanitized-failure convention (FR-2.4).
- [x] **Dependencies** — `boto3` justified in FR-2.3 (AWS Secrets Manager access) and kept
  optional/lazy so it's never required for the common case.
- [x] **Architecture & interfaces** — AR-1.1/AR-1.2/AR-1.3 cover the interface surface and
  confirm `INV-1` compliance; AR-1.4 (with `ADR.md`) resolves a real `INV-7` test-import
  conflict this spec's critique found, rather than leaving it unaddressed.
- [x] **Error handling & failure modes** — FR-1.2 (validation order), FR-1.4 (HTTPS
  rejection), FR-1.6 (fail-fast + error channel), FR-2.3, FR-2.4 (sanitization) each define
  a specific failure mode and its exact, verified behavior.
- [x] **Security review** — FR-1.4 bounds (not eliminates) arbitrary-endpoint credential
  exposure; FR-2.4 requires type-name-only diagnostics, never the underlying exception's
  message text; FR-2.5/Constraints rule out writing the key to disk; Constraints documents
  the `.env override=True` precedence surprise and the data-governance responsibility this
  spec does not itself validate; Data Requirements confirms `run_config.json` never
  records the key, URL, or config ID.
- [x] **Performance impact** — FR-2.5/FR-1.5 address the one caching-relevant cost
  (AWS Secrets Manager calls, now resolved exactly once per invocation); Constraints names
  the added-network-hop/no-timeout-policy cost as an accepted, undefined-budget risk rather
  than claiming it away.
- [x] **Rollout & migration** — Goals' "fully opt-in" line and FR-1.1 (default `False`,
  verified byte-identical completion kwargs when omitted) cover this: no migration needed.
- [x] **Assumptions & risks** — Constraints names the explicit assumptions (no workspace
  access presumed for gateway URLs; AWS coordinates default to a shared, previously-working
  secret path that may not apply to every AWS account; minimal input validation; gateway
  compatibility with structured-output requests unverified).
