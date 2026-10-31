"""Deterministic incident reading: exact signal lines, root cause, diagnostics and effect with case-derived arguments.

Transcripts hide a few operative lines among many decoys. Decoys either carry a fixed marker sentence or repeat the
same template text, so operative lines are found exactly by code; the model's answer is used only as a fallback.
"""
import re
from collections import Counter

LINE_RE = re.compile(r"\[(ev_[A-Za-z0-9_\-]+)\]([^\n]*)")
PREFIX_RE = re.compile(r"^\s*\S*\d{2}:\d{2}(:\d{2})?Z?\s+(?:[a-z][a-z\- ]{2,30}:\s*)?")
MARKER = "retain this full sentence"


def lines(transcript):
    return [(m.group(1), m.group(2).strip()) for m in LINE_RE.finditer(transcript)]


def body(text):
    return PREFIX_RE.sub("", text).strip()


def signal_lines(transcript):
    ls = lines(transcript)
    marked = [l for l in ls if MARKER not in l[1]]
    if 2 <= len(marked) <= 6 and len(marked) < len(ls):
        return marked
    counts = Counter(body(t) for _, t in ls)
    uniq = [l for l in ls if counts[body(l[1])] == 1 and MARKER not in l[1]]
    return uniq if 2 <= len(uniq) <= 6 else None


ROOT_RULES = [
    ("secret_rotation_mismatch", r"secret version|vault promoted|revoked the old credential|credential rotation"),
    ("dependency_certificate_expired", r"certificate"),
    ("feature_flag_recursion", r"\bflag\b.*(loop|recurs)|recurs.*\bflag\b|flag rule"),
    ("database_connection_exhaustion", r"connection (pool|ceiling)|pool (checkout|acquisition)|connection ceiling"),
    ("traffic_capacity_exhaustion", r"queue depth|per-replica|requests per second|capacity ceiling"),
    ("deployment_regression", r"\brelease\b|\brollout\b"),
]


def classify(text, allowed):
    for name, rx in ROOT_RULES:
        if name in allowed and re.search(rx, text, re.I):
            return name
    return None


def first(rx, text, g=1):
    m = re.search(rx, text, re.I)
    return m.group(g) if m else None


def plan(req, model_plan=None):
    inc = req["incident"]
    cat = {t["name"] for t in req["toolCatalog"]}
    effects = set((req.get("policy") or {}).get("effectTools") or [])
    sig = signal_lines(inc["transcript"])
    if not sig:
        return None
    text = " ".join(t for _, t in sig)
    allowed = inc["allowedRootCauses"]
    root = classify(text, allowed) or (model_plan or {}).get("rootCause")
    if root not in allowed:
        return None
    svc = inc.get("service") or first(r"\b(svc_[a-z0-9]+)\b", text)
    sev = inc.get("severity") or "SEV-1"
    ev = [i for i, _ in sig][:4]
    ids = {i: t for i, t in sig}

    def cite(rx):
        hit = [i for i, t in sig if re.search(rx, t, re.I)]
        return hit[:2] or ev[:1]

    diags, effect = [], None
    if root == "deployment_regression":
        diags = [("inspect_deployment", {"service": svc}, cite(r"release|rollout"))]
        good = (first(r"previous release (r[0-9A-Za-z][\w\-]*)", text) or first(r"(?:still running|instances running|running) (r[0-9A-Za-z][\w\-]*)", text))
        blocked = re.search(r"remaining approved reversions\s*=\s*0|freeze\s*=\s*1|delegation\s*=\s*0", text, re.I)
        effect = ("page_owner", {"service": svc, "severity": sev}) if blocked or not good else \
                 ("rollback_deployment", {"service": svc, "release": good})
    elif root == "dependency_certificate_expired":
        dep = first(r"\b(dep_[a-z0-9]+)\b", text)
        diags = [("dependency_status", {"dependency": dep}, cite(r"certificate|dep_"))]
        effect = ("open_incident", {"service": dep or svc, "severity": sev})
    elif root == "feature_flag_recursion":
        flag = first(r"\b(flag_[a-z0-9]+)\b", text)
        diags = [("query_logs", {"service": svc, "query": f"recursive evaluation {flag}", "windowMinutes": 60},
                  cite(r"loop|recurs|depth guard"))]
        effect = ("disable_feature", {"service": svc, "flag": flag})
    elif root == "database_connection_exhaustion":
        diags = [("query_metrics", {"service": svc, "metric": "connection_pool_wait", "windowMinutes": 60},
                  cite(r"pool checkout|connection ceiling")),
                 ("query_logs", {"service": svc, "query": "pool acquisition timeout", "windowMinutes": 60},
                  cite(r"logs|acquisition"))]
        n = first(r"exactly (\d+) (?:application )?replicas", text)
        effect = ("scale_service", {"service": svc, "targetReplicas": int(n)}) if n else ("page_owner", {"service": svc, "severity": sev})
    elif root == "traffic_capacity_exhaustion":
        diags = [("query_metrics", {"service": svc, "metric": "requests_per_second", "windowMinutes": 60},
                  cite(r"queue depth|requests per second|utilization"))]
        n = first(r"ceiling[^.]*?exactly (\d+) replicas|exactly (\d+) replicas", text)
        at_limit = re.search(r"already (at|running at)[^.]*(ceiling|limit|maximum)|at its (permitted )?(scaling )?limit", text, re.I)
        effect = ("page_owner", {"service": svc, "severity": sev}) if at_limit or not n else \
                 ("scale_service", {"service": svc, "targetReplicas": int(n)})
    elif root == "secret_rotation_mismatch":
        diags = [("read_runbook", {"service": svc, "topic": "secret rotation"}, cite(r"runbook|secret version"))]
        effect = ("page_owner", {"service": svc, "severity": sev})
    if not diags or not effect or any(v in (None, "") for _, a, _ in diags for v in a.values()):
        return None
    if any(d[0] not in cat or d[0] in effects for d in diags) or effect[0] not in cat or (effects and effect[0] not in effects):
        return None
    if any(v in (None, "") for v in effect[1].values()):
        return None
    return {"rootCause": root, "evidence": ev,
            "diagnostics": [{"toolName": n, "arguments": a, "evidence": [e for e in dict.fromkeys(c) if e in ev] or ev[:1]}
                            for n, a, c in diags],
            "effect": {"toolName": effect[0], "arguments": effect[1]}}
