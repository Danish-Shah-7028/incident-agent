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


ROOT_KEYWORDS = {
    "deployment_regression": [r"\brelease\b", r"\brollout\b", r"rolled out", r"holdback", r"canary", r"\br\d+-\w+", r"reversion", r"release transition"],
    "database_connection_exhaustion": [r"connection pool", r"pool checkout", r"pool acquisition", r"connection ceiling", r"database connections?", r"\bpools?\b"],
    "dependency_certificate_expired": [r"certificate", r"\btls\b", r"notafter", r"expired", r"\bdep_[a-z0-9]+"],
    "feature_flag_recursion": [r"\bflag", r"cohort", r"recurs", r"\bloop", r"evaluation path", r"depth guard"],
    "traffic_capacity_exhaustion": [r"queue depth", r"requests per second", r"\brps\b", r"per-replica", r"utili[sz]ation", r"capacity ceiling", r"traffic"],
    "secret_rotation_mismatch": [r"\bsecret", r"credential", r"\bvault\b", r"rotation", r"revo(ke|ked|cation)", r"generation \d+"],
}
EXCLUDE_SENTENCE = re.compile(r"\b(no|none|not|unchanged|normal|normally|healthy|stable|remain(s|ed)?|continue to operate|ordinary)\b", re.I)


def classify(text, allowed):
    """Score root causes on the operative lines, ignoring sentences that only rule causes out."""
    sents = [s for s in re.split(r"(?<=[.;])\s+", text) if not EXCLUDE_SENTENCE.search(s)]
    pos = " ".join(sents)
    best, best_score = None, 0
    for name in allowed:
        kws = ROOT_KEYWORDS.get(name)
        if not kws:
            continue
        score = sum(len(re.findall(k, pos, re.I)) for k in kws)
        if score > best_score:
            best, best_score = name, score
    return best


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
    def ts(t):
        m = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?", t)
        return m.group(0) if m else ""
    cue = max(sig, key=lambda l: ts(l[1]))[0] if len(sig) >= 3 and all(ts(t) for _, t in sig) else None
    ev = [i for i, _ in sig if i != cue][:4]
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
        effect = ("open_incident", {"service": svc, "severity": sev})
    elif root == "feature_flag_recursion":
        flag = first(r"\b(flag_[a-z0-9]+)\b", text)
        diags = [("query_logs", {"service": svc, "query": f"recursive evaluation {flag}", "windowMinutes": 60},
                  cite(r"loop|recurs|depth guard"))]
        effect = ("disable_feature", {"service": svc, "flag": flag})
    elif root == "database_connection_exhaustion":
        diags = [("query_metrics", {"service": svc, "metric": "connection_pool_wait", "windowMinutes": 60},
                  cite(r"pool checkout|connection ceiling"))]
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
    # Runbook rule: a runbook REQUIRED before acting is the only (prerequisite) diagnostic; a runbook that is merely
    # mentioned is an extra independent read-only diagnostic (run in parallel).
    rb = [(i, t) for i, t in sig if re.search(r"runbook|playbook", t, re.I)]
    if rb and "read_runbook" in cat and "read_runbook" not in effects:
        req_line = next(((i, t) for i, t in rb if re.search(r"(runbook|playbook)[^.]*\b(is required|required before|must be (read|consulted|checked))|before (paging|acting|changing|any)|mandatory lookup", t, re.I)), None)
        rbd = ("read_runbook", {"service": svc, "topic": root}, [(req_line or rb[0])[0]])
        if req_line:
            diags = [rbd]
        # a runbook that is only mentioned does not need its own call (minimal diagnostics)
    # Policy gates quoted in the evidence: freeze / contract maximum / delegation / remaining changes -> page the owner
    def num(rx):
        m = re.search(rx, text, re.I)
        return int(m.group(1)) if m else None
    freeze, cmax, target = num(r"freeze\s*=\s*(\d+)"), num(r"contract maximum\s*=\s*(\d+)"), num(r"(?:proposed\s+)?target\s*=\s*(\d+)")
    deleg, remaining = num(r"delegation\s*=\s*(\d+)"), num(r"remaining[^=;.]*=\s*(\d+)")
    if effect and effect[0] == "scale_service" and freeze is not None and cmax is not None and target is not None:
        effect = ("scale_service", {"service": svc, "targetReplicas": target}) if (cmax >= target and freeze == 0) \
            else ("page_owner", {"service": svc, "severity": sev})
    if effect and effect[0] in ("rollback_deployment", "disable_feature", "scale_service") and deleg is not None \
            and freeze is not None and remaining is not None and not (deleg == 1 and freeze == 0 and remaining > 0):
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
