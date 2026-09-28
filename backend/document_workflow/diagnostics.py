"""Content-free file-operation evidence. Never forward arbitrary tool data."""
STAGES = frozenset({"before_lease", "before_navigation", "guard_target", "guard_regex",
    "guard_rules", "observer_lease", "navigation_dispatch", "navigation_wait", "human_grant",
    "human_file", "validation", "authority", "owner_destination", "html_document",
    "chrome_document", "owner_after_observation", "changed_href", "preview_wait"})
CHECKS = frozenset({"probe_present", "url_matches", "snapshot_matches", "document_matches", "dom_unchanged"})


def safe_diagnostic(data):
    output = {}
    if not isinstance(data, dict):
        return output
    stage = data.get("preview_stage")
    if isinstance(stage, str) and stage in STAGES:
        output["stage"] = stage
    checks = data.get("snapshot_check")
    if isinstance(checks, dict):
        output["checks"] = {key: value for key, value in checks.items()
            if key in CHECKS and type(value) is bool}
    return output
