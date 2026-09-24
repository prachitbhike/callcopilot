"""Model-aware request kwargs. Sonnet 5 / Opus 5-era models reject `temperature` and think by default;
forced tool use needs thinking disabled there. Older models (Haiku 4.5) take temperature."""


def sampling_kwargs(model: str, temperature: float) -> dict:
    legacy = any(t in model for t in ("haiku-4", "sonnet-4", "opus-4", "3-"))
    return {"temperature": temperature} if legacy else {"thinking": {"type": "disabled"}}


def client_kwargs() -> dict:
    """Org-level (non-workspace-scoped) keys need the anthropic-workspace-id header."""
    import os
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return {"default_headers": {"anthropic-workspace-id": ws}} if ws else {}
