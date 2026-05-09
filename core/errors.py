"""Shared exception types used across services and backends."""


class ImaginaError(Exception):
    """Base for everything Imagina raises."""


class BackendUnavailable(ImaginaError):
    """Selected backend cannot be reached (missing key, daemon down, OOM)."""


class QuotaExceeded(ImaginaError):
    """Provider rejected the request for quota or rate-limit reasons."""


class ConfigError(ImaginaError):
    """Configuration is missing or malformed."""


class GenerationFailed(ImaginaError):
    """Backend ran but produced no usable output."""
