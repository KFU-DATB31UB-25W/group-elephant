"""
Shared interfaces/contracts across workstreams.

Put shared types (e.g. pydantic models), dataset schemas, and API contracts here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleContract:
    """Placeholder for shared schemas/contracts."""

    pass
