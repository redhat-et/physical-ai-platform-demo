"""Pure unit tests -- no live agent, LLM, or cluster required.

Overrides the parent tests/conftest.py's session-scoped autouse
`_agent_reachable` fixture (which otherwise skips every test in the session,
including these, when no live agent is reachable at PLATFORM_AGENT_URL).
"""
import pytest


@pytest.fixture(scope="session", autouse=True)
def _agent_reachable():
    pass
