from importlib import resources
from importlib.metadata import version
from pathlib import Path
from setuptools_scm import get_version


def get_agent_version():
    """Get a long "local" version."""

    try:
        with resources.path("jax_agents", "__init__.py") as path:
            repo = Path(path).parent.parent
            _v = get_version(repo)
    except (LookupError, AssertionError):
        print("lr")
        _v = version("jax_agents")

    return _v
