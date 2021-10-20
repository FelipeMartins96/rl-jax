from setuptools import setup

setup(
    name="jax-agents",
    url="https://github.com/robocin/rSoccer",
    description="jax reinforcement learning agents",
    packages=["jax_agents"],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=["setuptools_scm"],
)
