from setuptools import setup, find_packages

setup(
    name="gui-agent",
    version="1.0.0",
    description="Autonomous Visual GUI Operator for Linux (X11) UI and software testing",
    author="Google DeepMind Agentic Pair Programming",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "gui_agent.web": ["static/*"],
    },
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.8.0",
        "openai>=1.40.0",
        "google-genai>=2.0.0",
        "PyAutoGUI>=0.9.54",
        "pillow>=10.4.0",
        "python-xlib>=0.33",
        "requests>=2.32.0",
        "httpx>=0.27.0",
    ],
    entry_points={
        "console_scripts": [
            "run-agent=gui_agent.cli:main",
            "agent-dashboard=gui_agent.web.app:main",
        ],
    },
    python_requires=">=3.10",
)
