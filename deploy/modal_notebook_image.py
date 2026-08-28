"""Custom image for a Modal Notebook that *drives* the NextSearch harness.

Deploy this once so the notebook kernel boots with every runtime + experiment
dependency preinstalled — then your first notebook cell only clones the repo and
installs it with `--no-deps`, which is instant. In the Notebook sidebar, pick
this image by searching for the deployed app `nextsearch-notebook`.

    pip install modal && modal setup
    modal deploy deploy/modal_notebook_image.py

This image intentionally does **not** include vLLM: the notebook is a CPU driver
that talks to a GPU endpoint (deploy/modal_lfm_server.py) over
`NEXTSEARCH_BASE_URL`, so the kernel stays cheap and starts in seconds.
"""

import modal

# Mirrors pyproject.toml [project.dependencies] plus the 'experiment' extra.
# Keep in sync with pyproject, or when deploying from the repo root swap this
# for: modal.Image.debian_slim(...).pip_install_from_pyproject(
#     "pyproject.toml", optional_dependencies=["experiment"])
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "openai>=1.60", "httpx>=0.27", "datasets>=3.0",
        "huggingface-hub>=0.28", "python-dateutil>=2.9",
        "matplotlib>=3.7", "ipython>=8.0",
    )
)

app = modal.App("nextsearch-notebook", image=image)


@app.function()
def kernel_image():
    """Placeholder so the image is deployable and selectable in the Notebooks
    sidebar. The notebook borrows the image; it never calls this."""
    return "ok"
