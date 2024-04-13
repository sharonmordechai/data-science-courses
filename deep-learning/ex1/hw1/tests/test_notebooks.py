import pytest

import os
import logging
import nbformat
import nbconvert
from pathlib import Path
from contextlib import contextmanager

from tests import REPO_ROOT

NOTEBOOKS_DIR = Path(__file__).parent.joinpath("notebooks")
NOTEBOOK_PATHS = tuple(NOTEBOOKS_DIR.glob("*.ipynb"))
CELL_TIMEOUT_SECONDS = 60 * 5

HW_NOTEBOOK_PATHS = {
    f"hw{i}": tuple(REPO_ROOT.joinpath(f"hw{i}").glob("*.ipynb"))
    for i in range(1, 4 + 1)
}

_LOG = logging.getLogger(__name__)


@contextmanager
def _change_pwd(new_pwd):
    orig_pwd = os.getcwd()
    try:
        os.chdir(new_pwd)
        yield
    finally:
        os.chdir(orig_pwd)


def _run_notebook(notebook_path: Path):

    with _change_pwd(notebook_path.parent):
        _LOG.info(f"Executing notebook {notebook_path}...")

        # Parse notebook
        with open(str(notebook_path), "r") as f:
            nb = nbformat.read(f, as_version=4)

        # Create preprocessor which executes the notebook in memory
        # nothing is written back to the file.
        ep = nbconvert.preprocessors.ExecutePreprocessor(
            timeout=CELL_TIMEOUT_SECONDS, kernel_name="python3"
        )

        # Execute. If an exception is raised inside the notebook,
        # this test will fail.
        ep.preprocess(nb)


class TestNotebooks:
    @pytest.mark.parametrize(
        "notebook_path",
        HW_NOTEBOOK_PATHS["hw1"],
        ids=[f.stem for f in HW_NOTEBOOK_PATHS["hw1"]],
    )
    def test_hw1(self, notebook_path: Path):
        _run_notebook(notebook_path)

    @pytest.mark.parametrize(
        "notebook_path",
        HW_NOTEBOOK_PATHS["hw2"],
        ids=[f.stem for f in HW_NOTEBOOK_PATHS["hw2"]],
    )
    def test_hw2(self, notebook_path: Path):
        _run_notebook(notebook_path)

    @pytest.mark.parametrize(
        "notebook_path",
        HW_NOTEBOOK_PATHS["hw3"],
        ids=[f.stem for f in HW_NOTEBOOK_PATHS["hw3"]],
    )
    def test_hw3(self, notebook_path: Path):
        _run_notebook(notebook_path)

    # TODO: hw4
