# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from colab_mcp.notebooks import (
    UnsupportedNotebookUrl,
    connection_url,
    normalize_notebook_url,
)

C = "https://colab.research.google.com"


@pytest.mark.parametrize(
    "link,expected",
    [
        (f"{C}/drive/1AbC_dEf-123", f"{C}/drive/1AbC_dEf-123"),
        (f"{C}/drive/1AbC_dEf-123#scrollTo=abc", f"{C}/drive/1AbC_dEf-123"),
        (f"{C}/drive/1AbC_dEf-123?usp=sharing", f"{C}/drive/1AbC_dEf-123?usp=sharing"),
        (
            "https://colab.google.com/github/o/r/blob/main/a.ipynb",
            f"{C}/github/o/r/blob/main/a.ipynb",
        ),
        (
            "colab.research.google.com/notebooks/intro.ipynb",
            f"{C}/notebooks/intro.ipynb",
        ),
        (
            "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing",
            f"{C}/drive/1AbCdEfGhIjKlMnOp",
        ),
        (
            "https://drive.google.com/open?id=1AbCdEfGhIjKlMnOp",
            f"{C}/drive/1AbCdEfGhIjKlMnOp",
        ),
        ("1AbCdEfGhIjKlMnOp", f"{C}/drive/1AbCdEfGhIjKlMnOp"),
        (
            "https://github.com/org/repo/blob/main/nb/train.ipynb",
            f"{C}/github/org/repo/blob/main/nb/train.ipynb",
        ),
        (
            "https://raw.githubusercontent.com/org/repo/main/nb/train.ipynb",
            f"{C}/github/org/repo/blob/main/nb/train.ipynb",
        ),
        (
            "https://gist.github.com/someone/0123456789abcdef",
            f"{C}/gist/someone/0123456789abcdef",
        ),
    ],
)
def test_normalize(link, expected):
    assert normalize_notebook_url(link) == expected


@pytest.mark.parametrize(
    "link",
    [
        "",
        "https://example.com/x.ipynb",
        "https://github.com/org/repo",
        "https://github.com/org/repo/blob/main/README.md",
        f"{C}/",
    ],
)
def test_rejects(link):
    with pytest.raises(UnsupportedNotebookUrl):
        normalize_notebook_url(link)


def test_connection_url():
    assert (
        connection_url(f"{C}/drive/x", "tok", 9999)
        == f"{C}/drive/x#mcpProxyToken=tok&mcpProxyPort=9999"
    )
