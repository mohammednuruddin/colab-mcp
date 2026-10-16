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

from colab_mcp.state import ConnectionState, load_state, save_state


def test_round_trip(tmp_path):
    save_state(
        tmp_path,
        ConnectionState(token="tok", port=4321, last_notebook_url="https://x/nb"),
    )
    state = load_state(tmp_path)
    assert state == ConnectionState(
        token="tok", port=4321, last_notebook_url="https://x/nb"
    )
    assert (tmp_path / "connection.json").stat().st_mode & 0o777 == 0o600


def test_missing_and_malformed(tmp_path):
    assert load_state(tmp_path) is None
    (tmp_path / "connection.json").write_text("{not json")
    assert load_state(tmp_path) is None
    (tmp_path / "connection.json").write_text('{"token": "", "port": 99999}')
    assert load_state(tmp_path) is None
