from pathlib import Path

import pytest

from sentinel_runtime import ConfigError
from sentinel_runtime.config import (
    DEFAULT_MODEL,
    SentinelConfig,
    find_config_file,
    is_path_reference,
    resolve_eval_suite_path,
    resolve_model,
    resolve_predicate_set_path,
)

PREDICATE_FILE = """format: sentinel/predicates/v1
name: demo
predicates:
  a: {kind: noul, question: "q?"}
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_finds_config_in_parent_directories(tmp_path: Path) -> None:
    config_path = _write(tmp_path / "sentinel.toml", '[runtime]\nmodel = "jev-1.13.0"\n')
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_config_file(nested) == config_path
    config = SentinelConfig.load(nested)
    assert config.model == "jev-1.13.0"
    assert config.root == tmp_path.resolve()


def test_no_config_is_fine(tmp_path: Path) -> None:
    assert find_config_file(tmp_path) is None
    assert SentinelConfig().model is None


def test_paths_and_pricing_are_parsed_relative_to_the_file(tmp_path: Path) -> None:
    _write(
        tmp_path / "sentinel.toml",
        '[paths]\npredicates = "semantic/predicates"\nevals = "semantic/evals"\n'
        '[pricing."jev-1.13.0"]\ninput_per_mtok = 0.042\nas_of = "2026-09-17"\n',
    )
    config = SentinelConfig.from_file(tmp_path / "sentinel.toml")
    assert config.predicates_dir == tmp_path.resolve() / "semantic/predicates"
    assert config.evals_dir == tmp_path.resolve() / "semantic/evals"
    assert config.pricing["jev-1.13.0"].input_per_mtok == 0.042
    assert config.pricing["jev-1.13.0"].output_per_mtok == 0.0


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("[runtime]\nmodle = 'x'\n", "runtime.modle"),
        ("[unknown]\n", "unknown"),
        ("[pricing.m]\ninput_per_mtok = -1\n", "pricing.m.input_per_mtok"),
        ("not toml = = \n", "invalid TOML"),
    ],
)
def test_invalid_config_is_reported(tmp_path: Path, text: str, fragment: str) -> None:
    path = _write(tmp_path / "sentinel.toml", text)
    with pytest.raises(ConfigError, match=fragment.replace(".", r"\.")):
        SentinelConfig.from_file(path)


def test_model_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SentinelConfig(model="from-toml")
    assert resolve_model() == DEFAULT_MODEL
    assert resolve_model(config=config) == "from-toml"
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "from-typesafe-env")
    assert resolve_model(config=config) == "from-typesafe-env"
    monkeypatch.setenv("SENTINEL_MODEL", "from-sentinel-env")
    assert resolve_model(config=config) == "from-sentinel-env"
    assert resolve_model(runtime_model="from-runtime", config=config) == "from-runtime"
    assert resolve_model("explicit", runtime_model="from-runtime", config=config) == "explicit"
    monkeypatch.setenv("SENTINEL_MODEL", "   ")
    assert resolve_model(config=config) == "from-typesafe-env"


@pytest.mark.parametrize(
    ("reference", "is_path"),
    [
        ("agent_action", False),
        ("agent_action.yaml", True),
        ("predicates/agent_action", True),
        ("./agent_action", True),
        (Path("agent_action"), True),
    ],
)
def test_path_or_name(reference: str | Path, is_path: bool) -> None:
    assert is_path_reference(reference) is is_path


def test_resolves_predicate_sets_by_name_and_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    default = _write(tmp_path / "predicates" / "demo.yaml", PREDICATE_FILE)
    assert (
        resolve_predicate_set_path("demo", config=SentinelConfig()) == tmp_path / "predicates" / "demo.yaml"
    )
    configured = _write(tmp_path / "semantic" / "demo.yml", PREDICATE_FILE)
    config = SentinelConfig(predicates_dir=tmp_path / "semantic")
    assert resolve_predicate_set_path("demo", config=config) == configured
    assert resolve_predicate_set_path("predicates/demo.yaml", config=config) == Path("predicates/demo.yaml")
    assert resolve_predicate_set_path(
        "../predicates/demo.yaml", relative_to=tmp_path / "semantic", config=config
    ) == (tmp_path / "semantic" / "../predicates/demo.yaml")
    assert default.exists()


def test_missing_predicate_sets_list_where_they_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match=r"predicate set 'nope' not found; looked in: .*nope\.yaml"):
        resolve_predicate_set_path("nope", config=SentinelConfig())
    with pytest.raises(ConfigError, match="predicate file not found"):
        resolve_predicate_set_path("nope.yaml", config=SentinelConfig())


def test_resolves_eval_suites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cases = _write(tmp_path / "evals" / "demo" / "cases.yaml", "format: sentinel/evals/v1\n")
    config = SentinelConfig()
    assert resolve_eval_suite_path("demo", config=config) == tmp_path / "evals" / "demo" / "cases.yaml"
    assert resolve_eval_suite_path("evals/demo", config=config) == Path("evals/demo/cases.yaml")
    assert resolve_eval_suite_path(cases, config=config) == cases
    with pytest.raises(ConfigError, match="eval suite 'other' not found"):
        resolve_eval_suite_path("other", config=config)
    (tmp_path / "empty_suite").mkdir()
    with pytest.raises(ConfigError, match=r"has no cases\.yaml"):
        resolve_eval_suite_path("./empty_suite", config=config)
