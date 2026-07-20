"""Canonical tools 的仓库路径解耦合同测试。"""

from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def test_tool_examples_use_repo_relative_esd_path():
    for relative_path in (
        "tools/build_fedd_part_b_v2.py",
        "tools/inference_emo_film.py",
        "tools/run_inference_parallel.py",
    ):
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "datasets/ESD" in content
        assert "/home/hanlvyuan" not in content


def test_mfa_bin_resolution_prefers_cli_then_env_then_path(monkeypatch):
    from tools import run_mfa_align

    monkeypatch.setenv("MFA_BIN", "/env/mfa")
    monkeypatch.setattr(run_mfa_align.shutil, "which", lambda name: "/path/mfa")

    assert run_mfa_align.resolve_mfa_bin("/cli/mfa") == "/cli/mfa"
    assert run_mfa_align.resolve_mfa_bin() == "/env/mfa"

    monkeypatch.delenv("MFA_BIN")
    assert run_mfa_align.resolve_mfa_bin() == "/path/mfa"


def test_mfa_bin_resolution_reports_all_supported_configuration(monkeypatch):
    from tools import run_mfa_align

    monkeypatch.delenv("MFA_BIN", raising=False)
    monkeypatch.setattr(run_mfa_align.shutil, "which", lambda name: None)

    with pytest.raises(FileNotFoundError, match=r"--mfa_bin.*MFA_BIN.*PATH"):
        run_mfa_align.resolve_mfa_bin()


def test_mfa_subprocess_env_prepends_executable_parent_and_preserves_path(monkeypatch):
    from tools import run_mfa_align

    current_path = "/usr/local/bin:/usr/bin"
    monkeypatch.setenv("PATH", current_path)

    env = run_mfa_align.build_subprocess_env("/opt/mfa/bin/mfa")

    assert env["PATH"] == f"/opt/mfa/bin:{current_path}"


def test_download_iemocap_parser_defaults_to_repo_dataset_without_downloading():
    from tools import download_iemocap

    args = download_iemocap.build_parser().parse_args([])

    assert args.output_dir == ROOT / "datasets/IEMOCAP"


def test_download_iemocap_parser_accepts_output_dir_without_downloading(tmp_path):
    from tools import download_iemocap

    args = download_iemocap.build_parser().parse_args(
        ["--output-dir", str(tmp_path)]
    )

    assert args.output_dir == tmp_path
