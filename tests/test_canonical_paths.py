"""Canonical candidate 的活跃脚本与测试不得依赖旧用户目录。"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOTS = (
    "conf",
    "cosyvoice",
    "cosyvoice_emo",
    "eval",
    "scripts",
    "tests",
    "tools",
)
ACTIVE_FILES = ("README.md", "CONTEXT.md")
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".yaml", ".yml"}
FORBIDDEN_PATH_FRAGMENTS = (
    "/home/" + "hanlvyuan/",
    ".config/superpowers/" + "worktrees/",
)


def iter_active_paths():
    for relative_path in ACTIVE_FILES:
        yield REPO_ROOT / relative_path
    for relative_root in ACTIVE_ROOTS:
        for path in (REPO_ROOT / relative_root).rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                yield path
    yield from (REPO_ROOT / "docs").rglob("*.md")


def test_active_paths_do_not_reference_old_checkout_or_user_home():
    violations = []
    for path in iter_active_paths():
        relative_path = path.relative_to(REPO_ROOT)
        source = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), start=1):
            for fragment in FORBIDDEN_PATH_FRAGMENTS:
                if fragment in line:
                    violations.append(f"{relative_path}:{line_number}: {fragment}")

    assert not violations, "active path coupling found:\n" + "\n".join(violations)


def test_path_scan_covers_all_active_sources():
    scanned = {str(path.relative_to(REPO_ROOT)) for path in iter_active_paths()}
    assert "tools/download_iemocap.py" in scanned
    assert "tools/inference_emo_film.py" in scanned
    assert "README.md" in scanned
    assert "CONTEXT.md" in scanned
    assert "docs/adr/0018-rebuild-canonical-checkout-from-emofilm-v1.md" in scanned
