from __future__ import annotations

import json
from pathlib import Path

import pytest

import biasweave.dco as dco_module
from biasweave.dco import DCOError, main, verify_commit_file, verify_commit_pages

SHA1 = "1" * 40
SHA2 = "2" * 40


def commit(
    sha: str = SHA1,
    *,
    name: str = "Ada Lovelace",
    email: str = "ada@example.test",
    message: str | None = None,
) -> dict[str, object]:
    return {
        "sha": sha,
        "commit": {
            "author": {"name": name, "email": email},
            "message": message
            or f"Implement deterministic search\n\nSigned-off-by: {name} <{email}>",
        },
    }


def test_dco_accepts_every_author_matching_final_trailer() -> None:
    pages = [[commit()], [commit(SHA2, name="Renée", email="R@Example.Test")]]
    assert verify_commit_pages(pages, expected_count=2, expected_head=SHA2) == 2


@pytest.mark.parametrize(
    "message",
    [
        "Signed-off-by: Ada Lovelace <ada@example.test>\n\nExplanation after trailer",
        "Change\n\nSigned-off-by: Other Person <ada@example.test>",
        "Change\n\nSigned-off-by: Ada Lovelace <other@example.test>",
        "Change\n\nSigned-off-by: Ada Lovelace <ada@example.test> trailing",
        "Change\n\nSigned-off-by: Ada Lovelace <ada@example.test>\nnot a trailer",
        "Change\n\nnot a trailer\nSigned-off-by: Ada Lovelace <ada@example.test>",
    ],
)
def test_dco_rejects_deceptive_or_mismatched_trailers(message: str) -> None:
    with pytest.raises(DCOError, match="DCO sign-off"):
        verify_commit_pages([[commit(message=message)]], expected_count=1, expected_head=SHA1)


def test_dco_binds_count_head_order_and_unique_commits() -> None:
    pages = [[commit(), commit(SHA2)]]
    with pytest.raises(DCOError, match="declares"):
        verify_commit_pages(pages, expected_count=1, expected_head=SHA2)
    with pytest.raises(DCOError, match="immutable pull-request head"):
        verify_commit_pages(pages, expected_count=2, expected_head=SHA1)
    with pytest.raises(DCOError, match="duplicate commit"):
        verify_commit_pages([[commit(), commit()]], expected_count=2, expected_head=SHA1)


def test_dco_accepts_matching_signoff_in_a_strict_final_trailer_block() -> None:
    message = (
        "Change\n\n"
        "Co-authored-by: Grace Hopper <grace@example.test>\n"
        "Signed-off-by: Ada Lovelace <ada@example.test>"
    )
    assert verify_commit_pages([[commit(message=message)]], expected_count=1) == 1


@pytest.mark.parametrize("pages", [None, [], [{}], [[1]], [[]]])
def test_dco_rejects_malformed_page_and_commit_shapes(pages) -> None:
    with pytest.raises(DCOError, match="invalid|malformed|no commits"):
        verify_commit_pages(pages)


def test_dco_rejects_commit_resource_and_identity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dco_module, "_MAX_COMMITS", 1)
    with pytest.raises(DCOError, match="commit verification limit"):
        verify_commit_pages([[commit(), commit(SHA2)]])
    monkeypatch.setattr(dco_module, "_MAX_COMMITS", 250)
    for malformed, message in (
        ({"sha": SHA1}, "malformed commit record"),
        ({"sha": SHA1, "commit": {"author": None, "message": "x"}}, "commit author"),
        (commit("bad"), "invalid commit SHA"),
        (commit(name="Ada\x1fLovelace"), "invalid commit author"),
    ):
        with pytest.raises(DCOError, match=message):
            verify_commit_pages([[malformed]])


def test_dco_rejects_invalid_text_fields() -> None:
    malformed = commit()
    malformed["commit"]["message"] = ""
    with pytest.raises(DCOError, match="commit message"):
        verify_commit_pages([[malformed]])
    malformed = commit()
    malformed["commit"]["author"]["email"] = None
    with pytest.raises(DCOError, match="author email"):
        verify_commit_pages([[malformed]])
    malformed = commit(message="Change\n\nSigned-off-by: Ada \ud800 <ada@example.test>")
    with pytest.raises(DCOError, match="commit message"):
        verify_commit_pages([[malformed]])
    with pytest.raises(DCOError, match="expected.*count"):
        verify_commit_pages([[commit()]], expected_count=True)


def test_dco_file_is_strict_bounded_json(tmp_path: Path) -> None:
    source = tmp_path / "commits.json"
    source.write_text(json.dumps([[commit()]]), encoding="utf-8")
    assert verify_commit_file(source, expected_count=1, expected_head=SHA1) == 1
    source.write_text('[{"x": 1, "x": 2}]', encoding="utf-8")
    with pytest.raises(DCOError, match="duplicate key"):
        verify_commit_file(source, expected_count=1, expected_head=SHA1)
    source.unlink()
    with pytest.raises(DCOError, match="regular file"):
        verify_commit_file(source, expected_count=1, expected_head=SHA1)


def test_dco_cli_reports_success_and_bad_anchors(tmp_path: Path, capsys) -> None:
    source = tmp_path / "commits.json"
    source.write_text(json.dumps([[commit()]]), encoding="utf-8")
    assert main([str(source), "1", SHA1]) == 0
    assert "verified" in capsys.readouterr().out
    assert main([str(source), "0", SHA1]) == 1
    assert "count" in capsys.readouterr().err
    assert main([str(source), "1", SHA2]) == 1
    assert "immutable pull-request head" in capsys.readouterr().err
    assert main([]) == 2
