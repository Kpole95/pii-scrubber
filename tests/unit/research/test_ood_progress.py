"""Tests for the OOD progress script."""

from scripts.ood_progress import main


def test_prints_ood_progress(capsys) -> None:
    """The script should print current benchmark progress."""
    main()
    output = capsys.readouterr().out

    assert "200/200 examples" in output
    assert "positive: 176" in output
    assert "negative: 24" in output
    assert "entities: 316" in output
