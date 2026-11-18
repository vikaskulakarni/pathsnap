"""A changed model must not pass the release check silently."""
import pytest

from scripts import verify_release


def test_verifier_rejects_an_incorrect_digest(monkeypatch):
    monkeypatch.delenv("PATHSNAP_MODEL_RELEASE", raising=False)
    monkeypatch.setattr(verify_release, "fingerprint", lambda directory: {"model.joblib": "wrong"})
    with pytest.raises(ValueError, match="differs"):
        verify_release.validate(runtime_only=True)


@pytest.mark.parametrize("release", ["demonstrated", "all80"])
def test_recorded_model_matches_rehearsal(monkeypatch, release):
    monkeypatch.setenv("PATHSNAP_MODEL_RELEASE", release)
    verify_release.validate()
