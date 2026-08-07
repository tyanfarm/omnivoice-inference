from __future__ import annotations

import re

import numpy as np
import pytest


class FakeOmniVoice:
    """Stand-in for OmniVoice that records how it was called.

    generate() returns, for each input text containing a "job-<n>" token, an
    array filled with float(n). That lets tests assert a result reached the job
    that asked for it, which is the failure mode that would swap audio between
    users. The token may appear anywhere in the text, not just as the whole of
    it — streamed text arrives packed into chunks of several sentences.
    """

    sampling_rate = 24000

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail_texts: set[str] = set()
        self.prompt_calls: list[tuple[str, str]] = []

    def create_voice_clone_prompt(self, ref_audio, ref_text, preprocess_prompt=True):
        self.prompt_calls.append((ref_audio, ref_text))
        return f"prompt:{ref_audio}"

    def generate(self, text, **kwargs):
        self.calls.append({"text": list(text), **kwargs})
        failing = [t for t in text if t in self.fail_texts]
        if failing:
            raise RuntimeError(f"generate failed for {failing}")
        return [np.full(4, self._job_id(t), dtype=np.float32) for t in text]

    @staticmethod
    def _job_id(text: str) -> float:
        """First job-<n> token in the text.

        Streamed text arrives already packed into chunks of several sentences,
        so the id has to be found rather than assumed to be the whole string.
        """
        match = re.search(r"job-(\d+)", text)
        if match is None:
            raise AssertionError(
                f"FakeOmniVoice needs a 'job-<n>' token in the text, got {text!r}"
            )
        return float(match.group(1))


@pytest.fixture
def fake_model() -> FakeOmniVoice:
    return FakeOmniVoice()


@pytest.fixture
def client():
    """API client with the model replaced by a stub, so no GPU is needed.

    max_streams=1 so admission-control tests can exhaust the cap with a single
    held slot. The warmup guard in the service (`if sampling_rate > 0: return`)
    is what stops the app's startup event from replacing this stub scheduler.
    """
    from fastapi.testclient import TestClient

    import batch_scheduler
    import streaming_api_omnivoice as api
    from admission import AdmissionControl

    fake = FakeOmniVoice()
    api.service.scheduler = batch_scheduler.BatchScheduler(model_factory=lambda: fake)
    api.service.scheduler.start()
    assert api.service.scheduler.wait_ready(timeout=5)
    api.service.admission = AdmissionControl(max_streams=1)

    api.turn_service._session = object()
    api.turn_service.predict = lambda audio: {"prediction": 1, "probability": 0.91}

    with TestClient(api.app) as test_client:
        test_client.fake_model = fake
        test_client.voices_dir = api.VOICES_DIR
        yield test_client
    api.service.scheduler.stop()
