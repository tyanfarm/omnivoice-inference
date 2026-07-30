from __future__ import annotations

import threading

from admission import AdmissionControl


def test_acquires_up_to_the_cap():
    control = AdmissionControl(max_streams=2)
    assert control.try_acquire() is not None
    assert control.try_acquire() is not None
    assert control.try_acquire() is None


def test_releasing_frees_a_slot():
    control = AdmissionControl(max_streams=1)
    slot = control.try_acquire()
    assert control.try_acquire() is None
    slot.release()
    assert control.try_acquire() is not None


def test_release_is_idempotent():
    control = AdmissionControl(max_streams=1)
    slot = control.try_acquire()
    slot.release()
    slot.release()
    slot.release()
    assert control.active == 0
    assert control.try_acquire() is not None


def test_active_count_tracks_outstanding_slots():
    control = AdmissionControl(max_streams=4)
    slots = [control.try_acquire() for _ in range(3)]
    assert control.active == 3
    slots[0].release()
    assert control.active == 2


def test_concurrent_acquire_never_exceeds_the_cap():
    control = AdmissionControl(max_streams=5)
    acquired = []
    lock = threading.Lock()

    def worker():
        slot = control.try_acquire()
        if slot is not None:
            with lock:
                acquired.append(slot)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(acquired) == 5


def test_returns_503_when_no_slot_is_available(client):
    import streaming_api_omnivoice as api

    held = api.service.admission.try_acquire()
    assert held is not None

    response = client.post(
        "/api/stream-mp3", json={"text": "job-1", "voice_id": "af_heart"}
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"

    held.release()


def test_slot_is_returned_after_a_stream_completes(client):
    import streaming_api_omnivoice as api

    response = client.post(
        "/api/stream-mp3", json={"text": "job-1", "voice_id": "af_heart"}
    )
    assert response.status_code == 200
    _ = response.content
    assert api.service.admission.active == 0


def test_unknown_voice_leaks_no_slot(client):
    import streaming_api_omnivoice as api

    response = client.post(
        "/api/stream-mp3", json={"text": "job-1", "voice_id": "nope"}
    )
    assert response.status_code in (404, 500)
    assert api.service.admission.active == 0
