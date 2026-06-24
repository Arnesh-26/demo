# =====================================================================
# BACKGROUND DIARIZATION WORKER (app/services/diarization_worker.py)
# =====================================================================
# Purpose: Periodically runs Speaker Diarization on the sliding audio window,
#          performs greedy overlap matching, and updates SpeakerTimeline.
# =====================================================================

import asyncio
import time
import threading
import collections
from typing import Dict, List, Optional
import numpy as np
from app.core.config import settings
from app.utils.logging import logger
from app.services.speaker_timeline import speaker_timeline_manager

# ---------------------------------------------------------------------------
# Lazy imports for heavy production dependencies (torch / pyannote.audio).
# They are initialised once and shared across the entire process lifetime.
# ---------------------------------------------------------------------------
_torch = None
_Pipeline = None


def _lazy_init_pyannote() -> None:
    """Import torch and pyannote.audio exactly once per process."""
    global _torch, _Pipeline
    if _torch is None or _Pipeline is None:
        try:
            import torch as t
            from pyannote.audio import Pipeline as P
            _torch = t
            _Pipeline = P
        except ImportError as exc:
            logger.error(
                "DiarizationWorker: Failed to import torch/pyannote.audio",
                exc_info=True,
            )
            raise ImportError(
                "Failed to import 'torch' or 'pyannote.audio'. "
                "Ensure they are installed before using production diarization mode."
            ) from exc


# ---------------------------------------------------------------------------
# PCM → Tensor conversion (no temp files, no WAV headers)
# ---------------------------------------------------------------------------

def _pcm_bytes_to_tensor(pcm_bytes: bytes):
    """
    Converts raw PCM-16 mono bytes directly to a normalised float32 torch tensor.

    Returns a dict compatible with pyannote.audio Pipeline:
        {"waveform": Tensor[1, samples], "sample_rate": int}
    """
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    audio_np /= 32768.0                          # Normalise int16 → [-1.0, 1.0]
    waveform = _torch.from_numpy(audio_np).unsqueeze(0)  # Shape: [1, samples]
    return {"waveform": waveform, "sample_rate": settings.AUDIO_SAMPLE_RATE}


# ---------------------------------------------------------------------------
# Mock segments (development / offline mode only)
# ---------------------------------------------------------------------------

def _get_mock_speaker(timestamp: float) -> str:
    cycle = timestamp % 30.0
    if cycle < 5.0:
        return "Speaker_0"
    elif cycle < 12.0:
        return "Speaker_1"
    elif cycle < 18.0:
        return "Speaker_0"
    elif cycle < 22.0:
        return "Speaker_2"
    else:
        return "Speaker_0"


def _get_mock_segments(window_start: float, window_end: float) -> list:
    segments = []
    step = 0.1
    current_speaker = None
    segment_start = None

    t = window_start
    while t < window_end + 0.05:
        current_t = min(t, window_end)
        speaker = _get_mock_speaker(current_t)

        if current_speaker is None:
            current_speaker = speaker
            segment_start = current_t
        elif speaker != current_speaker:
            segments.append({"start": segment_start, "end": current_t, "label": current_speaker})
            current_speaker = speaker
            segment_start = current_t

        t += step

    if current_speaker is not None and segment_start < window_end:
        segments.append({"start": segment_start, "end": window_end, "label": current_speaker})

    return segments


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

class DiarizationSessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        # Sliding audio buffer — individual chunk bytes kept in insertion order.
        # Older bytes are evicted in feed_audio() to respect DIARIZATION_BUFFER_SECONDS.
        self.chunks_deque: collections.deque = collections.deque()
        self.total_bytes_in_chunks: int = 0
        self.total_bytes_received: int = 0
        self.lock = asyncio.Lock()
        self.task: Optional[asyncio.Task] = None
        self.is_active: bool = True
        self.prev_window_start: float = 0.0
        self.prev_window_end: float = 0.0
        self.has_new_audio: bool = False


# ---------------------------------------------------------------------------
# Worker manager
# ---------------------------------------------------------------------------

class DiarizationWorkerManager:
    """
    Manages active diarization sessions and owns the single shared Pyannote pipeline.

    Locking Hierarchy & Concurrency Model:
    1. Manager Lock (self._lock - threading.Lock):
       - Purpose: Protects CPU-bound, synchronous state-dict operations.
       - Constraints: Never held across `await` points.  Released immediately.
    2. Session Lock (state.lock - asyncio.Lock):
       - Purpose: Prevents two concurrent inference cycles for the same session.
       - Constraints: Held across `asyncio.to_thread` calls during pipeline execution.
    """

    def __init__(self):
        self._states: Dict[str, DiarizationSessionState] = {}
        self._lock = threading.Lock()
        self._pipeline = None  # Loaded once; shared across all sessions

    # ------------------------------------------------------------------
    # Public startup helper
    # ------------------------------------------------------------------

    async def preload_pipeline(self) -> None:
        """
        Eagerly loads the Pyannote pipeline at server startup so the first
        diarization window does not incur a cold-start penalty.

        Must be awaited directly from the FastAPI `startup` event handler
        (NOT wrapped in asyncio.to_thread) so that pyannote.audio's import-time
        asyncio code runs inside an active event loop.

        Raises:
            ValueError: If CLINICAL_COPILOT_HUGGINGFACE_TOKEN is not set.
            ImportError: If torch / pyannote.audio are not installed.
        """
        if not settings.HUGGINGFACE_TOKEN or not settings.HUGGINGFACE_TOKEN.strip():
            raise ValueError(
                "CLINICAL_COPILOT_HUGGINGFACE_TOKEN must be set when "
                "DIARIZATION_MODE is 'production'."
            )

        # Phase 1: Import pyannote HERE (in the async context which has a running
        # event loop). pyannote.audio's __init__ triggers asyncio code at import
        # time, which crashes if run inside asyncio.to_thread (no event loop there).
        _lazy_init_pyannote()

        if self._pipeline is not None:
            logger.info("DiarizationWorker: Pipeline already loaded, skipping preload.")
            return

        logger.info("DiarizationWorker: Loading Pyannote pipeline (this may take a moment)…")

        # Phase 2: Model download / weight loading is purely blocking I/O + CPU.
        # Offload to a thread so the event loop stays responsive during the download.
        self._pipeline = await asyncio.to_thread(self._load_model_from_pretrained)

        if _torch.cuda.is_available():
            self._pipeline.to(_torch.device("cuda"))
            logger.info("DiarizationWorker: Pipeline loaded on CUDA.")
        else:
            logger.info("DiarizationWorker: Pipeline loaded on CPU (CUDA not available).")

    def _load_model_from_pretrained(self):
        """Blocking helper — runs in a thread pool via asyncio.to_thread."""
        return _Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.HUGGINGFACE_TOKEN,
        )

    # ------------------------------------------------------------------
    # Audio ingestion
    # ------------------------------------------------------------------

    def feed_audio(self, session_id: str, chunk: bytes) -> None:
        """Appends an audio chunk and (re-)starts the background processing loop."""
        with self._lock:
            if session_id not in self._states:
                logger.info(
                    "DiarizationWorker: Creating new session state",
                    extra={"session_id": session_id},
                )
                self._states[session_id] = DiarizationSessionState(session_id)
            state = self._states[session_id]

        state.chunks_deque.append(chunk)
        state.total_bytes_in_chunks += len(chunk)
        state.total_bytes_received += len(chunk)
        state.has_new_audio = True

        # Bounded sliding buffer — evict chunks older than DIARIZATION_BUFFER_SECONDS
        bytes_per_second = (
            settings.AUDIO_SAMPLE_RATE * settings.AUDIO_CHANNELS * settings.bytes_per_sample
        )
        max_bytes = settings.DIARIZATION_BUFFER_SECONDS * bytes_per_second
        while state.total_bytes_in_chunks > max_bytes:
            oldest = state.chunks_deque.popleft()
            state.total_bytes_in_chunks -= len(oldest)

        # Spawn background loop only if not already running
        if state.task is None or state.task.done():
            state.task = asyncio.create_task(self._run_loop(session_id))

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def stop_worker(self, session_id: str) -> None:
        """Signals the background loop to stop, processes the final window, and cleans up."""
        state = None
        with self._lock:
            state = self._states.get(session_id)

        if state:
            logger.info(
                "DiarizationWorker: Stopping worker loop",
                extra={"session_id": session_id},
            )
            state.is_active = False
            if state.task:
                try:
                    await state.task
                except Exception:
                    logger.error(
                        "DiarizationWorker: Error waiting for worker task exit",
                        exc_info=True,
                        extra={"session_id": session_id},
                    )

            try:
                await self._process_latest_window(session_id, force=True)
            except Exception:
                logger.error(
                    "DiarizationWorker: Error during final window diarization",
                    exc_info=True,
                    extra={"session_id": session_id},
                )

    def clear_session(self, session_id: str) -> None:
        """Removes all in-memory state for a session to prevent memory leaks."""
        with self._lock:
            if session_id in self._states:
                del self._states[session_id]
                logger.info(
                    "DiarizationWorker: Session state cleared",
                    extra={"session_id": session_id},
                )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _run_loop(self, session_id: str) -> None:
        """Fires inference every DIARIZATION_INTERVAL_SECONDS while there is new audio."""
        logger.info(
            "DiarizationWorker: Worker loop started",
            extra={"session_id": session_id},
        )

        state = None
        with self._lock:
            state = self._states.get(session_id)

        if not state:
            return

        step_interval = float(settings.DIARIZATION_INTERVAL_SECONDS)
        try:
            while state.is_active:
                await asyncio.sleep(step_interval)
                if state.has_new_audio:
                    state.has_new_audio = False
                    await self._process_latest_window(session_id)
        except asyncio.CancelledError:
            logger.debug(
                "DiarizationWorker: Loop task cancelled",
                extra={"session_id": session_id},
            )
        except Exception:
            logger.error(
                "DiarizationWorker: Exception in worker loop",
                exc_info=True,
                extra={"session_id": session_id},
            )
        finally:
            logger.info(
                "DiarizationWorker: Worker loop stopped",
                extra={"session_id": session_id},
            )

    # ------------------------------------------------------------------
    # Core inference cycle
    # ------------------------------------------------------------------

    async def _process_latest_window(self, session_id: str, force: bool = False) -> None:
        """
        Extracts the latest DIARIZATION_WINDOW_SECONDS of audio, runs Pyannote
        (or mock) inference off-thread, then updates the SpeakerTimeline via
        greedy overlap matching.
        """
        state = None
        with self._lock:
            state = self._states.get(session_id)

        if not state:
            return

        async with state.lock:
            audio_bytes = b"".join(state.chunks_deque)
            total_bytes = state.total_bytes_received

            bytes_per_second = (
                settings.AUDIO_SAMPLE_RATE * settings.AUDIO_CHANNELS * settings.bytes_per_sample
            )
            total_seconds = total_bytes / float(bytes_per_second)

            window_size = float(settings.DIARIZATION_WINDOW_SECONDS)
            window_start = max(0.0, total_seconds - window_size)
            window_end = total_seconds

            if not force and (window_end - window_start < 1.0):
                return

            if not force and (window_end - state.prev_window_end < 0.5):
                return

            # Map absolute window times to byte offsets in the local buffer
            audio_data_start_time = max(
                0.0,
                (state.total_bytes_received - len(audio_bytes)) / float(bytes_per_second),
            )
            start_offset_sec = window_start - audio_data_start_time
            end_offset_sec = window_end - audio_data_start_time

            start_byte = int(start_offset_sec * bytes_per_second)
            end_byte = int(end_offset_sec * bytes_per_second)

            start_byte = max(0, min(start_byte, len(audio_bytes)))
            end_byte = max(0, min(end_byte, len(audio_bytes)))

            # Align to 2-byte sample boundaries (16-bit PCM)
            start_byte = (start_byte // 2) * 2
            end_byte = (end_byte // 2) * 2

            window_audio = audio_bytes[start_byte:end_byte]

            # ----------------------------------------------------------------
            # Run inference
            # ----------------------------------------------------------------
            try:
                use_production = (
                    settings.DIARIZATION_MODE == "production"
                    and bool(settings.HUGGINGFACE_TOKEN)
                )
                if use_production:
                    local_segments = await asyncio.to_thread(
                        self._run_pyannote_pipeline, session_id, window_audio, window_start
                    )
                else:
                    local_segments = await asyncio.to_thread(
                        _get_mock_segments, window_start, window_end
                    )
            except Exception:
                logger.error(
                    "DiarizationWorker: Diarization execution failed",
                    exc_info=True,
                    extra={"session_id": session_id},
                )
                return  # Keep the worker alive; retry on the next interval

            # ----------------------------------------------------------------
            # Greedy conflict-free bipartite overlap matching
            # ----------------------------------------------------------------
            timeline = speaker_timeline_manager.get_timeline(session_id)
            global_timeline = timeline.segments

            overlap_start = window_start
            overlap_end = state.prev_window_end

            unique_local = list(set(seg["label"] for seg in local_segments))
            mapping: Dict[str, str] = {}

            if overlap_start < overlap_end and global_timeline:
                triples = []
                for loc in unique_local:
                    for glob_seg in global_timeline:
                        for loc_seg in local_segments:
                            if loc_seg["label"] != loc:
                                continue
                            int_start = max(loc_seg["start"], glob_seg["start"], overlap_start)
                            int_end = min(loc_seg["end"], glob_seg["end"], overlap_end)
                            duration = int_end - int_start
                            if duration > 0.05:
                                triples.append((loc, glob_seg["label"], duration))

                # Aggregate and sort descending
                aggregated: Dict = {}
                for loc, glob, dur in triples:
                    key = (loc, glob)
                    aggregated[key] = aggregated.get(key, 0.0) + dur

                sorted_pairs = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

                # Greedy assignment
                claimed_global: set = set()
                for (loc, glob), _ in sorted_pairs:
                    if loc not in mapping and glob not in claimed_global:
                        mapping[loc] = glob
                        claimed_global.add(glob)

            # Map unmapped local labels to new/unused global slots
            allowed_labels = ["Speaker_0", "Speaker_1", "Speaker_2"]
            allocated_labels = set(mapping.values())
            timeline_labels = set(seg["label"] for seg in global_timeline)
            used_labels = allocated_labels.union(timeline_labels)

            for loc in unique_local:
                if loc not in mapping:
                    unused_allowed = [lbl for lbl in allowed_labels if lbl not in used_labels]
                    if unused_allowed:
                        mapped_label = unused_allowed[0]
                        mapping[loc] = mapped_label
                        used_labels.add(mapped_label)
                    else:
                        # Hard speaker-cap fallback
                        mapped_label = self._map_unmapped_to_existing(
                            loc, local_segments, global_timeline
                        )
                        mapping[loc] = mapped_label
                        logger.warning(
                            "DiarizationWorker: Exceeded maximum speaker cap",
                            extra={
                                "session_id": session_id,
                                "window_start": window_start,
                                "window_end": window_end,
                                "local_label": loc,
                                "mapped_to": mapped_label,
                            },
                        )

            # Apply mapping and push to global timeline
            mapped_segments = [
                {"start": seg["start"], "end": seg["end"], "label": mapping[seg["label"]]}
                for seg in local_segments
            ]

            timeline.update_timeline(mapped_segments, window_start)

            state.prev_window_start = window_start
            state.prev_window_end = window_end

            # Retroactively enrich any cached "UNKNOWN" events in the TranscriptEventBus
            try:
                from app.services.speaker_alignment import speaker_alignment_service
                speaker_alignment_service.enrich_cached_events(session_id)
            except Exception:
                logger.error(
                    "DiarizationWorker: Error triggering event enrichment",
                    exc_info=True,
                    extra={"session_id": session_id},
                )

    # ------------------------------------------------------------------
    # Pyannote inference (runs in a thread-pool via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _run_pyannote_pipeline(
        self, session_id: str, audio_bytes: bytes, window_start: float
    ) -> list:
        """
        Converts raw PCM bytes to a normalised float32 tensor and runs the
        Pyannote diarization pipeline.  No temporary WAV files are created.

        Returns a list of dicts:
            [{"start": float, "end": float, "label": str}, ...]
        where times are expressed in absolute session seconds (window_start + local_offset).
        """
        # Pipeline must be pre-loaded by preload_pipeline() at startup (async context).
        # _lazy_init_pyannote() must NOT be called here because this method runs inside
        # asyncio.to_thread — that worker thread has no event loop, and pyannote.audio's
        # import-time asyncio code would crash with "RuntimeError: no running event loop".
        if self._pipeline is None:
            logger.error(
                "DiarizationWorker: Pipeline not loaded — skipping inference window. "
                "Ensure preload_pipeline() was awaited at startup.",
                extra={"session_id": session_id},
            )
            return []

        if not audio_bytes:
            return []

        # Convert PCM bytes → normalised float32 tensor (no WAV intermediary)
        audio_input = _pcm_bytes_to_tensor(audio_bytes)

        t0 = time.perf_counter()
        annotation = self._pipeline(audio_input)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        local_segments = []
        for segment, _track, label in annotation.itertracks(yield_label=True):
            local_segments.append({
                "start": window_start + segment.start,
                "end": window_start + segment.end,
                "label": label,   # e.g. "SPEAKER_00", "SPEAKER_01" – mapped by greedy matcher
            })

        speaker_count = len(set(seg["label"] for seg in local_segments))
        logger.info(
            "DiarizationWorker: Diarization completed",
            extra={
                "session_id": session_id,
                "speaker_count": speaker_count,
                "window_seconds": settings.DIARIZATION_WINDOW_SECONDS,
                "inference_time_ms": round(inference_ms, 1),
            },
        )

        return local_segments

    # ------------------------------------------------------------------
    # Hard-cap fallback helper
    # ------------------------------------------------------------------

    def _map_unmapped_to_existing(
        self, loc: str, local_segments: list, global_timeline: list
    ) -> str:
        """
        Fallback matching when all three global speaker slots are occupied.
        Priority:
        1. Greatest temporal overlap with an existing global segment.
        2. Temporal proximity (closest in time).
        3. Continuity (the global speaker whose segment ended most recently
           before the current local segment starts).
        """
        allowed = ["Speaker_0", "Speaker_1", "Speaker_2"]

        # 1. Overlap duration
        overlaps = {lbl: 0.0 for lbl in allowed}
        for loc_seg in local_segments:
            if loc_seg["label"] != loc:
                continue
            for glob_seg in global_timeline:
                if glob_seg["label"] not in allowed:
                    continue
                int_start = max(loc_seg["start"], glob_seg["start"])
                int_end = min(loc_seg["end"], glob_seg["end"])
                dur = int_end - int_start
                if dur > 0:
                    overlaps[glob_seg["label"]] += dur

        best_overlap_speaker = max(overlaps, key=overlaps.get)
        if overlaps[best_overlap_speaker] > 0.01:
            return best_overlap_speaker

        # 2. Proximity
        proximity = {lbl: float("inf") for lbl in allowed}
        loc_segs_for_lbl = [s for s in local_segments if s["label"] == loc]
        for loc_seg in loc_segs_for_lbl:
            for glob_seg in global_timeline:
                if glob_seg["label"] not in allowed:
                    continue
                dist = min(
                    abs(loc_seg["start"] - glob_seg["end"]),
                    abs(loc_seg["end"] - glob_seg["start"]),
                )
                proximity[glob_seg["label"]] = min(proximity[glob_seg["label"]], dist)

        min_dist = min(proximity.values())
        if min_dist != float("inf"):
            closest_speakers = [lbl for lbl, d in proximity.items() if d == min_dist]
            if len(closest_speakers) == 1:
                return closest_speakers[0]
        else:
            closest_speakers = allowed

        # 3. Continuity
        first_loc_start = min(s["start"] for s in loc_segs_for_lbl) if loc_segs_for_lbl else 0.0
        latest_end_before_start = -1.0
        best_spk = closest_speakers[0]
        for glob_seg in global_timeline:
            if glob_seg["label"] in closest_speakers:
                if (
                    glob_seg["end"] <= first_loc_start
                    and glob_seg["end"] > latest_end_before_start
                ):
                    latest_end_before_start = glob_seg["end"]
                    best_spk = glob_seg["label"]

        return best_spk


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
diarization_worker_manager = DiarizationWorkerManager()
