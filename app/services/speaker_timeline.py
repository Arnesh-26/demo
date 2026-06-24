# =====================================================================
# SPEAKER TIMELINE TRACKER (app/services/speaker_timeline.py)
# =====================================================================
# Purpose: Manages global speaker timeline segments and provides time-range
#          queries to map transcripts to speakers with fallback logic.
# =====================================================================

import threading
from typing import Dict, List, Optional
from app.utils.logging import logger

class SpeakerTimeline:
    """
    Manages the global sequence of speaker segments for a single session.
    A segment is a dict: {"start": float, "end": float, "label": str}
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.segments: List[Dict[str, any]] = []
        self._lock = threading.Lock()

    def update_timeline(self, new_segments: List[Dict[str, any]], window_start: float) -> None:
        """
        Updates the global timeline with new diarized segments from window_start onwards.
        Replaces any existing timeline segments starting from or after window_start,
        and truncates any segments overlapping with window_start.
        """
        with self._lock:
            updated = []
            
            # Keep and/or truncate existing segments ending before window_start
            for seg in self.segments:
                if seg["end"] <= window_start:
                    updated.append(seg)
                elif seg["start"] < window_start:
                    # Truncate overlapping segment
                    seg_copy = seg.copy()
                    seg_copy["end"] = window_start
                    updated.append(seg_copy)
                    
            # Append the new mapped segments
            for new_seg in new_segments:
                # Ensure we only append allowed global speaker labels (Hard Cap: Max 3)
                if new_seg["label"] not in ["Speaker_0", "Speaker_1", "Speaker_2"]:
                    continue

                if new_seg["end"] <= window_start:
                    continue
                
                adjusted_seg = {
                    "start": max(new_seg["start"], window_start),
                    "end": new_seg["end"],
                    "label": new_seg["label"]
                }
                updated.append(adjusted_seg)
                
            # Merge adjacent segments with identical speaker labels for cleanliness
            merged = []
            if updated:
                updated.sort(key=lambda x: x["start"])
                curr = updated[0]
                for nxt in updated[1:]:
                    if nxt["label"] == curr["label"] and nxt["start"] <= curr["end"] + 0.05:
                        curr["end"] = max(curr["end"], nxt["end"])
                    else:
                        merged.append(curr)
                        curr = nxt
                merged.append(curr)
                
            self.segments = merged
            logger.debug(
                "SpeakerTimeline: Timeline updated",
                extra={"session_id": self.session_id, "segment_count": len(self.segments)}
            )

    def get_speaker_for_range(self, start: float, end: float) -> str:
        """
        Determines the active speaker for a time range [start, end].
        Priority Logic:
        1. Maximum overlap duration: Choose speaker with greatest overlap.
        2. Temporal proximity: If overlap = 0, choose speaker segment closest in time.
        3. Continuity fallback: If proximity ties, choose the speaker whose segment ended
           most recently BEFORE the start of the current range.
        4. Hard cap fallback: Always return one of Speaker_0, Speaker_1, or Speaker_2.
        """
        with self._lock:
            # Filter segments to enforce hard cap safety
            valid_segs = [s for s in self.segments if s["label"] in ["Speaker_0", "Speaker_1", "Speaker_2"]]
            if not valid_segs:
                return "UNKNOWN"

                
            # Priority 1: Overlap duration calculation
            overlaps = {}
            for seg in valid_segs:
                int_start = max(start, seg["start"])
                int_end = min(end, seg["end"])
                duration = int_end - int_start
                if duration > 0.01:
                    overlaps[seg["label"]] = overlaps.get(seg["label"], 0.0) + duration
                    
            if overlaps:
                best_speaker = max(overlaps, key=overlaps.get)
                if overlaps[best_speaker] > 0.01:
                    return best_speaker
                    
            # Priority 2: Temporal proximity calculation (find segment closest to [start, end])
            proximity_scores = {}
            for seg in valid_segs:
                dist = min(abs(start - seg["end"]), abs(end - seg["start"]))
                proximity_scores[seg["label"]] = min(proximity_scores.get(seg["label"], float("inf")), dist)
                
            min_dist = min(proximity_scores.values())
            closest_speakers = [spk for spk, dist in proximity_scores.items() if dist == min_dist]
            
            if len(closest_speakers) == 1:
                return closest_speakers[0]
                
            # Priority 3: Continuity fallback (choose segment that ended most recently before start)
            latest_end_before_start = -1.0
            best_spk = closest_speakers[0]
            for seg in valid_segs:
                if seg["label"] in closest_speakers:
                    if seg["end"] <= start and seg["end"] > latest_end_before_start:
                        latest_end_before_start = seg["end"]
                        best_spk = seg["label"]
            return best_spk


class SpeakerTimelineManager:
    def __init__(self):
        self._timelines: Dict[str, SpeakerTimeline] = {}
        self._lock = threading.Lock()

    def get_timeline(self, session_id: str) -> SpeakerTimeline:
        with self._lock:
            if session_id not in self._timelines:
                self._timelines[session_id] = SpeakerTimeline(session_id)
            return self._timelines[session_id]

    def clear_timeline(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._timelines:
                del self._timelines[session_id]
                logger.info("SpeakerTimelineManager: Cleared timeline", extra={"session_id": session_id})

# Singleton Manager Instance
speaker_timeline_manager = SpeakerTimelineManager()
