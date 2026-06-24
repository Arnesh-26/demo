# =====================================================================
# SPEAKER ALIGNMENT LAYER (app/services/speaker_alignment.py)
# =====================================================================
# Purpose: Maps transcripts/timestamps to speakers and retroactively
#          enriches cached events in-place.
# =====================================================================

import uuid
import time
from typing import List
from app.services.providers.models import TranscriptEvent
from app.services.speaker_timeline import speaker_timeline_manager
from app.utils.logging import logger

class SpeakerAlignmentService:
    def align_and_segment(self, session_id: str, role: str, res: dict, provider) -> List[TranscriptEvent]:
        """
        Maps ElevenLabs Scribe transcript output words to the SpeakerTimeline.
        Groups contiguous speaker segments, assigns monotonic sequence numbers,
        and generates a list of TranscriptEvents.
        """
        from app.services.session import session_manager
        
        session = session_manager.get_session(session_id)
        timeline = speaker_timeline_manager.get_timeline(session_id)
        
        is_partial = (res.get("type") == "partial")
        text = res.get("text", "")
        confidence = res.get("confidence")
        words = res.get("words", [])
        
        # 1. Partial/Empty words fallback
        if is_partial or not words:
            # Estimate current audio time based on total bytes processed
            current_time = 0.0
            if session:
                stream = session.streams.get(role)
                if stream:
                    current_time = stream.buffer.total_bytes_received / 32000.0
                else:
                    # Fallback to diarization worker accumulated audio size
                    from app.services.diarization_worker import diarization_worker_manager
                    state = diarization_worker_manager._states.get(session_id)
                    current_time = (state.total_bytes_received / 32000.0) if state else 0.0

                
            # Lookup speaker for the recent 3 seconds range
            start_time = max(0.0, current_time - 3.0)
            end_time = current_time
            speaker = timeline.get_speaker_for_range(start_time, end_time)
            
            seq = session.get_and_increment_transcript_seq(role) if session else provider.get_and_increment_event_seq()
            
            event = TranscriptEvent(
                session_id=session_id,
                role=role,
                sequence_number=seq,
                timestamp=time.time(),
                transcript=text,
                is_partial=is_partial,
                is_final=not is_partial,
                confidence=confidence,
                provider="scribe_v2",
                event_id=str(uuid.uuid4()),
                speaker_id=speaker,
                speaker_label=speaker,
                start_time=start_time,
                end_time=end_time,
                text=text
            )
            return [event]
            
        # 2. Map word-level timestamps to speakers
        word_speaker_pairs = []
        for w in words:
            w_text = w.get("text", "")
            w_start = w.get("start", 0.0)
            w_end = w.get("end", 0.0)
            speaker = timeline.get_speaker_for_range(w_start, w_end)
            word_speaker_pairs.append((w_text, w_start, w_end, speaker))
            
        # Group contiguous words belonging to the same speaker
        groups = []
        if word_speaker_pairs:
            curr_speaker = word_speaker_pairs[0][3]
            curr_words = [word_speaker_pairs[0][0]]
            curr_start = word_speaker_pairs[0][1]
            curr_end = word_speaker_pairs[0][2]
            
            for w_text, w_start, w_end, speaker in word_speaker_pairs[1:]:
                if speaker == curr_speaker:
                    curr_words.append(w_text)
                    curr_end = w_end
                else:
                    groups.append((curr_speaker, curr_words, curr_start, curr_end))
                    curr_speaker = speaker
                    curr_words = [w_text]
                    curr_start = w_start
                    curr_end = w_end
            groups.append((curr_speaker, curr_words, curr_start, curr_end))
            
        # 3. Create a TranscriptEvent for each grouped segment
        events = []
        for speaker, g_words, g_start, g_end in groups:
            group_text = " ".join(g_words).strip()
            if not group_text:
                continue
                
            seq = session.get_and_increment_transcript_seq(role) if session else provider.get_and_increment_event_seq()
            
            event = TranscriptEvent(
                session_id=session_id,
                role=role,
                sequence_number=seq,
                timestamp=time.time(),
                transcript=group_text,
                is_partial=False,
                is_final=True,
                confidence=confidence,
                provider="scribe_v2",
                event_id=str(uuid.uuid4()),
                speaker_id=speaker,
                speaker_label=speaker,
                start_time=g_start,
                end_time=g_end,
                text=group_text
            )
            events.append(event)
            
        return events

    def enrich_cached_events(self, session_id: str) -> None:
        """
        Scans the cached events in the TranscriptEventBus and retroactively
        resolves 'UNKNOWN' speaker labels based on the updated SpeakerTimeline.
        Updates the events in-place, preserving event_id and sequence_number.
        """
        from app.services.transcript_bus import transcript_bus
        
        timeline = speaker_timeline_manager.get_timeline(session_id)
        recent_events = transcript_bus.get_recent_events(session_id)
        
        for event in recent_events:
            # Enrich completed events that have an UNKNOWN speaker
            if event.is_final and event.speaker_id == "UNKNOWN":
                if event.start_time is not None and event.end_time is not None:
                    resolved = timeline.get_speaker_for_range(event.start_time, event.end_time)
                    if resolved != "UNKNOWN":
                        event.speaker_id = resolved
                        event.speaker_label = resolved
                        event.version += 1
                        event.updated_at = time.time()
                        logger.info(

                            "SpeakerAlignmentService: Retroactively enriched speaker label",
                            extra={
                                "session_id": session_id,
                                "event_id": event.event_id,
                                "seq": event.sequence_number,
                                "speaker": resolved,
                                "text": event.text
                            }
                        )

# Singleton Instance
speaker_alignment_service = SpeakerAlignmentService()
