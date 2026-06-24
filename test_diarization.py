# =====================================================================
# SPEAKER DIARIZATION UNIT TESTS (test_diarization.py)
# =====================================================================
# Purpose: Verifies all priority matching logic, 3-speaker hard cap,
#          continuity tie-breakers, and 4 consultation scenarios.
# =====================================================================

import sys
import unittest
from app.services.speaker_timeline import SpeakerTimeline
from app.services.providers.models import TranscriptEvent

class TestSpeakerDiarization(unittest.TestCase):
    def setUp(self):
        self.timeline = SpeakerTimeline("test-session")

    def test_scenario_1_doctor_only(self):
        """Scenario 1: Doctor only. Expect all segments to match Speaker_0."""
        self.timeline.segments = [
            {"start": 0.0, "end": 10.0, "label": "Speaker_0"},
            {"start": 10.0, "end": 20.0, "label": "Speaker_0"}
        ]
        
        # Word lookup check
        self.assertEqual(self.timeline.get_speaker_for_range(2.0, 3.0), "Speaker_0")
        self.assertEqual(self.timeline.get_speaker_for_range(12.0, 14.0), "Speaker_0")

    def test_scenario_2_doctor_patient_alternating(self):
        """Scenario 2: Doctor -> Patient -> Doctor -> Patient (0 -> 1 -> 0 -> 1)."""
        self.timeline.segments = [
            {"start": 0.0, "end": 5.0, "label": "Speaker_0"},
            {"start": 5.0, "end": 10.0, "label": "Speaker_1"},
            {"start": 10.0, "end": 15.0, "label": "Speaker_0"},
            {"start": 15.0, "end": 20.0, "label": "Speaker_1"}
        ]
        
        self.assertEqual(self.timeline.get_speaker_for_range(2.0, 3.0), "Speaker_0")
        self.assertEqual(self.timeline.get_speaker_for_range(7.0, 8.0), "Speaker_1")
        self.assertEqual(self.timeline.get_speaker_for_range(11.0, 12.0), "Speaker_0")
        self.assertEqual(self.timeline.get_speaker_for_range(16.0, 18.0), "Speaker_1")

    def test_scenario_3_three_speakers(self):
        """Scenario 3: Doctor -> Patient -> Attender -> Doctor -> Attender (0 -> 1 -> 2 -> 0 -> 2)."""
        self.timeline.segments = [
            {"start": 0.0, "end": 5.0, "label": "Speaker_0"},
            {"start": 5.0, "end": 10.0, "label": "Speaker_1"},
            {"start": 10.0, "end": 15.0, "label": "Speaker_2"},
            {"start": 15.0, "end": 20.0, "label": "Speaker_0"},
            {"start": 20.0, "end": 25.0, "label": "Speaker_2"}
        ]
        
        self.assertEqual(self.timeline.get_speaker_for_range(2.0, 4.0), "Speaker_0")
        self.assertEqual(self.timeline.get_speaker_for_range(6.0, 8.0), "Speaker_1")
        self.assertEqual(self.timeline.get_speaker_for_range(12.0, 14.0), "Speaker_2")
        self.assertEqual(self.timeline.get_speaker_for_range(16.0, 18.0), "Speaker_0")
        self.assertEqual(self.timeline.get_speaker_for_range(22.0, 24.0), "Speaker_2")

    def test_scenario_4_interruptions_and_overlaps(self):
        """
        Scenario 4: Interruptions and overlapping speech regions.
        Doctor speaks [0, 5], Patient interrupts at [4, 6] (overlap region),
        Doctor continues [6, 10], Attender joins late at [9, 12].
        """
        self.timeline.segments = [
            {"start": 0.0, "end": 5.0, "label": "Speaker_0"},
            {"start": 4.0, "end": 6.0, "label": "Speaker_1"},
            {"start": 6.0, "end": 10.0, "label": "Speaker_0"},
            {"start": 9.0, "end": 12.0, "label": "Speaker_2"}
        ]
        
        # Word lookup checks
        # Word 1: [1.0, 2.0] -> only overlaps Doctor [0, 5] -> Speaker_0
        self.assertEqual(self.timeline.get_speaker_for_range(1.0, 2.0), "Speaker_0")
        
        # Word 2: [5.2, 5.8] -> inside Patient's segment, overlaps Patient [4, 6]
        self.assertEqual(self.timeline.get_speaker_for_range(5.2, 5.8), "Speaker_1")
        
        # Word 3: [9.5, 10.5] -> overlaps Speaker_0 [6, 10] by 0.5s, Speaker_2 [9, 12] by 1.0s.
        # Speaker_2 has larger overlap -> Speaker_2
        self.assertEqual(self.timeline.get_speaker_for_range(9.5, 10.5), "Speaker_2")

    def test_priority_1_maximum_overlap(self):
        """Priority 1: Maximum overlap duration wins."""
        self.timeline.segments = [
            {"start": 0.0, "end": 4.0, "label": "Speaker_0"},
            {"start": 3.0, "end": 7.0, "label": "Speaker_1"}
        ]
        # Range [2.5, 4.5] overlaps Speaker_0 by 1.5s, Speaker_1 by 1.0s.
        # Speaker_0 wins.
        self.assertEqual(self.timeline.get_speaker_for_range(2.5, 4.5), "Speaker_0")
        
        # Range [3.5, 6.5] overlaps Speaker_0 by 0.5s, Speaker_1 by 2.5s.
        # Speaker_1 wins.
        self.assertEqual(self.timeline.get_speaker_for_range(3.5, 6.5), "Speaker_1")

    def test_priority_2_proximity(self):
        """Priority 2: If overlap is 0, choose speaker closest in time."""
        self.timeline.segments = [
            {"start": 0.0, "end": 2.0, "label": "Speaker_0"},
            {"start": 6.0, "end": 8.0, "label": "Speaker_1"}
        ]
        # Range [3.0, 4.0] is at distance 1.0 from Speaker_0 (3 - 2), and distance 2.0 from Speaker_1 (6 - 4).
        # Speaker_0 is closer -> Speaker_0.
        self.assertEqual(self.timeline.get_speaker_for_range(3.0, 4.0), "Speaker_0")
        
        # Range [4.5, 5.5] is at distance 2.5 from Speaker_0 (4.5 - 2), and distance 0.5 from Speaker_1 (6 - 5.5).
        # Speaker_1 is closer -> Speaker_1.
        self.assertEqual(self.timeline.get_speaker_for_range(4.5, 5.5), "Speaker_1")

    def test_priority_3_continuity_tie_breaker(self):
        """Priority 3: Continuity tie-breaker (ended most recently before current start)."""
        self.timeline.segments = [
            {"start": 0.0, "end": 2.0, "label": "Speaker_0"}, # ended at 2.0
            {"start": 4.0, "end": 6.0, "label": "Speaker_1"}  # starts at 4.0
        ]
        # Range [3.0, 3.0] (point in time) is at distance 1.0 from Speaker_0 (3.0 - 2.0 = 1.0),
        # and distance 1.0 from Speaker_1 (4.0 - 3.0 = 1.0).
        # Distance is tied!
        # Continuity tie-breaker: Choose speaker whose segment ended most recently before 3.0.
        # Speaker_0 ended at 2.0 (<= 3.0).
        # Speaker_1 ended at 6.0 (not <= 3.0).
        # So Speaker_0 wins!
        self.assertEqual(self.timeline.get_speaker_for_range(3.0, 3.0), "Speaker_0")

    def test_priority_4_hard_cap(self):
        """Priority 4: Speaker cap logic ensures only Speaker_0, Speaker_1, or Speaker_2 are returned."""
        self.timeline.segments = [
            {"start": 0.0, "end": 5.0, "label": "Speaker_0"},
            {"start": 5.0, "end": 10.0, "label": "Speaker_3"}, # invalid label
            {"start": 10.0, "end": 15.0, "label": "Speaker_1"}
        ]
        # Query in Speaker_3 range [6.0, 7.0] -> Speaker_3 is ignored.
        # Proximity maps to Speaker_0 (distance 1.0: 6 - 5) or Speaker_1 (distance 3.0: 10 - 7).
        # Speaker_0 is closer -> Speaker_0.
        self.assertEqual(self.timeline.get_speaker_for_range(6.0, 7.0), "Speaker_0")

    def test_event_enrichment_versioning(self):
        """Verify that retroactive event enrichment increments version and sets updated_at."""
        from app.services.speaker_alignment import speaker_alignment_service
        from app.services.transcript_bus import transcript_bus
        from app.services.speaker_timeline import speaker_timeline_manager
        import time

        session_id = "test-session-version"
        # Register timeline in singleton manager
        speaker_timeline_manager._timelines[session_id] = self.timeline
        self.timeline.segments = [
            {"start": 0.0, "end": 5.0, "label": "Speaker_1"}
        ]

        event = TranscriptEvent(
            session_id=session_id,
            role="doctor",
            sequence_number=1,
            timestamp=time.time(),
            transcript="Hello",
            is_partial=False,
            is_final=True,
            event_id="test-uuid-123",
            speaker_id="UNKNOWN",
            speaker_label="UNKNOWN",
            start_time=1.0,
            end_time=3.0,
            text="Hello"
        )
        
        # Publish to cache in the bus
        transcript_bus.publish(event)
        
        # Initial checks
        self.assertEqual(event.version, 1)
        self.assertIsNone(event.updated_at)
        
        # Trigger enrichment
        speaker_alignment_service.enrich_cached_events(session_id)
        
        # Verify version and timestamp updates
        self.assertEqual(event.speaker_id, "Speaker_1")
        self.assertEqual(event.version, 2)
        self.assertIsNotNone(event.updated_at)


if __name__ == "__main__":
    unittest.main()

