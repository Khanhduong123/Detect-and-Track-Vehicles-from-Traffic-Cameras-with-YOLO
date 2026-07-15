# ByteTrack Multi-Object Tracker integration
from typing import Dict, List, Tuple

from backend.app.services.cache import redis_cache
from backend.app.shared.logging import logger


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    def __init__(self, bbox: List[float], score: float, cls_name: str, track_id: int):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = score
        self.cls_name = cls_name
        self.track_id = track_id
        self.state = TrackState.New
        self.history: List[List[float]] = []
        self.raw_history: List[List[float]] = []
        self.start_frame = 0
        self.frame_id = 0

    @property
    def center(self) -> Tuple[float, float]:
        # Coordinate Extraction: Convert to Bottom-Center point (cx, cy)
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = self.bbox[3]
        return (cx, cy)

    def activate(self, frame_id: int):
        self.state = TrackState.Tracked
        self.start_frame = frame_id
        self.frame_id = frame_id
        pt = list(self.center)
        self.raw_history = [pt]
        self.history = [pt]

    def update(self, new_track: "STrack", frame_id: int):
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.cls_name = new_track.cls_name
        self.state = TrackState.Tracked
        self.frame_id = frame_id

        pt = list(self.center)
        self.raw_history.append(pt)

        # Smoothing & Window Filtering: Moving average of the last 5 frames to reduce coordinate noise
        window_size = 5
        last_pts = self.raw_history[-window_size:]
        avg_x = sum(p[0] for p in last_pts) / len(last_pts)
        avg_y = sum(p[1] for p in last_pts) / len(last_pts)

        # Window Filtering: Save history of the last 10 frames
        self.history.append([avg_x, avg_y])
        self.history = self.history[-10:]

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class ByteTracker:
    @staticmethod
    def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
        """
        Computes Intersection over Union (IoU) between two bounding boxes.
        Format: [x1, y1, x2, y2]
        """
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        union_area = float(area_a + area_b - inter_area)
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.7,
        max_time_lost: int = 30,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.frame_id = 0
        self.next_id = 1

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []

    def update(self, detections: List[dict]) -> List[STrack]:
        """
        Updates trackers using ByteTrack algorithm logic.
        Detections format: list of dicts with keys "bbox", "confidence", "class"
        """
        self.frame_id += 1

        # 1. Split detections into high score and low score detections
        det_high, det_low = self._preprocess_detections(detections)

        # 2. Formulate candidate tracking pool
        strack_pool: List[STrack] = self.tracked_stracks + self.lost_stracks

        # First association: High score detections and track pool
        matched_pairs_1, unmatched_tracks_1, unmatched_dets_1 = self._linear_assignment(
            strack_pool, det_high, self.match_thresh
        )

        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []

        # Update matched tracks from first association
        for track, matched_det in matched_pairs_1:
            if track.state == TrackState.Lost:
                track.update(matched_det, self.frame_id)
                refind_stracks.append(track)
            else:
                track.update(matched_det, self.frame_id)
                activated_stracks.append(track)

        # Second association: Low score detections and remaining tracks
        matched_pairs_2, unmatched_tracks_2, unmatched_dets_2 = self._linear_assignment(
            unmatched_tracks_1,
            det_low,
            0.5,  # Lower IoU threshold for low score detections
        )

        for track, matched_det in matched_pairs_2:
            if track.state == TrackState.Lost:
                track.update(matched_det, self.frame_id)
                refind_stracks.append(track)
            else:
                track.update(matched_det, self.frame_id)
                activated_stracks.append(track)

        # For remaining unmatched tracks, mark as lost
        lost_stracks: List[STrack] = []
        for track in unmatched_tracks_2:
            if track.state != TrackState.Lost:
                track.mark_lost()
            lost_stracks.append(track)

        # 3. Init new tracks from unmatched high-score detections
        for unmatched_det in unmatched_dets_1:
            if unmatched_det.score >= self.track_thresh:
                unmatched_det.track_id = self.next_id
                self.next_id += 1
                unmatched_det.activate(self.frame_id)
                activated_stracks.append(unmatched_det)

        # 4. Handle list updates and track expiration
        self._update_tracking_lists(activated_stracks, refind_stracks, lost_stracks)

        return self.tracked_stracks

    def _preprocess_detections(
        self, detections: List[dict]
    ) -> Tuple[List[STrack], List[STrack]]:
        det_high: List[STrack] = []
        det_low: List[STrack] = []
        for det_dict in detections:
            bbox = det_dict.get("bbox", [0.0, 0.0, 0.0, 0.0])
            score = det_dict.get("confidence", 0.0)
            cls_name = det_dict.get("class", "unknown")

            strack = STrack(bbox, score, cls_name, -1)
            if score >= self.track_thresh:
                det_high.append(strack)
            else:
                det_low.append(strack)
        return det_high, det_low

    def _update_tracking_lists(
        self,
        activated_stracks: List[STrack],
        refind_stracks: List[STrack],
        lost_stracks: List[STrack],
    ):
        new_tracked_stracks: List[STrack] = []
        new_lost_stracks: List[STrack] = []

        for track in self.tracked_stracks + self.lost_stracks:
            if track.frame_id == self.frame_id:
                new_tracked_stracks.append(track)
            else:
                if track.state == TrackState.Lost:
                    if self.frame_id - track.frame_id > self.max_time_lost:
                        track.mark_removed()
                    else:
                        new_lost_stracks.append(track)
                else:
                    new_lost_stracks.append(track)

        for track in activated_stracks + refind_stracks:
            if track not in new_tracked_stracks:
                new_tracked_stracks.append(track)

        for track in lost_stracks:
            if track not in new_lost_stracks and track not in new_tracked_stracks:
                new_lost_stracks.append(track)

        self.tracked_stracks = new_tracked_stracks
        self.lost_stracks = new_lost_stracks

    def _linear_assignment(
        self, tracks: List[STrack], detections: List[STrack], thresh: float
    ) -> Tuple[List[Tuple[STrack, STrack]], List[STrack], List[STrack]]:
        """
        Greedy linear assignment based on IoU cost matrix.
        """
        if not tracks or not detections:
            return [], list(tracks), list(detections)

        # Compute IoU matrix and find matches
        matches = self._get_potential_matches(tracks, detections, thresh)

        # Greedy selection
        matched_pairs, matched_tracks, matched_dets = self._greedy_match(
            matches, tracks, detections
        )

        unmatched_tracks = [
            t for idx, t in enumerate(tracks) if idx not in matched_tracks
        ]
        unmatched_dets = [
            d for idx, d in enumerate(detections) if idx not in matched_dets
        ]

        return matched_pairs, unmatched_tracks, unmatched_dets

    def _get_potential_matches(
        self, tracks: List[STrack], detections: List[STrack], thresh: float
    ) -> List[Tuple[int, int, float]]:
        matches: List[Tuple[int, int, float]] = []
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self.bbox_iou(track.bbox, det.bbox)
                if iou >= thresh:
                    matches.append((i, j, iou))
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    def _greedy_match(
        self,
        matches: List[Tuple[int, int, float]],
        tracks: List[STrack],
        detections: List[STrack],
    ) -> Tuple[List[Tuple[STrack, STrack]], set, set]:
        matched_tracks = set()
        matched_dets = set()
        matched_pairs = []
        for i, j, iou in matches:
            if i not in matched_tracks and j not in matched_dets:
                matched_tracks.add(i)
                matched_dets.add(j)
                matched_pairs.append((tracks[i], detections[j]))
        return matched_pairs, matched_tracks, matched_dets


class ByteTrackerService:
    def __init__(self):
        # Dictionary mapping video_id to its respective ByteTracker instance
        self.trackers: Dict[str, ByteTracker] = {}

    def get_tracker(self, video_id: str) -> ByteTracker:
        if video_id not in self.trackers:
            self.trackers[video_id] = ByteTracker()
        return self.trackers[video_id]

    def update_tracks(self, detections: list, video_id: str = "default") -> list:
        """
        Updates trackers using ByteTrack, maps detections to unique tracking IDs.
        Caches track coordinates and history in Redis for quick retrieval.
        Returns tracked objects with active tracking IDs.
        """
        tracker = self.get_tracker(video_id)
        active_stracks = tracker.update(detections)

        tracked_objects = []
        for track in active_stracks:
            track_data = {
                "bbox": track.bbox,
                "history": track.history,
                "class": track.cls_name,
                "confidence": track.score,
            }

            # Update cache in Redis
            redis_cache.update_track(
                track_id=track.track_id, data=track_data, video_id=video_id
            )

            tracked_objects.append(
                {
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "class": track.cls_name,
                    "confidence": track.score,
                    "history": track.history,
                }
            )

        logger.debug(
            "ByteTrack updated tracks",
            tracked_count=len(tracked_objects),
            video_id=video_id,
        )
        return tracked_objects


tracker_service = ByteTrackerService()
