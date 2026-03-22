"""
SIMS — Solapur Intelligent Mobility SaaS
AI Engine: 4-stage computer vision pipeline
Stage 1: Frame ingestion (handled by app.py)
Stage 2: YOLOv11n vehicle detection
Stage 3: ByteTrack + LineZone + PolygonZone spatial logic
Stage 4: Metrics dict returned to dashboard
"""

from __future__ import annotations

import numpy as np
import cv2

# YOLOv11 via ultralytics
from ultralytics import YOLO

# Supervision: ByteTrack, zones, annotators
import supervision as sv


class TrafficAnalyzer:
    """
    Stateful per-session traffic analyzer.
    Call process_frame(frame) on each video frame to get
    an annotated frame and a metrics dict.
    """

    # COCO class IDs → Indian traffic label remapping
    # NOTE: 6-seater autos classified as car/auto until
    # Solapur-specific fine-tuning is complete (Active Learning target)
    VEHICLE_CLASSES: dict[int, str] = {
        2: "car/auto",
        3: "motorcycle/2W",
        5: "ST bus",
        7: "truck/tempo",
    }

    # Lane capacity used for density % calculation
    LANE_CAPACITY = 20

    # Frames a vehicle must be stationary before flagging a parking violation
    STATIONARY_THRESHOLD = 150  # ~5 s at 30 fps

    # Density threshold for signal extension
    DENSITY_THRESHOLD = 80.0  # percent

    def __init__(self, model_path: str = "yolo11s.pt", confidence: float = 0.35) -> None:
        """
        Parameters
        ----------
        model_path : str
            YOLO model weights filename. Defaults to yolo11s.pt (~22 MB, auto-downloaded).
        confidence : float
            YOLO inference confidence threshold (0.1 – 1.0).
            Lowered to 0.35 to catch more vehicles in dense Indian traffic.
            Detections below 0.70 are still flagged for Active Learning review.
        """
        self.model_path = model_path
        self.confidence = confidence

        # --- initialised in subsequent atomic tasks ---
        self.model: YOLO | None = None
        self.tracker: sv.ByteTrack | None = None
        self.line_zone: sv.LineZone | None = None
        self.polygon_zone: sv.PolygonZone | None = None

        # Runtime state
        self._frame_dimensions: tuple[int, int] | None = None  # (h, w)
        self._stationary_counters: dict[int, int] = {}          # tracker_id → frames
        self._parking_violations: set[int] = set()              # tracker_ids currently violating
        self._total_vehicles: int = 0                           # cumulative LineZone crossings

        # Initialise components
        self._init_model()

    # ------------------------------------------------------------------
    # (b) YOLO initialisation
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Load YOLOv11s weights. Downloads on first run (~22 MB)."""
        self.model = YOLO("yolo11s.pt")

    # ------------------------------------------------------------------
    # (c) ByteTrack initialisation
    # ------------------------------------------------------------------

    def _init_tracker(self) -> None:
        """
        Initialise ByteTrack multi-object tracker.
        Called lazily on the first frame so we have frame dimensions available.
        """
        self.tracker = sv.ByteTrack()

    # ------------------------------------------------------------------
    # (d) LineZone definition
    # ------------------------------------------------------------------

    def _init_zones(self, h: int, w: int) -> None:
        """
        Create virtual sensors sized to the actual frame dimensions.
        All coordinates are expressed as fractions of (w, h) so the
        demo works with any input video resolution.

        LineZone: a horizontal counting line at 55 % of frame height,
                  spanning 10 %–90 % of frame width.
        """
        # LineZone — counts vehicles crossing from one side to the other
        line_start = sv.Point(x=int(w * 0.10), y=int(h * 0.55))
        line_end   = sv.Point(x=int(w * 0.90), y=int(h * 0.55))
        self.line_zone = sv.LineZone(start=line_start, end=line_end)

        # PolygonZone is added in task (e) below
        self._init_polygon_zone(h, w)

    # ------------------------------------------------------------------
    # (e) PolygonZone definition
    # ------------------------------------------------------------------

    def _init_polygon_zone(self, h: int, w: int) -> None:
        """
        PolygonZone: a rectangular region on the left shoulder of the road
        (10–30 % width, 30–70 % height) representing a no-parking zone.
        Vehicles that remain here beyond STATIONARY_THRESHOLD frames are
        flagged as parking violations.

        NOTE (data privacy): In production, Gaussian blur would be applied
        to all vehicles *not* triggering a violation before any frame is
        stored or transmitted. This is omitted in the MVP.
        """
        polygon = np.array([
            [int(w * 0.10), int(h * 0.30)],
            [int(w * 0.30), int(h * 0.30)],
            [int(w * 0.30), int(h * 0.70)],
            [int(w * 0.10), int(h * 0.70)],
        ])
        self.polygon_zone = sv.PolygonZone(polygon=polygon)

    # ------------------------------------------------------------------
    # (b-new) CLAHE preprocessing
    # ------------------------------------------------------------------

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE preprocessing to improve detection on low-contrast
        CCTV footage (rain, night, washed-out daytime feeds).

        Steps:
        1. Convert BGR to LAB colour space
        2. Apply CLAHE only to the L (lightness) channel
        3. Merge channels and convert back to BGR

        This preserves colour information while enhancing contrast.
        clipLimit=2.0 prevents over-amplification of noise.
        tileGridSize=(8,8) divides frame into 8x8 regions for local enhancement.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # (f) Tracking logic
    # ------------------------------------------------------------------

    def _run_detection_and_tracking(
        self, frame: np.ndarray, inference_frame: np.ndarray | None = None
    ) -> sv.Detections:
        """
        Run YOLOv11s inference, filter to vehicle classes, then update
        ByteTrack so every detection gets a persistent tracker_id.

        inference_frame: CLAHE-preprocessed frame used for YOLO inference.
                         If None, the raw frame is used.
        Returns a sv.Detections object with tracker_ids assigned.
        """
        input_frame = inference_frame if inference_frame is not None else frame
        results = self.model(
            input_frame,
            conf=self.confidence,
            iou=0.45,           # lower than default 0.7 — removes more duplicate boxes in crowds
            agnostic_nms=True,  # class-agnostic NMS — prevents same vehicle being
                                # detected as both car and truck simultaneously
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Keep only COCO vehicle classes
        vehicle_mask = np.isin(detections.class_id, list(self.VEHICLE_CLASSES.keys()))
        detections = detections[vehicle_mask]

        # Update tracker — assigns / maintains tracker_ids
        detections = self.tracker.update_with_detections(detections)

        return detections

    # ------------------------------------------------------------------
    # (g) Parking violation logic
    # ------------------------------------------------------------------

    def _update_parking_violations(self, detections: sv.Detections) -> list[str]:
        """
        For every tracked vehicle inside the PolygonZone, increment its
        stationary counter.  When the counter exceeds STATIONARY_THRESHOLD
        the vehicle is added to _parking_violations and an alert is raised.

        Vehicles that leave the zone have their counters reset.

        Returns a list of alert strings for this frame.
        """
        alerts: list[str] = []

        # Boolean mask: which detections are inside the polygon?
        in_zone_mask = self.polygon_zone.trigger(detections=detections)

        # Collect tracker_ids currently inside the zone
        ids_in_zone: set[int] = set()
        if detections.tracker_id is not None:
            for tid, inside in zip(detections.tracker_id, in_zone_mask):
                if inside:
                    ids_in_zone.add(int(tid))

        # Update stationary counters
        for tid in ids_in_zone:
            self._stationary_counters[tid] = self._stationary_counters.get(tid, 0) + 1
            if self._stationary_counters[tid] >= self.STATIONARY_THRESHOLD:
                self._parking_violations.add(tid)

        # Remove IDs that have left the zone
        exited = set(self._stationary_counters.keys()) - ids_in_zone
        for tid in exited:
            del self._stationary_counters[tid]
            self._parking_violations.discard(tid)

        # Build alert strings for active violations
        for tid in self._parking_violations:
            alerts.append(f"Illegal parking detected! Vehicle ID {tid}")

        return alerts

    # ------------------------------------------------------------------
    # (h) Signal logic + Active Learning flag
    # ------------------------------------------------------------------

    def _compute_signal_and_density(
        self, detections: sv.Detections
    ) -> tuple[float, str, list[str], list[int]]:
        """
        Computes:
          density_percentage — vehicles visible in this frame / LANE_CAPACITY * 100
          signal_action      — Max-Pressure decision based on density
          low_conf_alerts    — Active Learning: flag detections below 0.70 confidence
          low_conf_indices   — indices into detections of low-confidence detections

        Returns (density_percentage, signal_action, low_conf_alerts, low_conf_indices).
        """
        visible_count = len(detections)
        density = min(visible_count / self.LANE_CAPACITY * 100.0, 100.0)

        signal_action = (
            "Extend Green 15s" if density >= self.DENSITY_THRESHOLD else "Maintain Cycle"
        )

        # Active Learning loop: flag low-confidence detections
        low_conf_alerts: list[str] = []
        low_conf_indices: list[int] = []
        if detections.confidence is not None:
            for i, conf in enumerate(detections.confidence):
                if conf < 0.70:
                    tid = (
                        int(detections.tracker_id[i])
                        if detections.tracker_id is not None
                        else "?"
                    )
                    low_conf_alerts.append(
                        f"Low-confidence detection (conf={conf:.2f}) — Vehicle ID {tid} flagged for review"
                    )
                    low_conf_indices.append(i)

        return density, signal_action, low_conf_alerts, low_conf_indices

    # ------------------------------------------------------------------
    # (i) process_frame() — public API (the Data Contract)
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Run the full 4-stage pipeline on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from OpenCV VideoCapture.

        Returns
        -------
        annotated_frame : np.ndarray
            Frame with bounding boxes, zone overlays, and IDs drawn.
        metrics : dict
            Exactly the keys defined in the Data Contract:
            total_vehicles, parking_violations, density_percentage,
            signal_action, alerts.
        """
        try:
            h, w = frame.shape[:2]

            # Lazy-initialise tracker and zones on first frame
            if self._frame_dimensions is None:
                self._frame_dimensions = (h, w)
                self._init_tracker()
                self._init_zones(h, w)

            # Stage 2 — CLAHE preprocessing + detection + tracking
            # Preprocessing applied only for inference; annotations drawn on original frame
            preprocessed = self._preprocess_frame(frame)
            detections = self._run_detection_and_tracking(frame, inference_frame=preprocessed)

            # Stage 3a — LineZone crossing count
            self.line_zone.trigger(detections=detections)
            # in_count gives cumulative vehicles that crossed into the zone
            self._total_vehicles = (
                self.line_zone.in_count + self.line_zone.out_count
            )

            # Stage 3b — parking violations
            parking_alerts = self._update_parking_violations(detections)

            # Stage 3c — density + signal
            density, signal_action, low_conf_alerts, low_conf_indices = (
                self._compute_signal_and_density(detections)
            )

            # Stage 4 — annotated frame
            annotated = self._draw_annotations(frame.copy(), detections)

            # class_counts: vehicles visible in this frame by Indian label
            class_counts: dict[str, int] = {
                "car/auto": 0, "motorcycle/2W": 0,
                "ST bus": 0, "truck/tempo": 0,
            }
            if detections.class_id is not None:
                for cid in detections.class_id:
                    label = self.VEHICLE_CLASSES.get(int(cid))
                    if label and label in class_counts:
                        class_counts[label] += 1

            metrics = {
                "total_vehicles": self._total_vehicles,
                "parking_violations": len(self._parking_violations),
                "density_percentage": round(density, 1),
                "signal_action": signal_action,
                "alerts": parking_alerts + low_conf_alerts,
                "class_counts": class_counts,
                "flagged_crops": self._get_flagged_crops(frame, detections, low_conf_indices),
            }

            return annotated, metrics

        except Exception as exc:
            # Never crash the app on a bad frame — return the raw frame + safe metrics
            safe_metrics = {
                "total_vehicles": self._total_vehicles,
                "parking_violations": len(self._parking_violations),
                "density_percentage": 0.0,
                "signal_action": "Maintain Cycle",
                "alerts": [f"Frame processing error: {exc}"],
                "class_counts": {"car/auto": 0, "motorcycle/2W": 0, "ST bus": 0, "truck/tempo": 0},
                "flagged_crops": [],
            }
            return frame, safe_metrics

    # ------------------------------------------------------------------
    # Active Learning: crop flagged detections for human review
    # ------------------------------------------------------------------

    def _get_flagged_crops(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        low_conf_indices: list[int],
    ) -> list[np.ndarray]:
        """
        Crop the bounding box region of each low-confidence detection.
        Returns list of BGR image crops for saving/display.
        """
        crops: list[np.ndarray] = []
        if detections.xyxy is None or len(low_conf_indices) == 0:
            return crops
        for idx in low_conf_indices:
            if idx < len(detections.xyxy):
                x1, y1, x2, y2 = map(int, detections.xyxy[idx])
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
        return crops

    # ------------------------------------------------------------------
    # (j) Annotated frame drawing
    # ------------------------------------------------------------------

    def _draw_annotations(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """
        Draw on the frame (in-place on the copy passed by process_frame):
          - Bounding boxes with tracker IDs and class labels
          - LineZone overlay
          - PolygonZone overlay (red tint for no-parking area)
          - Red boxes around parking violators
        """
        # --- Bounding box annotator ---
        box_annotator = sv.BoxAnnotator(thickness=2)

        # Build label strings: "car #3  0.87"
        labels: list[str] = []
        if detections.tracker_id is not None and detections.class_id is not None:
            for tid, cid, conf in zip(
                detections.tracker_id,
                detections.class_id,
                detections.confidence if detections.confidence is not None else [None] * len(detections),
            ):
                class_name = self.VEHICLE_CLASSES.get(int(cid), "vehicle")
                conf_str = f"  {conf:.2f}" if conf is not None else ""
                labels.append(f"{class_name} #{int(tid)}{conf_str}")
        else:
            labels = [""] * len(detections)

        frame = box_annotator.annotate(scene=frame, detections=detections)

        # --- Label annotator ---
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        frame = label_annotator.annotate(
            scene=frame, detections=detections, labels=labels
        )

        # --- PolygonZone overlay ---
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.polygon_zone,
            color=sv.Color.RED,
            thickness=2,
        )
        frame = zone_annotator.annotate(scene=frame)

        # --- LineZone overlay ---
        line_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.5)
        frame = line_annotator.annotate(frame=frame, line_counter=self.line_zone)

        # --- Red border on parking violators ---
        if detections.tracker_id is not None and detections.xyxy is not None:
            for tid, xyxy in zip(detections.tracker_id, detections.xyxy):
                if int(tid) in self._parking_violations:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

        return frame
