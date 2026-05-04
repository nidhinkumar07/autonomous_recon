import cv2
import os
import sys
import time
import tempfile
import threading
import streamlit as st
from detection import detect_with_yolo, draw_detections, load_yolo_model
from ui_components import display_summary

_SUMMARY_EVERY = 10
_VEHICLE_SET   = frozenset(['car', 'truck', 'bus', 'motorcycle', 'bicycle'])
_ANIMAL_SET    = frozenset(['dog', 'cat', 'bird', 'horse', 'sheep', 'cow'])

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE — threading worker pattern (works on all Streamlit versions)
#
# Problem with while-loop approach:
#   Streamlit reruns the entire script on every sidebar interaction.
#   A `while` loop inside the script gets killed mid-frame → crash/freeze.
#
# Problem with st.rerun()-as-loop approach:
#   st.rerun() destroys and recreates every widget including st.image(),
#   causing a visible blink on every frame regardless of VideoCapture caching.
#
# Problem with st.fragment:
#   Only available in Streamlit ≥ 1.33. Users on 1.29 get AttributeError.
#
# Solution — background WorkerThread:
#   • A daemon thread owns the VideoCapture. It opens once on Start and
#     runs a tight read→detect→draw loop independently of Streamlit reruns.
#   • The thread writes its latest drawn frame into a shared FrameBuffer
#     (a plain object with a threading.Lock). No Streamlit calls in the thread.
#   • The main script runs a display while-loop that only calls
#     `placeholder.image(frame)`. This is extremely fast (no YOLO), so
#     sidebar-triggered reruns that kill it are harmless — the worker thread
#     keeps processing and the next rerun immediately re-enters the display loop.
#   • Sidebar filter settings are stored in a thread-safe AtomicConfig dict
#     that the worker reads on every frame. Changes apply within one frame.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Thread-safe primitives
# ══════════════════════════════════════════════════════════════════════════════

class FrameBuffer:
    """Holds the latest processed frame + stats. Thread-safe via a lock."""
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None
        self._stats = {}

    def put(self, frame, stats: dict):
        with self._lock:
            self._frame = frame
            self._stats = stats

    def get(self):
        with self._lock:
            return self._frame, dict(self._stats)


class AtomicConfig:
    """Thread-safe snapshot of sidebar settings, updated by the main thread."""
    def __init__(self):
        self._lock = threading.Lock()
        self._cfg  = {}

    def update(self, cfg: dict):
        with self._lock:
            self._cfg = dict(cfg)

    def get(self) -> dict:
        with self._lock:
            return dict(self._cfg)


# ══════════════════════════════════════════════════════════════════════════════
# Worker thread
# ══════════════════════════════════════════════════════════════════════════════

class DetectionWorker(threading.Thread):
    """
    Background thread that owns the VideoCapture and runs detection.
    Never calls any Streamlit function — only writes to FrameBuffer.
    """

    def __init__(self, source, model, tracker, frame_buffer: FrameBuffer,
                 config: AtomicConfig, stop_event: threading.Event,
                 is_webcam: bool = True):
        super().__init__(daemon=True)
        self.source       = source        # 0 for webcam, file path for video
        self.model        = model
        self.tracker      = tracker
        self.buf          = frame_buffer
        self.cfg          = config
        self.stop_event   = stop_event
        self.is_webcam    = is_webcam
        self.total_frames = 0             # filled in for video files

    def _open_cap(self) -> cv2.VideoCapture:
        if self.is_webcam:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cfg = self.cfg.get()
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.get('webcam_width',  640))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get('webcam_height', 480))
        else:
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap

    def _filter(self, detections: list, cfg: dict) -> list:
        threshold     = cfg.get('confidence_threshold', 0.5)
        show_all      = cfg.get('show_everything',  False)
        show_persons  = cfg.get('show_persons',     True)
        show_vehicles = cfg.get('show_vehicles',    True)
        show_animals  = cfg.get('show_animals',     False)
        out = []
        for det in detections:
            if det['confidence'] < threshold:
                continue
            cn = det['class'].lower()
            if (show_all
                    or (cn == 'person'     and show_persons)
                    or (cn in _VEHICLE_SET and show_vehicles)
                    or (cn in _ANIMAL_SET  and show_animals)):
                out.append(det)
        return out

    def run(self):
        cap = self._open_cap()
        if not cap.isOpened():
            self.buf.put(None, {"error": "Could not open capture source."})
            return

        frame_count     = 0
        processed_count = 0
        detected_total  = 0
        start_time      = time.time()

        try:
            while not self.stop_event.is_set():
                cfg        = self.cfg.get()
                frame_skip = max(1, cfg.get('frame_skip', 2))

                # For webcam: drain buffer, keep freshest frame
                grabbed = None
                if self.is_webcam:
                    for _ in range(frame_skip):
                        ret, frame = cap.read()
                        if ret:
                            grabbed = frame
                    frame_count += frame_skip
                else:
                    ret, frame = cap.read()
                    if not ret:
                        # End of video
                        self.buf.put(None, {"finished": True,
                                            "processed": processed_count})
                        break
                    grabbed = frame
                    frame_count += 1
                    # Skip frames for video
                    if frame_count % frame_skip != 0:
                        continue

                if grabbed is None:
                    time.sleep(0.01)
                    continue

                frame_rgb = cv2.cvtColor(grabbed, cv2.COLOR_BGR2RGB)

                t0    = time.perf_counter()
                raw   = detect_with_yolo(self.model, frame_rgb)
                filt  = self._filter(raw, cfg)
                track = self.tracker.update_tracking(filt, frame_count)
                proc_time = time.perf_counter() - t0

                detected_total  += len(filt)
                processed_count += 1
                elapsed          = time.time() - start_time
                fps_val          = 1.0 / proc_time if proc_time > 0 else 0

                drawn, det_count, _ = draw_detections(
                    frame_rgb, track, frame_count,
                    "webcam" if self.is_webcam else "video",
                    draw_cfg=cfg,                        # thread-safe: no session_state
                    object_data_map=self.tracker.object_data)

                stats = {
                    "frame_count":      frame_count,
                    "processed_count":  processed_count,
                    "det_count":        det_count,
                    "detected_total":   detected_total,
                    "fps":              fps_val,
                    "elapsed":          elapsed,
                    "active_objects":   len(track["active_objects"]),
                    "total_frames":     self.total_frames,
                    "finished":         False,
                    "error":            None,
                }
                self.buf.put(drawn, stats)

        finally:
            cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_model():
    if st.session_state.current_model is None:
        from session_state import YOLO_MODELS
        model_file = YOLO_MODELS[st.session_state.selected_model_label]
        st.session_state.current_model, _ = load_yolo_model(model_file)
        st.session_state.model_ready = True
    return st.session_state.current_model


def _elapsed_str(secs: float) -> str:
    s = int(secs)
    return f"{s // 60:02d}:{s % 60:02d}"


def _current_config() -> dict:
    """Snapshot all sidebar settings for the worker thread."""
    return {
        'confidence_threshold': st.session_state.confidence_threshold,
        'show_everything':      st.session_state.show_everything,
        'show_persons':         st.session_state.show_persons,
        'show_vehicles':        st.session_state.show_vehicles,
        'show_animals':         st.session_state.show_animals,
        'frame_skip':           st.session_state.frame_skip,
        'webcam_width':         st.session_state.webcam_width,
        'webcam_height':        st.session_state.webcam_height,
        # Draw settings — passed to draw_detections() in the worker thread
        # so it never needs to touch st.session_state (no ScriptRunContext)
        'show_labels':          st.session_state.show_labels,
        'show_ids':             st.session_state.show_ids,
        'show_confidence':      st.session_state.show_confidence,
        'box_thickness':        st.session_state.box_thickness,
    }


def _save_upload_once(uploaded_file) -> str:
    file_id     = getattr(uploaded_file, 'file_id', None) or id(uploaded_file)
    cached_id   = st.session_state.get('_uploaded_file_id')
    cached_path = st.session_state.get('_uploaded_file_path', '')
    if cached_id == file_id and cached_path and os.path.exists(cached_path):
        return cached_path
    if cached_path and os.path.exists(cached_path):
        try: os.remove(cached_path)
        except OSError: pass
    suffix = os.path.splitext(uploaded_file.name)[-1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    st.session_state._uploaded_file_id   = file_id
    st.session_state._uploaded_file_path = path
    return path


def _get_video_info(video_path: str) -> dict:
    key = f'_vinfo_{video_path}'
    if key in st.session_state:
        return st.session_state[key]
    info = {}
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info   = {"fps": fps, "frames": frames, "width": w, "height": h,
                  "duration": frames / fps if fps > 0 else 0}
        cap.release()
    st.session_state[key] = info
    return info


def _stop_worker(key_worker='_worker', key_event='_stop_event',
                 key_buf='_frame_buf', key_cfg='_worker_cfg'):
    """Stop and clean up any running DetectionWorker."""
    event = st.session_state.get(key_event)
    if event:
        event.set()
    worker = st.session_state.get(key_worker)
    if worker and worker.is_alive():
        worker.join(timeout=2.0)
    for k in (key_worker, key_event, key_buf, key_cfg):
        st.session_state.pop(k, None)


def _start_worker(source, model, tracker, is_webcam: bool,
                  key_worker='_worker', key_event='_stop_event',
                  key_buf='_frame_buf', key_cfg='_worker_cfg'):
    """Create, configure, and start a DetectionWorker. Returns (buf, cfg)."""
    _stop_worker(key_worker, key_event, key_buf, key_cfg)  # clean up old one
    stop_event = threading.Event()
    buf        = FrameBuffer()
    cfg        = AtomicConfig()
    cfg.update(_current_config())
    worker = DetectionWorker(source, model, tracker, buf, cfg,
                             stop_event, is_webcam=is_webcam)
    worker.start()
    st.session_state[key_worker] = worker
    st.session_state[key_event]  = stop_event
    st.session_state[key_buf]    = buf
    st.session_state[key_cfg]    = cfg
    return buf, cfg


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Webcam
# ══════════════════════════════════════════════════════════════════════════════

def _render_webcam_tab():
    st.markdown("## Live Webcam Detection")
    st.session_state.active_tab = "webcam"

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        if not st.session_state.webcam_active:
            # ── Idle ───────────────────────────────────────────────────────
            if st.session_state.get('model_missing', False):
                err = st.session_state.get('model_error') or ''
                st.error(f"⚠️ **YOLO model could not be loaded.**\n\n{err}")
            elif st.button("▶ Start Webcam Detection", key="start_webcam_main", type="primary"):
                model = _ensure_model()
                st.session_state.webcam_active    = True
                st.session_state.video_processing = False
                st.session_state.object_tracker.reset()
                st.session_state.webcam_stats = {}
                _stop_worker()                         # clean up any old worker
                _start_worker(0, model,
                              st.session_state.object_tracker,
                              is_webcam=True)
                st.rerun()
            return

        # ── Active ─────────────────────────────────────────────────────────
        st.markdown(
            '<div class="processing-card"><span class="pulse-dot"></span>'
            'Webcam active — sidebar changes apply within one frame, no interruption.'
            '</div>', unsafe_allow_html=True)

        if st.button("⏹ Stop Webcam", key="stop_webcam_btn", type="secondary"):
            st.session_state.webcam_active = False
            _stop_worker()
            st.rerun()
            return

        buf: FrameBuffer  = st.session_state.get('_frame_buf')
        cfg: AtomicConfig = st.session_state.get('_worker_cfg')
        worker            = st.session_state.get('_worker')

        if buf is None or worker is None:
            st.error("Worker not initialised. Please stop and restart.")
            st.session_state.webcam_active = False
            st.rerun()
            return

        # Push latest sidebar settings to worker on every rerun
        if cfg:
            cfg.update(_current_config())

        placeholder    = st.empty()
        stats_ph       = st.empty()
        summary_ticker = st.session_state.get('_wc_summary_ticker', 0)

        # Display loop — only calls placeholder.image(), extremely fast
        # Sidebar reruns that interrupt this loop are harmless because the
        # worker thread keeps processing; next rerun re-enters this loop.
        while st.session_state.webcam_active:
            frame, stats = buf.get()

            # Error from worker
            if stats.get('error'):
                st.error(stats['error'])
                st.session_state.webcam_active = False
                _stop_worker()
                break

            if frame is not None:
                placeholder.image(frame, channels="RGB",
                    caption=f"Frame {stats.get('frame_count', 0)} — "
                            f"Tracking {stats.get('active_objects', 0)} objects")

                with stats_ph.container():
                    st.markdown("### Live Statistics")
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    with mc1: st.metric("Frames",     stats.get('frame_count', 0))
                    with mc2: st.metric("Detections", stats.get('det_count', 0))
                    with mc3: st.metric("Total Det.", stats.get('detected_total', 0))
                    with mc4: st.metric("FPS",        f"{stats.get('fps', 0):.1f}")
                    with mc5: st.metric("Elapsed",    _elapsed_str(stats.get('elapsed', 0)))

                    summary_ticker += 1
                    st.session_state._wc_summary_ticker = summary_ticker
                    if summary_ticker % _SUMMARY_EVERY == 0:
                        display_summary()

            # Store last stats for right panel
            st.session_state.webcam_stats = stats

            time.sleep(0.03)   # ~30 fps display cap, prevents CPU spin

    with col2:
        if not st.session_state.webcam_active:
            st.markdown("#### Webcam Settings")
            st.markdown(f"""
            <div class="info-card">
                <strong style="color:#94A3B8;">Resolution:</strong>
                <span style="color:#CBD5E1;">{st.session_state.webcam_width}×{st.session_state.webcam_height}</span><br>
                <strong style="color:#94A3B8;">Frame skip:</strong>
                <span style="color:#CBD5E1;">every {st.session_state.frame_skip} frames</span><br>
                <strong style="color:#94A3B8;">Confidence:</strong>
                <span style="color:#CBD5E1;">&ge; {st.session_state.confidence_threshold:.2f}</span><br>
                <strong style="color:#94A3B8;">Model:</strong>
                <span style="color:#CBD5E1;">{st.session_state.selected_model_label}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card" style="margin-top:0.75rem;">
                <strong style="color:#94A3B8;">Steps:</strong><br>
                <span style="color:#94A3B8;">1. Configure settings in sidebar</span><br>
                <span style="color:#94A3B8;">2. Click Start Webcam Detection</span><br>
                <span style="color:#94A3B8;">3. Sidebar changes apply live</span><br>
                <span style="color:#94A3B8;">4. Click Stop to end the session</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("#### Session Stats")
            stats = st.session_state.get('webcam_stats', {})
            if stats:
                st.metric("FPS",             f"{stats.get('fps', 0):.1f}")
                st.metric("Frames Captured", stats.get('frame_count', 0))
                st.metric("Total Detections",stats.get('detected_total', 0))
                st.metric("Elapsed",         _elapsed_str(stats.get('elapsed', 0)))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Upload Video
# ══════════════════════════════════════════════════════════════════════════════

def _render_video_tab():
    st.markdown("## Upload Video for Detection")
    st.session_state.active_tab = "video"

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'mpg', 'mpeg', 'wmv'],
        help="Upload a video file for object detection and tracking",
        key="video_uploader_main",
    )

    if uploaded_file is None:
        st.session_state.video_path    = None
        st.session_state.uploaded_file = None
        st.info("Please upload a video file to start detection.")
        return

    video_path = _save_upload_once(uploaded_file)
    st.session_state.video_path    = video_path
    st.session_state.uploaded_file = uploaded_file

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        if not st.session_state.video_processing:
            # ── Idle ───────────────────────────────────────────────────────
            st.video(uploaded_file)
            if st.session_state.get('model_missing', False):
                err = st.session_state.get('model_error') or ''
                st.error(f"⚠️ **YOLO model could not be loaded.**\n\n{err}")
            elif st.button("▶ Start Video Processing", key="start_video_main", type="primary"):
                model = _ensure_model()
                info  = _get_video_info(video_path)
                st.session_state.video_processing      = True
                st.session_state.webcam_active         = False
                st.session_state._vid_total_frames     = info.get('frames', 0)
                st.session_state.video_stats           = {}
                st.session_state.object_tracker.reset()
                _stop_worker('_vid_worker', '_vid_stop', '_vid_buf', '_vid_cfg')
                _start_worker(video_path, model,
                              st.session_state.object_tracker,
                              is_webcam=False,
                              key_worker='_vid_worker', key_event='_vid_stop',
                              key_buf='_vid_buf',    key_cfg='_vid_cfg')
                st.rerun()

        else:
            # ── Active ─────────────────────────────────────────────────────
            st.markdown(
                '<div class="processing-card"><span class="pulse-dot"></span>'
                'Video processing active — sidebar changes apply within one frame.'
                '</div>', unsafe_allow_html=True)

            if st.button("⏹ Stop Processing", key="stop_video_btn", type="secondary"):
                st.session_state.video_processing = False
                _stop_worker('_vid_worker', '_vid_stop', '_vid_buf', '_vid_cfg')
                st.rerun()
                return

            buf: FrameBuffer  = st.session_state.get('_vid_buf')
            cfg: AtomicConfig = st.session_state.get('_vid_cfg')
            worker            = st.session_state.get('_vid_worker')
            total_frames      = st.session_state.get('_vid_total_frames', 0)

            if buf is None or worker is None:
                st.error("Worker not initialised. Please stop and restart.")
                st.session_state.video_processing = False
                st.rerun()
                return

            if cfg:
                cfg.update(_current_config())

            frame_ph    = st.empty()
            stats_ph    = st.empty()
            progress_ph = st.empty()
            summary_ticker = st.session_state.get('_vid_summary_ticker', 0)

            while st.session_state.video_processing:
                frame, stats = buf.get()

                if stats.get('error'):
                    st.error(stats['error'])
                    st.session_state.video_processing = False
                    _stop_worker('_vid_worker', '_vid_stop', '_vid_buf', '_vid_cfg')
                    break

                if stats.get('finished'):
                    processed = stats.get('processed', 0)
                    st.session_state.video_processing = False
                    _stop_worker('_vid_worker', '_vid_stop', '_vid_buf', '_vid_cfg')
                    st.success(f"✅ Completed! Processed {processed} frames.")
                    break

                if frame is not None:
                    fc = stats.get('frame_count', 0)
                    if total_frames > 0:
                        progress_ph.progress(min(fc / total_frames, 1.0))

                    frame_ph.image(frame, channels="RGB",
                        caption=f"Frame {fc}/{total_frames} — "
                                f"Tracking {stats.get('active_objects', 0)} objects")

                    with stats_ph.container():
                        st.markdown("#### Processing Statistics")
                        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                        pct = (fc / total_frames * 100) if total_frames > 0 else 0
                        with mc1: st.metric("Progress",   f"{pct:.1f}%")
                        with mc2: st.metric("Detections", stats.get('det_count', 0))
                        with mc3: st.metric("Total Det.", stats.get('detected_total', 0))
                        with mc4: st.metric("FPS",        f"{stats.get('fps', 0):.1f}")
                        with mc5: st.metric("Elapsed",    _elapsed_str(stats.get('elapsed', 0)))

                        summary_ticker += 1
                        st.session_state._vid_summary_ticker = summary_ticker
                        if summary_ticker % _SUMMARY_EVERY == 0:
                            display_summary()

                st.session_state.video_stats = stats
                time.sleep(0.03)

    with col2:
        if not st.session_state.video_processing:
            st.markdown("#### Video Information")
            info = _get_video_info(video_path)
            if info:
                ci1, ci2 = st.columns(2)
                with ci1:
                    st.metric("Frame Rate",   f"{info['fps']:.1f} fps")
                    st.metric("Total Frames", info['frames'])
                with ci2:
                    st.metric("Resolution",   f"{info['width']}×{info['height']}")
                    st.metric("Duration",     f"{info['duration']:.1f}s")
            else:
                st.warning("Could not read video properties.")
            st.markdown("""
            <div class="info-card">
                <strong style="color:#94A3B8;">Steps:</strong><br>
                <span style="color:#94A3B8;">1. Upload a video file</span><br>
                <span style="color:#94A3B8;">2. Click Start Video Processing</span><br>
                <span style="color:#94A3B8;">3. Sidebar changes apply instantly</span><br>
                <span style="color:#94A3B8;">4. Click Stop to pause anytime</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("#### Processing Stats")
            vstats = st.session_state.get('video_stats', {})
            if vstats:
                tf = vstats.get('total_frames', 1)
                fc = vstats.get('frame_count', 0)
                st.metric("FPS",              f"{vstats.get('fps', 0):.1f}")
                st.metric("Frames Done",      fc)
                st.metric("Remaining",        max(0, tf - fc))
                st.metric("Total Detections", vstats.get('detected_total', 0))


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def render_tabs():
    tab1, tab2 = st.tabs(["📷 Live Webcam", "🎬 Upload Video"])
    with tab1:
        _render_webcam_tab()
    with tab2:
        _render_video_tab()