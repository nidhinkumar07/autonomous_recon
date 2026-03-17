import cv2
import os
import sys
import time
import tempfile
import atexit
import streamlit as st
from detection import detect_with_yolo, filter_detections, draw_detections, load_yolo_model
from ui_components import display_summary

# How often (processed frames) to refresh the tracking summary widget
_SUMMARY_EVERY = 10


# ── Helpers ────────────────────────────────────────────────────────────────

def _ensure_model():
    """Return the cached model; reload if the selected model changed."""
    if st.session_state.current_model is None:
        from session_state import YOLO_MODELS
        model_file = YOLO_MODELS[st.session_state.selected_model_label]
        st.session_state.current_model = load_yolo_model(model_file)
        st.session_state.model_ready   = True
    return st.session_state.current_model


def _reset_webcam_stats():
    st.session_state.webcam_start_time       = time.time()
    st.session_state.webcam_frame_count      = 0
    st.session_state.webcam_detected_objects = 0
    st.session_state.webcam_processing_time  = 0.0


def _reset_video_stats():
    st.session_state.video_start_time       = time.time()
    st.session_state.video_frame_count      = 0
    st.session_state.video_detected_objects = 0
    st.session_state.video_processing_time  = 0.0


def _save_upload_once(uploaded_file) -> str:
    """
    Write the uploaded file to disk exactly once per unique file (fix #8).
    The temp path is cached in session_state.  When a new file is uploaded,
    the old temp file is deleted (fix #8 — cleanup).
    """
    file_id = getattr(uploaded_file, 'file_id', None) or id(uploaded_file)
    cached_id   = st.session_state.get('_uploaded_file_id')
    cached_path = st.session_state.get('_uploaded_file_path', '')

    if cached_id == file_id and cached_path and os.path.exists(cached_path):
        return cached_path

    # Delete old temp file before writing new one (fix #8)
    if cached_path and os.path.exists(cached_path):
        try:
            os.remove(cached_path)
        except OSError:
            pass

    suffix = os.path.splitext(uploaded_file.name)[-1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    st.session_state._uploaded_file_id   = file_id
    st.session_state._uploaded_file_path = path
    return path


def _get_video_info(video_path: str) -> dict:
    """Cache video metadata to avoid reopening the file on every rerun."""
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


def _open_webcam() -> cv2.VideoCapture:
    """
    Open webcam with the platform-correct backend (fix #7).
    Returns a VideoCapture; caller is responsible for releasing it.
    """
    # CAP_DSHOW only exists on Windows; use default backend on other platforms
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  st.session_state.webcam_width)   # fix #15
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, st.session_state.webcam_height)  # fix #15
    return cap


def _elapsed_str(start_ts: float) -> str:
    """Format elapsed seconds as mm:ss."""
    secs = int(time.time() - start_ts)
    return f"{secs // 60:02d}:{secs % 60:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Webcam
# ══════════════════════════════════════════════════════════════════════════════

def _render_webcam_tab():
    st.markdown("## Live Webcam Detection")
    st.session_state.active_tab = "webcam"

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        if st.session_state.webcam_active:
            # Animated pulse card (fix #19)
            st.markdown(
                '<div class="processing-card">'
                '<span class="pulse-dot"></span>'
                'Webcam is active — objects are being tracked with unique IDs.'
                '</div>',
                unsafe_allow_html=True,
            )

            frame_placeholder = st.empty()
            stats_placeholder = st.empty()

            if st.button("⏹ Stop Webcam", key="stop_webcam_btn", type="secondary"):
                st.session_state.webcam_active = False
                frame_placeholder.empty()
                st.rerun()

            try:
                cap = _open_webcam()
                if not cap.isOpened():
                    st.error("Could not access webcam. Please check your connection.")
                    st.session_state.webcam_active = False
                    st.rerun()

                model = _ensure_model()
                _reset_webcam_stats()

                frame_count     = 0
                processed_count = 0
                frame_skip      = st.session_state.frame_skip  # snapshotted at start

                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue

                    frame_count += 1
                    st.session_state.webcam_frame_count += 1

                    if frame_count % frame_skip != 0:
                        continue  # cap.read() self-paces — no sleep needed

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    t0 = time.perf_counter()
                    raw   = detect_with_yolo(model, frame_rgb)
                    filt  = filter_detections(raw)
                    track = st.session_state.object_tracker.update_tracking(filt, frame_count)
                    proc_time = time.perf_counter() - t0

                    st.session_state.webcam_detected_objects += len(filt)
                    st.session_state.webcam_processing_time  += proc_time
                    # Populate webcam_stats dict (fix #17)
                    st.session_state.webcam_stats = {
                        "frame_count":      st.session_state.webcam_frame_count,
                        "total_detections": st.session_state.webcam_detected_objects,
                        "avg_fps":          (processed_count / st.session_state.webcam_processing_time
                                             if st.session_state.webcam_processing_time > 0 else 0),
                    }

                    drawn, det_count, _ = draw_detections(
                        frame_rgb, track, frame_count, "webcam")
                    st.session_state.current_frame     = drawn
                    st.session_state.detection_counter = det_count

                    frame_placeholder.image(drawn, channels="RGB",
                        caption=f"Frame {frame_count} — Tracking "
                                f"{len(track['active_objects'])} objects")

                    processed_count += 1
                    fps_val = 1.0 / proc_time if proc_time > 0 else 0

                    with stats_placeholder.container():
                        st.markdown("## Live Statistics")
                        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                        with mc1: st.metric("Frames",     st.session_state.webcam_frame_count)
                        with mc2: st.metric("Detections", det_count)
                        with mc3: st.metric("Total Det.", st.session_state.webcam_detected_objects)
                        with mc4: st.metric("FPS",        f"{fps_val:.1f}")
                        with mc5: st.metric("Elapsed",    # fix #16
                                            _elapsed_str(st.session_state.webcam_start_time))

                        if processed_count % _SUMMARY_EVERY == 0:
                            display_summary()

                cap.release()
                _reset_webcam_stats()

            except Exception as e:
                st.error(f"Webcam error: {e}")
                st.session_state.webcam_active = False
                st.rerun()

        else:
            if st.session_state.get('model_missing', False):
                err = st.session_state.get('model_error') or ''
                st.error(f"⚠️ **YOLO model could not be loaded.**\n\n{err}")
            elif st.button("▶ Start Webcam Detection", key="start_webcam_main", type="primary"):
                st.session_state.webcam_active    = True
                st.session_state.video_processing = False
                st.session_state.object_tracker.reset()
                st.rerun()

    # col2 — webcam settings summary (fix #20 — was empty `pass`)
    with col2:
        if not st.session_state.webcam_active:
            st.markdown("#### Webcam Settings")
            st.markdown(f"""
            <div class="info-card">
                <strong style="color:#94A3B8;">Resolution:</strong>
                <span style="color:#CBD5E1;">
                    {st.session_state.webcam_width}×{st.session_state.webcam_height}
                </span><br>
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
                <span style="color:#94A3B8;">3. Objects are tracked with unique IDs</span><br>
                <span style="color:#94A3B8;">4. Click Stop to end the session</span><br>
                <span style="color:#94A3B8;">5. Export results as CSV or JSON</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show live session stats in right panel when active (fix #20)
            st.markdown("#### Session Stats")
            stats = st.session_state.webcam_stats
            if stats:
                st.metric("Avg FPS",         f"{stats.get('avg_fps', 0):.1f}")
                st.metric("Frames Captured", stats.get('frame_count', 0))
                st.metric("Total Detections",stats.get('total_detections', 0))
            # Export available even while running
            st.markdown("#### Export")
            


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
            st.video(uploaded_file)

        if st.session_state.video_processing:
            # Animated pulse card (fix #19)
            st.markdown(
                '<div class="processing-card">'
                '<span class="pulse-dot"></span>'
                'Video processing active — objects are being tracked with unique IDs.'
                '</div>',
                unsafe_allow_html=True,
            )

            frame_ph    = st.empty()
            stats_ph    = st.empty()
            progress_ph = st.empty()

            if st.button("⏹ Stop Processing", key="stop_video_btn", type="secondary"):
                st.session_state.video_processing = False
                frame_ph.empty()
                st.rerun()

            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Could not open video file. It may be corrupted.")
                    st.session_state.video_processing = False
                    st.rerun()

                fps          = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if fps <= 0 or total_frames <= 0:
                    st.error("Invalid video: could not read properties.")
                    cap.release()
                    st.session_state.video_processing = False
                    st.rerun()

                model = _ensure_model()
                _reset_video_stats()

                frame_count     = 0
                processed_count = 0
                frame_skip      = st.session_state.frame_skip  # snapshotted at start

                while st.session_state.video_processing and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    st.session_state.video_frame_count += 1

                    # Update progress only on processed frames (fix #9)
                    if frame_count % frame_skip == 0 and total_frames > 0:
                        progress_ph.progress(frame_count / total_frames)

                    if frame_count % frame_skip != 0:
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    t0 = time.perf_counter()
                    raw   = detect_with_yolo(model, frame_rgb)
                    filt  = filter_detections(raw)
                    track = st.session_state.object_tracker.update_tracking(filt, frame_count)
                    proc_time = time.perf_counter() - t0

                    st.session_state.video_detected_objects += len(filt)
                    st.session_state.video_processing_time  += proc_time
                    # Populate video_stats dict (fix #17)
                    st.session_state.video_stats = {
                        "frame_count":      frame_count,
                        "total_frames":     total_frames,
                        "total_detections": st.session_state.video_detected_objects,
                        "avg_fps":          (processed_count / st.session_state.video_processing_time
                                             if st.session_state.video_processing_time > 0 else 0),
                    }

                    drawn, det_count, _ = draw_detections(
                        frame_rgb, track, frame_count, "video")
                    st.session_state.current_frame     = drawn
                    st.session_state.detection_counter = det_count

                    frame_ph.image(drawn, channels="RGB",
                        caption=f"Frame {frame_count}/{total_frames} — "
                                f"Tracking {len(track['active_objects'])} objects")

                    processed_count += 1
                    fps_val = 1.0 / proc_time if proc_time > 0 else 0

                    with stats_ph.container():
                        st.markdown("#### Processing Statistics")
                        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                        with mc1:
                            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
                            st.metric("Progress", f"{pct:.1f}%")
                        with mc2: st.metric("Detections", det_count)
                        with mc3: st.metric("Total Det.", st.session_state.video_detected_objects)
                        with mc4: st.metric("FPS",        f"{fps_val:.1f}")
                        with mc5: st.metric("Elapsed",    # fix #16
                                            _elapsed_str(st.session_state.video_start_time))

                        if processed_count % _SUMMARY_EVERY == 0:
                            display_summary()

                cap.release()
                _reset_video_stats()
                st.success(f"✅ Completed! Processed {frame_count} frames.")

            except Exception as e:
                st.error(f"Video processing error: {e}")
                st.session_state.video_processing = False
                st.rerun()

        else:
            if st.session_state.get('model_missing', False):
                err = st.session_state.get('model_error') or ''
                st.error(f"⚠️ **YOLO model could not be loaded.**\n\n{err}")
            elif st.button("▶ Start Video Processing", key="start_video_main", type="primary"):
                st.session_state.video_processing = True
                st.session_state.webcam_active    = False
                st.session_state.object_tracker.reset()
                st.rerun()

    # Right column — video info + export
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
                <span style="color:#94A3B8;">1. Video uploaded successfully</span><br>
                <span style="color:#94A3B8;">2. Click Start Video Processing</span><br>
                <span style="color:#94A3B8;">3. Objects tracked with unique IDs</span><br>
                <span style="color:#94A3B8;">4. Click Stop to pause anytime</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Live session stats in right panel (fix #20)
            st.markdown("#### Processing Stats")
            vstats = st.session_state.video_stats
            if vstats:
                tf = vstats.get('total_frames', 1)
                fc = vstats.get('frame_count', 0)
                st.metric("Avg FPS",         f"{vstats.get('avg_fps', 0):.1f}")
                st.metric("Frames Done",     fc)
                st.metric("Remaining",       max(0, tf - fc))
                st.metric("Total Detections",vstats.get('total_detections', 0))


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def render_tabs():
    """Render the two main tabs."""
    tab1, tab2 = st.tabs(["📷 Live Webcam", "🎬 Upload Video"])
    with tab1:
        _render_webcam_tab()
    with tab2:
        _render_video_tab()