import streamlit as st
from session_state import YOLO_MODELS
from detection import load_yolo_model


def render_sidebar():
    """Render the sidebar configuration panel."""
    with st.sidebar:
        # ── App title ──────────────────────────────────────────────────────
        st.markdown('<h2 style="margin-bottom:0.75rem;">SETTINGS</h2>',
                    unsafe_allow_html=True)

        # ── Model-ready badge (fix #18) ────────────────────────────────────
        if st.session_state.get('model_ready', False):
            st.markdown(
                '<div style="display:flex;align-items:center;gap:6px;'
                'margin-bottom:0.75rem;">'
                '<span style="width:10px;height:10px;border-radius:50%;'
                'background:#10B981;display:inline-block;"></span>'
                '<span style="font-size:0.78rem;color:#10B981;">Model ready</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:6px;'
                'margin-bottom:0.75rem;">'
                '<span style="width:10px;height:10px;border-radius:50%;'
                'background:#F59E0B;display:inline-block;"></span>'
                '<span style="font-size:0.78rem;color:#F59E0B;">Loading model…</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Model selector (fix #14) ───────────────────────────────────────
        st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)
        model_labels = list(YOLO_MODELS.keys())
        selected_label = st.selectbox(
            "YOLO Model",
            options=model_labels,
            index=model_labels.index(st.session_state.selected_model_label),
            key="model_selector",
            help="n=fastest, x=most accurate. Changing reloads the model.",
            label_visibility="collapsed",
        )
        if selected_label != st.session_state.selected_model_label:
            st.session_state.selected_model_label = selected_label
            model_file = YOLO_MODELS[selected_label]
            with st.spinner(f"Loading {selected_label}…"):
                model, err = load_yolo_model(model_file)
                st.session_state.current_model = model
                st.session_state.model_ready   = model is not None
                st.session_state.model_missing = model is None
                st.session_state.model_error   = err
            st.rerun()

        st.markdown("---")

        # ── Detection classes ──────────────────────────────────────────────
        st.markdown('<div class="section-header">Detection Classes</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.session_state.show_persons  = st.checkbox("Persons",    value=st.session_state.show_persons,  key="sidebar_persons")
            st.session_state.show_vehicles = st.checkbox("Vehicles",   value=st.session_state.show_vehicles, key="sidebar_vehicles")
        with col2:
            st.session_state.show_animals  = st.checkbox("Animals",    value=st.session_state.show_animals,  key="sidebar_animals")
            st.session_state.show_everything = st.checkbox("All Classes", value=st.session_state.show_everything, key="sidebar_everything")

        st.markdown("---")

        # ── Visualization ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Visualization</div>',
                    unsafe_allow_html=True)
        st.session_state.show_labels     = st.checkbox("Show Labels",           value=st.session_state.show_labels,     key="vis_labels")
        st.session_state.show_confidence = st.checkbox("Show Confidence Scores", value=st.session_state.show_confidence, key="vis_confidence")
        st.session_state.show_ids        = st.checkbox("Show Object IDs",        value=st.session_state.show_ids,        key="vis_ids")

        st.markdown("---")

        # ── Configuration ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Configuration</div>',
                    unsafe_allow_html=True)

        st.session_state.confidence_threshold = st.slider(
            "Confidence Threshold", 0.1, 1.0,
            value=st.session_state.confidence_threshold, step=0.05,
            help="Minimum confidence score for detections",
            key="conf_threshold",
        )
        st.session_state.frame_skip = st.slider(
            "Frame Skip", 1, 10,
            value=st.session_state.frame_skip, step=1,
            help="Process every nth frame (higher = faster but choppier)",
            key="frame_skip_slider",
        )
        st.session_state.box_thickness = st.slider(
            "Box Thickness", 1, 5,
            value=st.session_state.box_thickness, step=1,
            help="Thickness of detection boxes",
            key="box_thickness_slider",
        )

        st.markdown("---")

        # ── Webcam resolution (fix #15) ────────────────────────────────────
        st.markdown('<div class="section-header">Webcam Resolution</div>',
                    unsafe_allow_html=True)
        res_options = {"640×480 (default)": (640, 480),
                       "1280×720 (HD)":     (1280, 720),
                       "320×240 (low)":     (320, 240)}
        res_key = st.selectbox(
            "Resolution", list(res_options.keys()),
            index=0, key="webcam_res_select", label_visibility="collapsed",
        )
        st.session_state.webcam_width, st.session_state.webcam_height = res_options[res_key]

        st.markdown("---")

        # ── Colour legend ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Detection Colors</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom:0.75rem;">
            <div style="display:flex;align-items:center;margin-bottom:0.25rem;">
                <div style="width:12px;height:12px;border-radius:50%;background:#10B981;margin-right:8px;"></div>
                <span style="font-size:0.85rem;">Persons</span>
            </div>
            <div style="display:flex;align-items:center;margin-bottom:0.25rem;">
                <div style="width:12px;height:12px;border-radius:50%;background:#3B82F6;margin-right:8px;"></div>
                <span style="font-size:0.85rem;">Vehicles</span>
            </div>
            <div style="display:flex;align-items:center;margin-bottom:0.25rem;">
                <div style="width:12px;height:12px;border-radius:50%;background:#EF4444;margin-right:8px;"></div>
                <span style="font-size:0.85rem;">Animals</span>
            </div>
            <div style="display:flex;align-items:center;">
                <div style="width:12px;height:12px;border-radius:50%;background:#F59E0B;margin-right:8px;"></div>
                <span style="font-size:0.85rem;">Other</span>
            </div>
        </div>
        """, unsafe_allow_html=True)