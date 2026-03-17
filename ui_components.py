import json
import streamlit as st


def load_css_from_file(file_path: str = "styles.css"):
    """Load and inject CSS from external file."""
    try:
        with open(file_path, 'r') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file 'styles.css' not found. Using minimal styling.")


def display_summary():
    """
    Display the object tracking summary.
    Uses tracker's cached summary — only rebuilt when state changed (fix #11).
    avg_confidence is now read from the stored running mean (fix #1).
    """
    summary = st.session_state.object_tracker.generate_summary()

    if not summary["objects"]:
        st.info("No objects tracked yet. Start detection to see tracking data.")
        return

    st.markdown("### Object Tracking Summary")

    total_objects = summary["statistics"]["total_objects"]

    for class_name, objects in summary["objects"].items():
        cat_stats      = summary["statistics"]["categories"][class_name]
        obj_count      = cat_stats["count"]
        total_frames   = cat_stats["total_frames"]
        avg_frames     = cat_stats["avg_frames_per_object"]
        current_active = cat_stats.get("current_active", 0)

        with st.container():
            st.markdown(
                f"**{class_name.capitalize()}** "
                f"({obj_count} objects, {current_active} currently active)"
            )
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1: st.metric("Total objects",    obj_count)
            with mc2: st.metric("Currently active", current_active,
                                delta=current_active - 0)      # fix #21 delta
            with mc3: st.metric("Total frames",     total_frames)
            with mc4: st.metric("Avg frames/obj",   f"{avg_frames:.1f}")

            with st.expander(f"View {obj_count} {class_name} object(s)", expanded=False):
                for obj in objects:
                    obj_id     = obj["id"]
                    fc         = obj["frame_count"]
                    avg_conf   = obj.get("avg_confidence", 0.0)
                    is_active  = obj.get("is_active", False)

                    with st.container():
                        st.markdown(f"**Object ID:** `{obj_id}`")
                        d1, d2, d3 = st.columns(3)
                        with d1:
                            st.markdown(f"**Frames detected:** {fc}")
                            st.markdown(f"**First frame:** {obj.get('first_frame', 'N/A')}")
                        with d2:
                            st.markdown(f"**Avg confidence:** {avg_conf:.2f}")
                            st.markdown(f"**Last frame:** {obj.get('last_frame', 'N/A')}")
                        with d3:
                            badge_color = "#10B981" if is_active else "#64748B"
                            badge_text  = "Active" if is_active else "Inactive"
                            st.markdown(
                                f'<span style="background:{badge_color};color:#fff;'
                                f'padding:2px 8px;border-radius:4px;font-size:0.78rem;">'
                                f'{badge_text}</span>',
                                unsafe_allow_html=True,
                            )
                        st.markdown("---")


def render_export_buttons():
    """
    Render CSV and JSON export buttons for the tracking summary (fix #13).
    Only shown when there is data to export.
    """
    tracker = st.session_state.object_tracker
    if not tracker.object_instances:
        return

    export_data = tracker.export_to_dict()

    col_csv, col_json = st.columns(2)

    with col_csv:
        # Build CSV string in-memory
        lines = ["object_id,class,frame_count,first_frame,last_frame,"
                 "avg_confidence,is_active"]
        for row in export_data["objects"]:
            lines.append(
                f"{row['object_id']},{row['class']},{row['frame_count']},"
                f"{row['first_frame']},{row['last_frame']},"
                f"{row['avg_confidence']},{row['is_active']}"
            )
        csv_str = "\n".join(lines)
        st.download_button(
            label="⬇ Download CSV",
            data=csv_str,
            file_name="tracking_results.csv",
            mime="text/csv",
            key="download_csv",
        )

    with col_json:
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="⬇ Download JSON",
            data=json_str,
            file_name="tracking_results.json",
            mime="application/json",
            key="download_json",
        )
