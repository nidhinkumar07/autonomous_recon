# ui/components.py
import streamlit as st
import time
import cv2
import numpy as np

def load_css_from_file(file_path="styles.css"):
    """Load and inject CSS from external file"""
    try:
        with open(file_path, 'r') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.title("YOLOv8 Detection Dashboard")
        st.caption("Real-time object detection and tracking using YOLOv8. Upload a video file or use your webcam for detection with advanced object tracking.")
        st.warning("CSS file 'styles.css' not found. Using minimal styling.")

def display_title():
    """Display the main title"""
    st.markdown("""
    <div style="padding-top: 0; margin-top: 0;">
        <h1 style="font-size: 1.75rem; color: #1E3A8A; font-weight: 700; margin-bottom: 0.25rem; padding-top: 0;">
            YOLOv8 Detection Dashboard
        </h1>
        <div style='color: #94A3B8; font-weight: 400; line-height: 1.5; margin-bottom: 1.5rem;'>
            Real-time object detection and tracking using YOLOv8. Upload a video file or use your webcam for detection with advanced object tracking.
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with all settings"""
    with st.sidebar:
        st.markdown('<h2 style="margin-bottom: 0.75rem;">SETTINGS</h2>', unsafe_allow_html=True)
        
        # Detection classes section
        st.markdown('<div class="section-header" style="margin-top: 0.5rem;">Detection Classes</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.session_state.show_persons = st.checkbox("Persons", value=st.session_state.show_persons, key="sidebar_persons")
            st.session_state.show_vehicles = st.checkbox("Vehicles", value=st.session_state.show_vehicles, key="sidebar_vehicles")
        with col2:
            st.session_state.show_animals = st.checkbox("Animals", value=st.session_state.show_animals, key="sidebar_animals")
            st.session_state.show_everything = st.checkbox("All Classes", value=st.session_state.show_everything, key="sidebar_everything")
        
        st.markdown("---")
        
        # Visualization section
        st.markdown('<div class="section-header">Visualization</div>', unsafe_allow_html=True)
        
        st.session_state.show_labels = st.checkbox("Show Labels", value=st.session_state.show_labels, key="vis_labels")
        st.session_state.show_confidence = st.checkbox("Show Confidence Scores", value=st.session_state.show_confidence, key="vis_confidence")
        st.session_state.show_ids = st.checkbox("Show Object IDs", value=st.session_state.show_ids, key="vis_ids")
        
        st.markdown("---")
        
        # Configuration section
        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
        
        st.session_state.confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            help="Minimum confidence score for detections",
            key="conf_threshold"
        )
        
        st.session_state.frame_skip = st.slider(
            "Frame Skip",
            min_value=1,
            max_value=10,
            value=st.session_state.frame_skip,
            step=1,
            help="Process every nth frame",
            key="frame_skip_slider"
        )
        
        st.session_state.box_thickness = st.slider(
            "Box Thickness",
            min_value=1,
            max_value=5,
            value=st.session_state.box_thickness,
            step=1,
            help="Thickness of detection boxes",
            key="box_thickness_slider"
        )
        
        st.markdown("---")
        
        # Color legend
        st.markdown('<div class="section-header">Detection Colors</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #10B981; margin-right: 8px;"></div>
                <span style="font-size: 0.85rem;">Persons</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #3B82F6; margin-right: 8px;"></div>
                <span style="font-size: 0.85rem;">Vehicles</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #EF4444; margin-right: 8px;"></div>
                <span style="font-size: 0.85rem;">Animals</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #F59E0B; margin-right: 8px;"></div>
                <span style="font-size: 0.85rem;">Other</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_summary():
    """Display the comprehensive object tracking summary"""
    summary = st.session_state.object_tracker.generate_summary()
    
    if not summary["objects"]:
        st.info("No objects tracked yet. Start detection to see tracking data.")
        return
    
    st.markdown("### Comprehensive Object Tracking Summary")
    
    # Display by category
    for class_name, objects in summary["objects"].items():
        category_stats = summary["statistics"]["categories"][class_name]
        object_count = category_stats["count"]
        total_frames = category_stats["total_frames"]
        avg_frames = category_stats["avg_frames_per_object"]
        current_active = category_stats.get("current_active", 0)
        
        # Create a clean container for each class
        with st.container():
            st.markdown(f"**{class_name.capitalize()}** ({object_count} objects, {current_active} currently active)")
            
            # Use Streamlit's native columns and metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total objects", object_count)
            with col2:
                st.metric("Currently active", current_active)
            with col3:
                st.metric("Total frames", total_frames)
            with col4:
                st.metric("Avg frames/obj", f"{avg_frames:.1f}")
            
            # Show individual objects with details
            with st.expander(f"View {object_count} {class_name} objects", expanded=False):
                for obj in objects:
                    obj_id = obj["id"]
                    frame_count = obj["frame_count"]
                    avg_confidence = obj.get("avg_confidence", 0)
                    is_active = obj.get("is_active", False)
                    
                    with st.container():
                        st.markdown(f"**Object ID:** `{obj_id}`")
                        
                        detail_col1, detail_col2, detail_col3 = st.columns(3)
                        
                        with detail_col1:
                            st.markdown(f"**Frames detected:** {frame_count}")
                            st.markdown(f"**First frame:** {obj.get('first_frame', 'N/A')}")
                        with detail_col2:
                            st.markdown(f"**Avg confidence:** {avg_confidence:.2f}")
                            st.markdown(f"**Last frame:** {obj.get('last_frame', 'N/A')}")
                        with detail_col3:
                            status_text = "Active" if is_active else "Inactive"
                            st.markdown(f"**Status:** {status_text}")
                        
                        st.markdown("---")