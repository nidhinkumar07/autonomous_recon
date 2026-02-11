# main.py - Professional SaaS UI with modular architecture
import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import warnings
import os
import atexit
import gc
import uuid

# Suppress warnings
warnings.filterwarnings('ignore')

# ========== FIX: Streamlit Media File Cache Issues ==========
# Disable Streamlit's file watcher to prevent media cache issues
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'

# Cleanup function for media files
def cleanup_streamlit_media():
    """Force cleanup of Streamlit media cache"""
    try:
        from streamlit.runtime import get_instance
        runtime = get_instance()
        if runtime and hasattr(runtime, '_media_file_storage'):
            # Clear all stored media files
            runtime._media_file_storage._files_by_id.clear()
            runtime._media_file_storage._files_by_name.clear()
            # Force garbage collection
            gc.collect()
    except:
        pass

# Register cleanup on exit
atexit.register(cleanup_streamlit_media)

# Safe image display function - UPDATED to use width='stretch' (new Streamlit parameter)
def safe_display_image(placeholder, image, caption=None, channels="RGB"):
    """Safely display image with error recovery - uses width='stretch' for container width"""
    try:
        if caption:
            placeholder.image(image, channels=channels, caption=caption, width='stretch')
        else:
            placeholder.image(image, channels=channels, width='stretch')
    except Exception as e:
        if "MediaFileStorageError" in str(e) or "Bad filename" in str(e):
            # Recreate the placeholder on error
            placeholder.empty()
            time.sleep(0.05)
            try:
                if caption:
                    placeholder.image(image, channels=channels, caption=caption, width='stretch')
                else:
                    placeholder.image(image, channels=channels, width='stretch')
            except:
                pass
        else:
            raise

# ========== End of Media File Fixes ==========

# Import modules
from config.classes import COCO_CLASSES, VEHICLE_KEYWORDS, ANIMAL_KEYWORDS
from core.tracker import ObjectTracker
from core.detector import load_yolo_model, detect_with_yolo, filter_detections
from ui.draw_utils import draw_detections
from ui.components import load_css_from_file, display_title, display_sidebar, display_summary

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="YOLOv8 Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Load CSS
load_css_from_file()

# Display title
display_title()

# Initialize session state for detection settings
if 'show_persons' not in st.session_state:
    st.session_state.show_persons = True
if 'show_vehicles' not in st.session_state:
    st.session_state.show_vehicles = True
if 'show_animals' not in st.session_state:
    st.session_state.show_animals = False
if 'show_everything' not in st.session_state:
    st.session_state.show_everything = False
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.5
if 'frame_skip' not in st.session_state:
    st.session_state.frame_skip = 2
if 'box_thickness' not in st.session_state:
    st.session_state.box_thickness = 2
if 'show_labels' not in st.session_state:
    st.session_state.show_labels = True
if 'show_confidence' not in st.session_state:
    st.session_state.show_confidence = True
if 'show_ids' not in st.session_state:
    st.session_state.show_ids = True

# Initialize tracker in session state
if 'object_tracker' not in st.session_state:
    st.session_state.object_tracker = ObjectTracker()

# Display sidebar with all settings
display_sidebar()

# Initialize session state for processing
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'video_processing' not in st.session_state:
    st.session_state.video_processing = False
if 'detection_counter' not in st.session_state:
    st.session_state.detection_counter = 0
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "webcam"
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'webcam_stats' not in st.session_state:
    st.session_state.webcam_stats = {}
if 'video_stats' not in st.session_state:
    st.session_state.video_stats = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'webcam_cap' not in st.session_state:
    st.session_state.webcam_cap = None
if 'video_cap' not in st.session_state:
    st.session_state.video_cap = None
if 'cleanup_done' not in st.session_state:
    cleanup_streamlit_media()
    st.session_state.cleanup_done = True

# Main tabs - Clean underline system
tab1, tab2 = st.tabs(["Live Webcam", "Upload Video"])

# Tab 1: Live Webcam
with tab1:
    st.markdown("## Live Webcam Detection")
    
    # Store current tab
    st.session_state.active_tab = "webcam"
    
    # Use consistent 1.5rem gap between columns
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        # Check if webcam is active
        if st.session_state.webcam_active:
            st.markdown('<div class="processing-card">Webcam is active. Objects are being tracked with unique IDs.</div>', unsafe_allow_html=True)
            
            # Create a unique container for webcam to prevent cache issues
            webcam_container = st.container()
            webcam_placeholder = webcam_container.empty()
            webcam_stats_placeholder = st.empty()
            
            # Control button
            if st.button("🛑 Stop Webcam", key="stop_webcam_btn", type="secondary"):
                st.session_state.webcam_active = False
                # Release webcam if exists
                if st.session_state.webcam_cap is not None:
                    st.session_state.webcam_cap.release()
                    st.session_state.webcam_cap = None
                webcam_placeholder.empty()
                webcam_container.empty()
                # Clear media cache
                cleanup_streamlit_media()
                st.rerun()
            
            # Webcam processing logic
            try:
                # Initialize webcam with error suppression
                if st.session_state.webcam_cap is None:
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    st.session_state.webcam_cap = cap
                else:
                    cap = st.session_state.webcam_cap
                
                if not cap.isOpened():
                    st.error("Could not access webcam. Please check if webcam is connected.")
                    st.session_state.webcam_active = False
                    st.rerun()
                
                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Load model if not already loaded
                if st.session_state.current_model is None:
                    with st.spinner("Loading detection model..."):
                        st.session_state.current_model = load_yolo_model()
                
                # Initialize stats if needed
                if 'webcam_start_time' not in st.session_state:
                    st.session_state.webcam_start_time = time.time()
                    st.session_state.webcam_frame_count = 0
                    st.session_state.webcam_detected_objects = 0
                    st.session_state.webcam_processing_time = 0
                
                frame_count = 0
                frame_skip_counter = 0
                
                # Process webcam frames
                while st.session_state.webcam_active:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.warning("Failed to capture frame from webcam")
                        time.sleep(0.1)
                        continue
                    
                    frame_count += 1
                    st.session_state.webcam_frame_count += 1
                    frame_skip_counter += 1
                    
                    # Skip frames for performance
                    if frame_skip_counter % st.session_state.frame_skip != 0:
                        continue
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Perform detection
                    start_time = time.time()
                    raw_detections = detect_with_yolo(st.session_state.current_model, frame_rgb)
                    processing_time = time.time() - start_time
                    
                    # Filter detections
                    filtered_detections = filter_detections(raw_detections)
                    
                    # Update object tracking - get the full tracking data
                    tracking_data = st.session_state.object_tracker.update_tracking(
                        filtered_detections, 
                        frame_count
                    )
                    
                    # Update statistics
                    st.session_state.webcam_detected_objects += len(filtered_detections)
                    st.session_state.webcam_processing_time += processing_time
                    
                    # Draw detections on frame with object IDs
                    frame_with_detections, detection_count, current_counts = draw_detections(
                        frame_rgb, 
                        tracking_data, 
                        frame_count, 
                        "webcam"
                    )
                    
                    # Update session state
                    st.session_state.current_frame = frame_with_detections
                    st.session_state.detection_counter = detection_count
                    
                    # FIX: Use safe display function with width='stretch' (new Streamlit parameter)
                    safe_display_image(
                        webcam_placeholder, 
                        frame_with_detections, 
                        caption=f"Frame {frame_count} - Tracking {len(tracking_data['active_objects'])} objects",
                        channels="RGB"
                    )
                    
                    # Update statistics display with responsive grid
                    with webcam_stats_placeholder.container():
                        fps_value = 1.0 / processing_time if processing_time > 0 else 0
                        
                        st.markdown("## Live Statistics")
                        
                        # Responsive metrics grid using CSS classes
                        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                        
                        # Create metrics columns
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        with metrics_col1:
                            st.metric("Frames", st.session_state.webcam_frame_count)
                        with metrics_col2:
                            st.metric("Current Detections", detection_count)
                        with metrics_col3:
                            st.metric("Total Detections", st.session_state.webcam_detected_objects)
                        with metrics_col4:
                            st.metric("FPS", f"{fps_value:.1f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                         
                        # Show tracking summary
                        display_summary()
                    
                    # Small delay to prevent overwhelming the UI
                    time.sleep(0.01)
                
            except Exception as e:
                st.error(f"Error in webcam processing: {str(e)}")
                st.session_state.webcam_active = False
                if st.session_state.webcam_cap is not None:
                    st.session_state.webcam_cap.release()
                    st.session_state.webcam_cap = None
                st.rerun()
        
        else:
            # Show start button
            if st.button("🎥 Start Webcam Detection", key="start_webcam_main", type="primary"):
                # Clean up any existing webcam
                if st.session_state.webcam_cap is not None:
                    st.session_state.webcam_cap.release()
                    st.session_state.webcam_cap = None
                # Clear media cache
                cleanup_streamlit_media()
                st.session_state.webcam_active = True
                st.session_state.video_processing = False
                # Reset tracker when starting
                st.session_state.object_tracker.reset()
                st.rerun()
    
    with col2:
        pass

# Tab 2: Upload Video
with tab2:
    st.markdown("## Upload Video for Detection")
    
    # Store current tab
    st.session_state.active_tab = "video"
    
    # Video uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'mpg', 'mpeg', 'wmv'],
        help="Upload a video file for detection analysis",
        key="video_uploader_main"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        st.session_state.video_path = video_path
        st.session_state.uploaded_file = uploaded_file
        
        # Use consistent 1.5rem gap between columns
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            # Show video preview if not processing
            if not st.session_state.video_processing:
                st.video(uploaded_file)
            
            # Check if video processing is active
            if st.session_state.video_processing:
                st.markdown('<div class="processing-card">Video processing is active. Objects are being tracked with unique IDs.</div>', unsafe_allow_html=True)
                
                # Create placeholders for video with unique container
                video_container = st.container()
                video_placeholder = video_container.empty()
                video_stats_placeholder = st.empty()
                video_progress_placeholder = st.empty()
                
                # Control button
                if st.button("🛑 Stop Processing", key="stop_video_btn", type="secondary"):
                    st.session_state.video_processing = False
                    # Release video capture if exists
                    if st.session_state.video_cap is not None:
                        st.session_state.video_cap.release()
                        st.session_state.video_cap = None
                    video_placeholder.empty()
                    video_container.empty()
                    # Clear media cache
                    cleanup_streamlit_media()
                    st.rerun()
                
                # Video processing logic
                try:
                    # Initialize video capture
                    if st.session_state.video_cap is None:
                        cap = cv2.VideoCapture(video_path)
                        st.session_state.video_cap = cap
                    else:
                        cap = st.session_state.video_cap
                    
                    if not cap.isOpened():
                        st.error("Could not open video file. The file might be corrupted.")
                        st.session_state.video_processing = False
                        st.rerun()
                    
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if fps <= 0 or total_frames <= 0:
                        st.error("Invalid video file. Could not read video properties.")
                        cap.release()
                        st.session_state.video_cap = None
                        st.session_state.video_processing = False
                        st.rerun()
                    
                    # Load model if not already loaded
                    if st.session_state.current_model is None:
                        with st.spinner("Loading detection model..."):
                            st.session_state.current_model = load_yolo_model()
                    
                    # Initialize stats if needed
                    if 'video_start_time' not in st.session_state:
                        st.session_state.video_start_time = time.time()
                        st.session_state.video_frame_count = 0
                        st.session_state.video_detected_objects = 0
                        st.session_state.video_processing_time = 0
                    
                    frame_count = 0
                    frame_skip_counter = 0
                    
                    # Process video frames
                    while st.session_state.video_processing and cap.isOpened():
                        ret, frame = cap.read()
                        
                        if not ret:
                            break
                        
                        frame_count += 1
                        st.session_state.video_frame_count += 1
                        frame_skip_counter += 1
                        
                        # Update progress
                        if total_frames > 0:
                            progress = frame_count / total_frames
                            video_progress_placeholder.progress(progress)
                        
                        # Skip frames for performance
                        if frame_skip_counter % st.session_state.frame_skip != 0:
                            continue
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Perform detection
                        start_time = time.time()
                        raw_detections = detect_with_yolo(st.session_state.current_model, frame_rgb)
                        processing_time = time.time() - start_time
                        
                        # Filter detections
                        filtered_detections = filter_detections(raw_detections)
                        
                        # Update object tracking - get the full tracking data
                        tracking_data = st.session_state.object_tracker.update_tracking(
                            filtered_detections, 
                            frame_count
                        )
                        
                        # Update statistics
                        st.session_state.video_detected_objects += len(filtered_detections)
                        st.session_state.video_processing_time += processing_time
                        
                        # Draw detections on frame with object IDs
                        frame_with_detections, detection_count, current_counts = draw_detections(
                            frame_rgb, 
                            tracking_data, 
                            frame_count, 
                            "video"
                        )
                        
                        # Update session state
                        st.session_state.current_frame = frame_with_detections
                        st.session_state.detection_counter = detection_count
                        
                        # FIX: Periodically refresh placeholder to prevent cache buildup
                        if frame_count % 30 == 0:  # Every 30 frames
                            video_placeholder.empty()
                            video_placeholder = video_container.empty()
                        
                        # FIX: Use safe display function with width='stretch' (new Streamlit parameter)
                        safe_display_image(
                            video_placeholder, 
                            frame_with_detections, 
                            caption=f"Frame {frame_count}/{total_frames} - Tracking {len(tracking_data['active_objects'])} objects",
                            channels="RGB"
                        )
                        
                        # Update statistics display with responsive grid
                        with video_stats_placeholder.container():
                            fps_value = 1.0 / processing_time if processing_time > 0 else 0
                            
                            st.markdown("#### Processing Statistics")
                            
                            # Responsive metrics grid
                            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                            
                            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                            with metrics_col1:
                                if total_frames > 0:
                                    progress_percent = (frame_count / total_frames) * 100
                                    st.metric("Progress", f"{progress_percent:.1f}%")
                                else:
                                    st.metric("Frames", frame_count)
                            with metrics_col2:
                                st.metric("Current Detections", detection_count)
                            with metrics_col3:
                                st.metric("Total Detections", st.session_state.video_detected_objects)
                            with metrics_col4:
                                st.metric("FPS", f"{fps_value:.1f}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show tracking summary
                            display_summary()
                        
                        # Check if stop button was pressed
                        if not st.session_state.video_processing:
                            break
                    
                    # Release video capture when done
                    cap.release()
                    st.session_state.video_cap = None
                    
                    # Clear video-specific stats
                    st.session_state.video_frame_count = 0
                    st.session_state.video_detected_objects = 0
                    st.session_state.video_processing_time = 0
                    
                    st.success(f"✅ Video processing completed! Processed {frame_count} frames.")
                    
                except Exception as e:
                    st.error(f"Error in video processing: {str(e)}")
                    st.session_state.video_processing = False
                    if st.session_state.video_cap is not None:
                        st.session_state.video_cap.release()
                        st.session_state.video_cap = None
                    st.rerun()
            
            else:
                # Show start button
                if st.button("▶️ Start Video Processing", key="start_video_main", type="primary"):
                    # Clean up any existing video capture
                    if st.session_state.video_cap is not None:
                        st.session_state.video_cap.release()
                        st.session_state.video_cap = None
                    # Clear media cache
                    cleanup_streamlit_media()
                    st.session_state.video_processing = True
                    st.session_state.webcam_active = False
                    # Reset tracker when starting
                    st.session_state.object_tracker.reset()
                    st.rerun()
        
        with col2:
            # Only show video info and instructions when NOT processing
            if not st.session_state.video_processing:
                st.markdown("#### Video Information")
                
                try:
                    # Display video info
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        with st.container():
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.metric("Frame Rate", f"{fps:.1f} FPS")
                                st.metric("Total Frames", f"{frame_count:,}")
                            with col_info2:
                                st.metric("Resolution", f"{width}×{height}")
                                st.metric("Duration", f"{duration:.1f}s")
                        cap.release()
                    else:
                        st.warning("Could not read video properties")
                except:
                    st.warning("Could not read video properties")
                
                st.markdown("""
                    <div class="info-card">
                        <strong style="color: #94A3B8;">📋 Steps:</strong><br>
                        <span style="color: #94A3B8;">1. ✓ Video uploaded successfully</span><br>
                        <span style="color: #94A3B8;">2. ▶️ Click Start Video Processing</span><br>
                        <span style="color: #94A3B8;">3. 🔍 Objects are tracked with unique IDs</span><br>
                        <span style="color: #94A3B8;">4. 📊 View real-time results</span><br>
                        <span style="color: #94A3B8;">5. 🛑 Click Stop Processing to pause</span>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Clear video path if no file is uploaded
        st.session_state.video_path = None
        st.session_state.uploaded_file = None
        st.info("📹 Please upload a video file to start detection.")