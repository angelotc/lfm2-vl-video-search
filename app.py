import streamlit as st
import tempfile
import os
from pathlib import Path

# Configure the Streamlit page
st.set_page_config(
    page_title="Video Upload App",
    page_icon="🎥",
    layout="wide"
)

def main():
    st.title("🎥 Video Upload Application")
    st.write("Upload a video file to get started")

    # Create a file uploader widget
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
        help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM"
    )

    if uploaded_file is not None:
        # Display video information
        st.success(f"Video uploaded successfully: {uploaded_file.name}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Video Information")
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
                "File type": uploaded_file.type
            }

            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")

        with col2:
            st.subheader("Video Preview")
            # Display the video
            st.video(uploaded_file)

        # Option to save the video
        st.subheader("Save Video")

        if st.button("Save Video to Disk"):
            # Create uploads directory if it doesn't exist
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)

            # Save the file
            save_path = upload_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Video saved to: {save_path.absolute()}")

        # Additional options
        with st.expander("Advanced Options"):
            st.write("**Video Processing Options**")
            st.checkbox("Extract frames")
            st.checkbox("Generate thumbnail")
            st.checkbox("Extract audio")
            st.slider("Quality", 0, 100, 80)

    else:
        st.info("👆 Please upload a video file to begin")

if __name__ == "__main__":
    main()
