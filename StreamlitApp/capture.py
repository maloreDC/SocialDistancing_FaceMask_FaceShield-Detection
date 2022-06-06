import threading
from typing import Union
import av
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import os 
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def main():
    class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
        
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            out_image = frame.to_ndarray(format="bgr24")
            print(out_image)
            h,w,c = out_image.shape
            print(h)
            print(w)
            with self.frame_lock:
                
                self.out_image = out_image
            return out_image

    ctx = webrtc_streamer(
            key="snapshot", 
            video_transformer_factory=VideoTransformer, 
            media_stream_constraints={
                "video": True,
            },
        )

    if ctx.video_transformer:

        snap = st.button("Snapshot")
        if snap:
            with ctx.video_transformer.frame_lock:
                out_image = ctx.video_transformer.out_image

            if out_image is not None:
                
                st.write("Output image:")
                st.image(out_image, channels="BGR")
                # my_path = os.path.abspath(os.path.dirname(__file__))
                my_path = "C:/Users/peysi/Downloads/ThesisApp/Final-App/ReferenceImages"       
                cv2.imwrite(os.path.join(my_path, "reference.jpg"), out_image)

            else:
                st.warning("No frames available yet.")


if __name__ == "__main__":
    main()