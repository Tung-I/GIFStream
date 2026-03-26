import os
import cv2
import numpy as np
import pycolmap
import shutil
from argparse import ArgumentParser
from pathlib import Path

def extract_frames(video_path, output_dir, fps):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 1
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Save frames in 00001.png format
        cv2.imwrite(os.path.join(output_dir, f"{count:05d}.png"), frame)
        count += 1
    cap.release()

def run_pycolmap_triangulation(scene_path, frame_idx):
    """
    Replaces the 'colmap point_triangulator' and 'image_undistorter' shell calls
    """
    # The trainer expects /sparse/0/ in the root of the scene for the first frame
    output_path = Path(scene_path)
    sparse_path = output_path / "sparse" / "0"
    sparse_path.mkdir(parents=True, exist_ok=True)
    
    # Define paths (assuming you've already created input.db and manual/ images.txt)
    database_path = output_path / f"input_{frame_idx}.db"
    image_path = output_path / "input_images" # Temporary folder with GOP start frames
    
    # 1. Feature Extraction & Matching via pycolmap
    pycolmap.extract_features(database_path, image_path)
    pycolmap.match_exhaustive(database_path)
    
    # 2. Triangulation (equivalent to point_triangulator)
    # Note: pycolmap reconstruction requires a reference sparse model (your manual folder)
    reconstruction = pycolmap.Reconstruction(output_path / "manual")
    pycolmap.triangulate_points(reconstruction, database_path, image_path, sparse_path)
    
    # 3. Save model
    reconstruction.write(sparse_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Iterate through scenes (e.g., flame_steak)
    for scene_name in os.listdir(args.root_dir):
        scene_path = os.path.join(args.root_dir, scene_name)
        if not os.path.isdir(scene_path): continue
        
        print(f"Processing {scene_name}...")
        
        # 1. Extract Frames
        png_base = os.path.join(scene_path, "png")
        video_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".mp4")])
        
        for idx, v_file in enumerate(video_files):
            cam_dir = os.path.join(png_base, f"cam{idx:02d}")
            extract_frames(os.path.join(scene_path, v_file), cam_dir, args.fps)

        # 2. Setup directory for GIFStream trainer
        # Move one frame per camera into an 'images' folder (trainer expects this)
        final_image_dir = os.path.join(scene_path, "images")
        os.makedirs(final_image_dir, exist_ok=True)
        
        # Copy frame 00001 from all cams to images/camXX.png
        for cam_idx in range(len(video_files)):
            src = os.path.join(png_base, f"cam{cam_idx:02d}", "00001.png")
            dst = os.path.join(final_image_dir, f"cam{cam_idx:02d}.png")
            shutil.copy(src, dst)

        # 3. Final Step: Ensure the 'sparse/0' folder exists
        # To bypass your current error, manually create the path:
        target_sparse = os.path.join(scene_path, "sparse", "0")
        os.makedirs(target_sparse, exist_ok=True)
        
        # Run your existing convertdynerftocolmapdb logic here...
        # But change 'projectfolder' to scene_path so output goes to sparse/0