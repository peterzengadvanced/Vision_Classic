import os
import glob
import cv2
import numpy as np

# Input raw images path
RAW_IMG_PATH = "C:/Users/Peter Zeng/Desktop/vtry/rawimages/"

# Output paths
PREPROCESSED_IMG_PATH = "C:/Users/Peter Zeng/Desktop/vtry/preprocessed/"
EDGEDETECTED_IMG_PATH = "C:/Users/Peter Zeng/Desktop/vtry/edgedetected/"
SEGMENTED_IMG_PATH = "C:/Users/Peter Zeng/Desktop/vtry/segmented/"
OBJECTDETECTED_IMG_PATH = "C:/Users/Peter Zeng/Desktop/vtry/objectdetected/"
RECOGNIZED_IMG_PATH = "C:/Users/Peter Zeng/Desktop/vtry/recognized/"

# Feature extraction output paths
POINT_FEATURES_PATH = "C:/Users/Peter Zeng/Desktop/vtry/features/point/"
LINE_FEATURES_PATH = "C:/Users/Peter Zeng/Desktop/vtry/features/line/"
LOCAL_REGION_FEATURES_PATH = "C:/Users/Peter Zeng/Desktop/vtry/features/localregion/"

# Pose output path (e.g., saving text or CSV with 2D pose info)
POSE_OUTPUT_PATH = "C:/Users/Peter Zeng/Desktop/vtry/pose/"

# Create necessary directories if they don't exist
os.makedirs(PREPROCESSED_IMG_PATH, exist_ok=True)
os.makedirs(EDGEDETECTED_IMG_PATH, exist_ok=True)
os.makedirs(SEGMENTED_IMG_PATH, exist_ok=True)
os.makedirs(OBJECTDETECTED_IMG_PATH, exist_ok=True)
os.makedirs(RECOGNIZED_IMG_PATH, exist_ok=True)
os.makedirs(POINT_FEATURES_PATH, exist_ok=True)
os.makedirs(LINE_FEATURES_PATH, exist_ok=True)
os.makedirs(LOCAL_REGION_FEATURES_PATH, exist_ok=True)
os.makedirs(POSE_OUTPUT_PATH, exist_ok=True)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def edge_detection(img_gray):
    edges = cv2.Canny(img_gray, 50, 150)
    return edges

def segment_roi(img_color):
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    
    # Define thresholds for red color in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    
    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    return red_mask

def detect_object(img_color, red_mask):
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None, None, None
    
    # Choose the largest contour by area
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # For a more precise shape + orientation
    min_area_rect = cv2.minAreaRect(c)  
    
    return (x, y, w, h), min_area_rect, c, contours

def recognize_object(img_color, box_roi):
    recognized_label = "Box Cover with Snap-Fit"
    return recognized_label

def extract_features(img_color, mask):
    # 1.  Point feature
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(img_color, mask)
    
    # 2. line feature
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(mask)[0]  # returns (N,1,4) array of lines
    
    # 3. Local region 
    region_des = des
    
    return kp, des, lines, region_des

def match_features(des1, des2):
    """
    Simple version of feature matching for this step
    """
    if des1 is None or des2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def compute_2d_pose(min_area_rect):
    """
    2D pose
    """
    (cx, cy), (w, h), angle = min_area_rect
    
    # angle to radians
    angle_rad = np.deg2rad(angle)
    
    # 2D rotation matrix
    T = np.eye(3, dtype=np.float32)
    cosA = np.cos(angle_rad)
    sinA = np.sin(angle_rad)
    
    # Rotation around the center
    # translate center to origin
    T_center_to_origin = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0,  1]
    ], dtype=np.float32)
    
    # rotate
    R = np.array([
        [cosA, -sinA, 0],
        [sinA,  cosA, 0],
        [0,     0,    1]
    ], dtype=np.float32)
    
    # translate back from origin to center
    T_origin_to_center = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Final
    T_2d = T_origin_to_center @ R @ T_center_to_origin
    
    # return parameters and transform
    pose_info = {
        "center_x": float(cx),
        "center_y": float(cy),
        "width": float(w),
        "height": float(h),
        "angle_deg": float(angle)
    }
    
    return pose_info, T_2d

def main():
    # read all images
    img_files = (glob.glob(os.path.join(RAW_IMG_PATH, "*.jpg")) +
                 glob.glob(os.path.join(RAW_IMG_PATH, "*.png")) +
                 glob.glob(os.path.join(RAW_IMG_PATH, "*.bmp")))

    if not img_files:
        print("No input images found in:", RAW_IMG_PATH)
        return

    for img_path in img_files:
        img_color = cv2.imread(img_path)
        if img_color is None:
            print(f"Failed to read image {img_path}")
            continue
        
        basename = os.path.basename(img_path)
        filename, ext = os.path.splitext(basename)
        
        img_preprocessed = preprocess_image(img_color)
        # Save the preprocessed image
        out_preprocessed = os.path.join(PREPROCESSED_IMG_PATH, f"{filename}_preprocessed{ext}")
        cv2.imwrite(out_preprocessed, img_preprocessed)
        
        edges = edge_detection(img_preprocessed)
        out_edged = os.path.join(EDGEDETECTED_IMG_PATH, f"{filename}_edges{ext}")
        cv2.imwrite(out_edged, edges)
        
        red_mask = segment_roi(img_color)
        out_segmented = os.path.join(SEGMENTED_IMG_PATH, f"{filename}_segmented{ext}")
        cv2.imwrite(out_segmented, red_mask)
        
        box_rect, min_area_rect, largest_contour, all_contours = detect_object(img_color, red_mask)
        if box_rect is None:
            print(f"No box cover found in {img_path}")
            continue
        
        (x, y, w, h) = box_rect
        
        # Draw
        object_detected_img = img_color.copy()
        cv2.rectangle(object_detected_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        box_points = cv2.boxPoints(min_area_rect)  # 4 corner points
        box_points = box_points.astype(int)
        cv2.drawContours(object_detected_img, [box_points], 0, (255, 0, 0), 2)
        
        out_detected = os.path.join(OBJECTDETECTED_IMG_PATH, f"{filename}_detected{ext}")
        cv2.imwrite(out_detected, object_detected_img)
        
        # Object Recognition
        box_roi = img_color[y:y+h, x:x+w]
        recognized_label = recognize_object(img_color, box_roi)
        recognized_img = img_color.copy()
        cv2.putText(recognized_img, recognized_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        out_recognized = os.path.join(RECOGNIZED_IMG_PATH, f"{filename}_recognized{ext}")
        cv2.imwrite(out_recognized, recognized_img)
        
        # Feature Extraction
        box_region = img_color[y:y+h, x:x+w]
        mask_region = red_mask[y:y+h, x:x+w]
        
        kp_box, des_box, lines_box, region_des_box = extract_features(box_region, mask_region)
        
        # point features
        feat_point_img = cv2.drawKeypoints(box_region, kp_box, None, color=(255, 0, 0))
        out_points = os.path.join(POINT_FEATURES_PATH, f"{filename}_point_features{ext}")
        cv2.imwrite(out_points, feat_point_img)
        
        # line features
        feat_line_img = box_region.copy()
        lsd = cv2.createLineSegmentDetector(0)
        if lines_box is not None:
            drawn_lines = lsd.drawSegments(feat_line_img, lines_box)
            out_lines = os.path.join(LINE_FEATURES_PATH, f"{filename}_line_features{ext}")
            cv2.imwrite(out_lines, drawn_lines)
        
        # Local region
        out_local = os.path.join(LOCAL_REGION_FEATURES_PATH, f"{filename}_local_features{ext}")
        cv2.imwrite(out_local, feat_point_img)  # placeholder
        
        # Feature matching simple
        ref_des = des_box
        matches = match_features(des_box, ref_des)
        
        matched_vis = None
        if matches:
            matched_vis = cv2.drawMatches(box_region, kp_box,
                                          box_region, kp_box,
                                          matches[:50], None, flags=2)
        out_matched = os.path.join(OBJECTDETECTED_IMG_PATH, f"{filename}_matched{ext}")
        if matched_vis is not None:
            cv2.imwrite(out_matched, matched_vis)
        
        # 2D Pose Measurement
        pose_info, T_2d = compute_2d_pose(min_area_rect)
        
        pose_output_file = os.path.join(POSE_OUTPUT_PATH, f"{filename}_pose.txt")
        with open(pose_output_file, 'w') as f:
            f.write("2D Pose from minAreaRect\n")
            f.write(f"Center: ({pose_info['center_x']:.3f}, {pose_info['center_y']:.3f})\n")
            f.write(f"Width: {pose_info['width']:.3f}\n")
            f.write(f"Height: {pose_info['height']:.3f}\n")
            f.write(f"Angle (deg): {pose_info['angle_deg']:.3f}\n\n")
            f.write("2D Homogeneous Transform (3x3):\n")
            f.write(str(T_2d) + "\n")
        
        print(f"Processed {img_path} successfully. Pose saved to {pose_output_file}")

if __name__ == "__main__":
    main()
