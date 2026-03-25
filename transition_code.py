"""
RGB Color Detection and Coordinate Mapping Script (WebODM Orthophoto Support)
Aa code WebODM mathi baneli RGB Orthophoto (GeoTIFF) mathi specific color (jem ke Yellow) detect karse ane ena coordinates JSON ma save karse.
Blocks: 0.3m x 0.3m
"""

import cv2
import numpy as np
import json
import os
from osgeo import gdal

# --- CONFIGURATION ---

# GeoTIFF Image path (WebODM output - RGB Image)
#IMAGE_PATH = "odm_orthophoto.tif"
IMAGE_DIR = "geotag_images"
# Output JSON filename
OUTPUT_JSON = "yellow_zones_rgb.json"

# Block size in meters (User requirement: 0.3m)
BLOCK_SIZE_METERS = 0.3

# HSV Color Range for "Yellow" (Stressed) Vegetation
# Aa values adjust karvi padi sake che lighting na hisabe
# OpenCV HSV Range: H (0-179), S (0-255), V (0-255)
# Yellow Hue approx 30 (OpenCV scale ma). Range 20-40.
LOWER_YELLOW = np.array([20, 100, 100]) # Min Yellow
UPPER_YELLOW = np.array([40, 255, 255]) # Max Yellow

def detect_color_zones_rgb():
    print(f"GeoTIFF load kariye chiye: {IMAGE_PATH}")
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: File nathi mali: {IMAGE_PATH}")
        return

    # GDAL thi open karvanu
    ds = gdal.Open(IMAGE_PATH)
    if ds is None:
        print("Error: GDAL file open nathi kari sakyu.")
        return

    # GeoTransform melavvanu (Coordinates mate)
    gt = ds.GetGeoTransform()
    # gt[0] = Top-Left X (Longitude)
    # gt[1] = Pixel Width (Resolution)
    # gt[2] = Rotation (0)
    # gt[3] = Top-Left Y (Latitude)
    # gt[4] = Rotation (0)
    # gt[5] = Pixel Height (Negative value)

    width = ds.RasterXSize
    height = ds.RasterYSize
    
    print(f"Image Size: {width}x{height}")
    print(f"Origin: ({gt[3]}, {gt[0]})")

    # Pixel resolution meters ma convert karvanu logic
    pixel_res_x = gt[1]
    
    # Check kariye ke projection Degrees ma che ke Meters ma
    if abs(pixel_res_x) < 1.0:
        # Degrees ma che (WGS84)
        # 1 degree approx 111320 meters (Equator pase)
        meters_per_pixel = abs(pixel_res_x) * 111320
        print(f"Projection: Geographic (Lat/Lon)")
    else:
        # Meters ma che (UTM)
        meters_per_pixel = abs(pixel_res_x)
        print(f"Projection: Projected (Meters)")

    print(f"Resolution: {meters_per_pixel:.4f} meters/pixel")

    # Block size pixels ma calculate kariye
    block_px = int(BLOCK_SIZE_METERS / meters_per_pixel)
    print(f"Block Size: {BLOCK_SIZE_METERS}m = {block_px} pixels")

    if block_px < 1:
        print("Warning: Block size 1 pixel thi nano che. 1 pixel use karishu.")
        block_px = 1

    # RGB Data read karvanu
    print("RGB Raster data read thay che...")
    
    # WebODM GeoTIFF usually 3 bands hoy che (R, G, B) sometimes 4 (Alpha)
    if ds.RasterCount < 3:
        print("Error: Image ma 3 bands nathi (RGB nathi lagtu).")
        return

    # Read bands as arrays (Band 1=Red, 2=Green, 3=Blue)
    red_band = ds.GetRasterBand(1).ReadAsArray()
    green_band = ds.GetRasterBand(2).ReadAsArray()
    blue_band = ds.GetRasterBand(3).ReadAsArray()

    # Stack bands to create (Height, Width, 3) array
    img_rgb = np.dstack((red_band, green_band, blue_band))
    
    # Ensure data is uint8 for OpenCV (WebODM usually 8-bit)
    if img_rgb.dtype != np.uint8:
        # Jo 16-bit hoy to convert karvu pade, pan atyare assume kariye ke 8-bit che
        # Or simple conversion:
        # img_rgb = (img_rgb / 256).astype(np.uint8)
        pass

    # Convert RGB to HSV
    # OpenCV ma COLOR_RGB2HSV use karvanu kem ke aapne RGB banavyu che
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Mask banavvanu based on HSV threshold
    # Aa range "Yellow" color detect karse
    mask = cv2.inRange(img_hsv, LOWER_YELLOW, UPPER_YELLOW)

    detected_points = []
    print("Color zones scan thay che...")

    for y in range(0, height, block_px):
        for x in range(0, width, block_px):
            roi = mask[y:y+block_px, x:x+block_px]
            
            # Count pixels in the range
            target_pixels = np.count_nonzero(roi)
            total_pixels = roi.size
            
            # Jo 30% thi vadhare target color pixels hoy to detect karvanu
            if total_pixels > 0 and (target_pixels / total_pixels) > 0.3:
                center_x = x + (block_px // 2)
                center_y = y + (block_px // 2)
                
                # Coordinate Calculation using GeoTransform
                # X_geo = gt[0] + x * gt[1] + y * gt[2]
                # Y_geo = gt[3] + x * gt[4] + y * gt[5]
                
                lon = gt[0] + center_x * gt[1] + center_y * gt[2]
                lat = gt[3] + center_x * gt[4] + center_y * gt[5]
                
                detected_points.append([round(lat, 7), round(lon, 7)])

    print(f"Total {len(detected_points)} blocks detect thaya.")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(detected_points, f, indent=2)
        
    print(f"Coordinates save thaya: {OUTPUT_JSON}")

if __name__ == "__main__":
    detect_color_zones_rgb()