"""
Autonomous Precision Agriculture Mission Script
-----------------------------------------------
This script integrates the entire pipeline for a precision agriculture mission:
1. Image Stitching (using WebODM/PyODM)
2. NDVI Calculation (using Rasterio)
3. Yellow Zone/Stress Detection (using OpenCV/GDAL)
4. Drone Spraying Mission (using DroneKit)

Usage:
    Ensure WebODM is running on localhost:8000
    Ensure Drone/SITL is reachable at the connection string.
    Run: python autonomous_mission.py
"""

import os
import glob
import sys
import json
import time
import math
import csv
import numpy as np
import cv2
import rasterio
from pyodm import Node
from osgeo import gdal

try:
    import piexif
except ImportError:
    print("This script requires the 'piexif' library.")
    print("Please install it using: pip install piexif")
    sys.exit(1)

from PIL import Image

# Monkey patch for dronekit on Python 3.10+
import collections
import collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

from dronekit import connect, VehicleMode, LocationGlobalRelative

# --- CONFIGURATION ---

# Paths
INPUT_IMAGE_FOLDER = r"C:\Users\Admin\Documents\NIDAR\Nidar-Agrobotics-master\Nidar-Agrobotics-master\geotag_images"  # Folder containing source JPG images
OUTPUT_BASE_FOLDER = r"C:\Users\Admin\Documents\NIDAR\Nidar-Agrobotics-master\autonomous script\mission_output_1"
# GEOTAG_SCRIPT_PATH removed as logic is now internal

# WebODM Settings
WEBODM_HOST = "172.19.216.59"
WEBODM_PORT = 8000

# Drone Settings
DRONE_CONNECTION_STRING = 'udp:127.0.0.1:14550' # Change to your connection string (e.g., /dev/ttyACM0)
TARGET_ALTITUDE = 10 # meters
TANK_CAPACITY = 10 # liters

# Detection Settings
BLOCK_SIZE_METERS = 0.3
# HSV Color Range for "Yellow" (Stressed) Vegetation
# OpenCV HSV Range: H (0-179), S (0-255), V (0-255)
LOWER_YELLOW = np.array([20, 100, 100]) # Min Yellow
UPPER_YELLOW = np.array([40, 255, 255]) # Max Yellow

# ---------------------

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- Geotagging Helper Functions ---

def to_deg(value, loc):
    # Decimal coordinates ne degrees, minutes, seconds ma convert karo
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg =  int(abs_value)
    t1 = (abs_value-deg)*60
    min = int(t1)
    sec = round((t1 - min)* 60, 5)
    return (deg, min, sec), loc_value

def change_to_rational(number):
    # Number ne rational format (numerator, denominator) ma convert karo
    f = float(number)
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    denominator = 1000000
    numerator = int(f * denominator)
    common_divisor = gcd(numerator, denominator)
    return (numerator // common_divisor, denominator // common_divisor)

def parse_version(version_str):
    # Version string '2.2.0.0' ne tuple (2, 2, 0, 0) ma convert karo
    try:
        parts = [int(x) for x in version_str.split('.')]
        if len(parts) != 4:
            return (2, 2, 0, 0)
        return tuple(parts)
    except:
        return (2, 2, 0, 0)

def set_gps_location(file_name, lat, lng, altitude, lat_ref=None, lng_ref=None, alt_ref=None, version=None, date_time_original=None, date_time_digitized=None):
    # EXIF metadata ma GPS position add karo
    
    lat_deg, calc_lat_ref = to_deg(lat, ["S", "N"])
    lng_deg, calc_lng_ref = to_deg(lng, ["W", "E"])
    
    # Jo refs apela hoy to e use karo, nahi to calculate karo
    final_lat_ref = lat_ref if lat_ref else calc_lat_ref
    final_lng_ref = lng_ref if lng_ref else calc_lng_ref
    
    # Alt ref calculate karo jo na apelo hoy
    if alt_ref is None:
        final_alt_ref = 0 if altitude >= 0 else 1
    else:
        try:
            final_alt_ref = int(alt_ref)
        except:
            final_alt_ref = 0

    exiv_lat = (change_to_rational(lat_deg[0]), change_to_rational(lat_deg[1]), change_to_rational(lat_deg[2]))
    exiv_lng = (change_to_rational(lng_deg[0]), change_to_rational(lng_deg[1]), change_to_rational(lng_deg[2]))

    final_version = version if version else (2, 2, 0, 0)

    gps_ifd = {
        piexif.GPSIFD.GPSVersionID: final_version,
        piexif.GPSIFD.GPSAltitudeRef: final_alt_ref,
        piexif.GPSIFD.GPSAltitude: change_to_rational(abs(altitude)),
        piexif.GPSIFD.GPSLatitudeRef: final_lat_ref,
        piexif.GPSIFD.GPSLatitude: exiv_lat,
        piexif.GPSIFD.GPSLongitudeRef: final_lng_ref,
        piexif.GPSIFD.GPSLongitude: exiv_lng,
    }

    try:
        exif_dict = piexif.load(file_name)
    except Exception:
        exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
        
    exif_dict["GPS"] = gps_ifd
    
    # DateTimes add karo jo apela hoy
    if date_time_original:
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_time_original.encode('utf-8')
    if date_time_digitized:
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = date_time_digitized.encode('utf-8')
    
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, file_name)

def step0_geotag_images():
    """
    Applies GPS data to images using internal logic.
    """
    print("\n[STEP 0] Geotagging Images...")
    
    # Assume CSV is in the parent directory of the data folder, or same folder
    # Based on previous context: F:\NIDAR\GeoTagging Normal Images\gps_data.csv
    # INPUT_IMAGE_FOLDER is F:\NIDAR\GeoTagging Normal Images\data
    
    parent_dir = os.path.dirname(INPUT_IMAGE_FOLDER)
    csv_file = os.path.join(parent_dir, 'gps_data.csv')
    
    if not os.path.exists(csv_file):
        # Fallback to checking inside the folder itself
        csv_file = os.path.join(INPUT_IMAGE_FOLDER, 'gps_data.csv')
        
    if not os.path.exists(csv_file):
        print(f"Error: GPS Data CSV not found at {csv_file}")
        sys.exit(1)
        
    print(f"Reading GPS data from {csv_file}...")
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            # Headers mathi whitespace kadhi nakho
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
            count = 0
            for row in reader:
                filename = row['filename'].strip()
                
                # Image path banavo INPUT_IMAGE_FOLDER use karine
                image_path = os.path.join(INPUT_IMAGE_FOLDER, filename)
                    
                try:
                    lat = float(row['GPSLatitude'])
                    lng = float(row['GPSLongitude'])
                    alt = float(row['GPSAltitude'])
                    
                    # Optional fields
                    lat_ref = row.get('GPSLatitudeRef', '').strip()
                    lng_ref = row.get('GPSLongitudeRef', '').strip()
                    alt_ref = row.get('GPSAltitudeRef', '').strip()
                    version_str = row.get('GPSVersionID', '').strip()
                    
                    # New DateTime fields
                    dt_original = row.get('DateTimeOriginal', '').strip()
                    dt_digitized = row.get('DateTimeDigitized', '').strip()
                    
                    version = parse_version(version_str) if version_str else None
                    alt_ref_val = int(alt_ref) if alt_ref else None
                    
                    if os.path.exists(image_path):
                        # print(f"Processing {filename}...") # Verbose output reduced
                        set_gps_location(image_path, lat, lng, alt, lat_ref, lng_ref, alt_ref_val, version, dt_original, dt_digitized)
                        count += 1
                        print(f"Geotagged {filename}", end='\r')
                    else:
                        print(f"\nImage {filename} not found at {image_path}, skipping.")
                except ValueError as e:
                    print(f"\nError parsing data for {filename}: {e}")
                except Exception as e:
                    print(f"\nError processing {filename}: {e}")
            
            print(f"\nGeotagging complete. Processed {count} images.")
            
    except Exception as e:
        print(f"Error reading CSV or processing: {e}")
        sys.exit(1)

def step1_stitch_images(image_folder, output_folder):
    """
    Stitches images using WebODM to create an orthophoto.
    """
    print("\n[STEP 1] Starting Image Stitching...")
    
    node = Node(WEBODM_HOST, WEBODM_PORT)
    
    # Get images
    image_list = glob.glob(os.path.join(image_folder, "*.JPG"))
    if not image_list:
        # Try lowercase extension too
        image_list = glob.glob(os.path.join(image_folder, "*.jpg"))
        
    if not image_list:
        print(f"Error: No images found in {image_folder}")
        sys.exit(1)
        
    print(f"Found {len(image_list)} images.")

    # Processing Options (Fast/Lean)
    options = {
        "fast-orthophoto": True,
        "orthophoto-kmz": True,
        "dsm": False,
        "dtm": False,
        "feature-quality": "low",
        "pc-quality": "lowest",
        "max-concurrency": 8
    }

    try:
        print("Creating stitching task...")
        task = node.create_task(image_list, options)
        print(f"Task Created! ID: {task.uuid}")

        print("Processing... (This may take a while)")
        task.wait_for_completion(status_callback=lambda info: print(f"Stitching Progress: {info.progress}%", end="\r"))
        print("\nStitching complete.")

        print("Downloading assets...")
        task.download_assets(output_folder)
        
        # Locate the orthophoto
        # PyODM usually downloads to output_folder/odm_orthophoto/odm_orthophoto.tif
        ortho_path = os.path.join(output_folder, "odm_orthophoto", "odm_orthophoto.tif")
        
        if not os.path.exists(ortho_path):
            # Fallback check in root of output folder
            ortho_path = os.path.join(output_folder, "odm_orthophoto.tif")
            
        if os.path.exists(ortho_path):
            print(f"Orthophoto generated at: {ortho_path}")
            return ortho_path
        else:
            print("Error: Orthophoto file not found after download.")
            sys.exit(1)

    except Exception as e:
        # Check if the file exists despite the error (common with WinError 32 on zip cleanup)
        ortho_path_check = os.path.join(output_folder, "odm_orthophoto", "odm_orthophoto.tif")
        ortho_path_check_2 = os.path.join(output_folder, "odm_orthophoto.tif")
        
        if os.path.exists(ortho_path_check):
            print(f"Warning: An error occurred ({e}), but the orthophoto was found at {ortho_path_check}. Proceeding...")
            return ortho_path_check
        elif os.path.exists(ortho_path_check_2):
            print(f"Warning: An error occurred ({e}), but the orthophoto was found at {ortho_path_check_2}. Proceeding...")
            return ortho_path_check_2
        else:
            print(f"Error during stitching: {e}")
            sys.exit(1)

def step2_detect_color_zones_rgb(image_path, output_folder):
    """
    Detects yellow/stressed zones in the RGB orthophoto using HSV thresholding.
    """
    print("\n[STEP 2] Detecting Color Zones (RGB)...")
    
    output_json = os.path.join(output_folder, "mission_coordinates.json")
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    try:
        # Open with GDAL to get georeferencing
        ds = gdal.Open(image_path)
        if ds is None:
            print("Error: GDAL could not open the file.")
            sys.exit(1)

        gt = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        # Calculate resolution
        pixel_res_x = gt[1]
        if abs(pixel_res_x) < 1.0:
            # Degrees (WGS84)
            meters_per_pixel = abs(pixel_res_x) * 111320
        else:
            # Meters (UTM)
            meters_per_pixel = abs(pixel_res_x)

        block_px = int(BLOCK_SIZE_METERS / meters_per_pixel)
        if block_px < 1: block_px = 1
        
        print(f"Block Size: {BLOCK_SIZE_METERS}m = {block_px} pixels")

        # Read RGB Data
        print("Reading RGB raster data...")
        if ds.RasterCount < 3:
            print("Error: Image does not have 3 bands (RGB).")
            sys.exit(1)

        red_band = ds.GetRasterBand(1).ReadAsArray()
        green_band = ds.GetRasterBand(2).ReadAsArray()
        blue_band = ds.GetRasterBand(3).ReadAsArray()

        # Stack bands
        img_rgb = np.dstack((red_band, green_band, blue_band))
        
        # Convert to HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Create Mask based on HSV threshold
        mask = cv2.inRange(img_hsv, LOWER_YELLOW, UPPER_YELLOW)

        detected_points = []
        print("Scanning for color zones...")

        for y in range(0, height, block_px):
            for x in range(0, width, block_px):
                roi = mask[y:y+block_px, x:x+block_px]
                
                # Count pixels in the range
                target_pixels = np.count_nonzero(roi)
                total_pixels = roi.size
                
                # Threshold: >30% of the block is in the target color range
                if total_pixels > 0 and (target_pixels / total_pixels) > 0.3:
                    center_x = x + (block_px // 2)
                    center_y = y + (block_px // 2)
                    
                    # Coordinate Calculation
                    lon = gt[0] + center_x * gt[1] + center_y * gt[2]
                    lat = gt[3] + center_x * gt[4] + center_y * gt[5]
                    
                    detected_points.append([round(lat, 7), round(lon, 7)])

        print(f"Total {len(detected_points)} zones detected.")

        with open(output_json, 'w') as f:
            json.dump(detected_points, f, indent=2)
            
        print(f"Coordinates saved to: {output_json}")
        return output_json

    except Exception as e:
        print(f"Error detecting zones: {e}")
        sys.exit(1)

def get_distance_metres(aLocation1, aLocation2):
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

def step4_execute_drone_mission(json_path):
    """
    Connects to the drone and executes the spraying mission based on coordinates.
    """
    print("\n[STEP 4] Executing Drone Mission...")
    
    try:
        with open(json_path, 'r') as f:
            points = json.load(f)
    except Exception as e:
        print(f"Error loading mission points: {e}")
        sys.exit(1)

    if not points:
        print("No points to visit. Mission complete (nothing to do).")
        return

    print(f"Connecting to drone at {DRONE_CONNECTION_STRING}...")
    try:
        vehicle = connect(DRONE_CONNECTION_STRING, wait_ready=True)
        print("Drone Connected!")
    except Exception as e:
        print(f"Failed to connect to drone: {e}")
        print("Ensure drone/SITL is running and connection string is correct.")
        sys.exit(1)

    # Mission Prep
    total_points = len(points)
    spray_per_point = TANK_CAPACITY / total_points
    print(f"Mission Plan: {total_points} points. Spraying {spray_per_point:.2f}L per point.")

    # Arm and Takeoff
    print("Arming motors...")
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialise...", end="\r")
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...", end="\r")
        time.sleep(1)

    print(f"\nTaking off to {TARGET_ALTITUDE} meters...")
    vehicle.simple_takeoff(TARGET_ALTITUDE)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {alt:.1f}m", end="\r")
        if alt >= TARGET_ALTITUDE * 0.95:
            print("\nTarget altitude reached.")
            break
        time.sleep(1)

    # Execute Waypoints
    for i, point in enumerate(points):
        lat = point[0]
        lon = point[1]
        
        target_location = LocationGlobalRelative(lat, lon, TARGET_ALTITUDE)
        print(f"Going to Point {i+1}/{total_points}: Lat {lat}, Lon {lon}")
        
        vehicle.simple_goto(target_location)
        
        while True:
            current_loc = vehicle.location.global_relative_frame
            dist = get_distance_metres(current_loc, target_location)
            print(f"Distance: {dist:.1f}m", end="\r")
            if dist < 1.0:
                print("\nArrived at target.")
                break
            time.sleep(1)
            
        # Spray Mechanism
        print(f"Spraying {spray_per_point:.2f}L...")
        # Simulate spray duration
        time.sleep(2) 
        print("Spray complete.")

    # RTL
    print("\nMission complete. Returning to Launch (RTL)...")
    vehicle.mode = VehicleMode("RTL")
    
    print("Closing connection.")
    vehicle.close()

def main():
    print("=== AUTONOMOUS PRECISION AGRICULTURE SYSTEM ===")
    ensure_dir(OUTPUT_BASE_FOLDER)
    
    # 0. Geotag
    step0_geotag_images()

    # 1. Stitch
    ortho_path = step1_stitch_images(INPUT_IMAGE_FOLDER, OUTPUT_BASE_FOLDER)
    
    if ortho_path:
        # 2. Detect Zones (Using RGB Orthophoto)
        mission_json = step2_detect_color_zones_rgb(ortho_path, OUTPUT_BASE_FOLDER)
        
        # 3. Fly
        # step4_execute_drone_mission(mission_json)
    else:
        print("Stitching failed. Aborting mission generation.")
    
    print("\n=== SYSTEM SHUTDOWN ===")

if __name__ == "__main__":
    main()