import os
import tifffile
import numpy as np
import zarr

CH_DAPI = 0

raw_images_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\raw_images"

PATIENT_METADATA = {
    "SNT354": {"age": 35, "file": "SNT354_PC24058_Scan1.qptiff"},
    "SNT899": {"age": 35, "file": "SNT899_PC24046_Scan1.qptiff"},
    "SNT348": {"age": 37, "file": "SNT348_PC24056_Scan1.qptiff"},
    "SNT393": {"age": 37, "file": "SNT393_PC24043_Scan1.qptiff"},
    "SNT227": {"age": 69, "file": "SNT227_PC24038_Scan1.qptiff"},
    "SNT484": {"age": 69, "file": "SNT484_PC24045_Scan1.qptiff"},
    "SNT675": {"age": 69, "file": "SNT675_PC24049_Scan1.qptiff"},
}

datasets = [
    {"age": meta["age"], "id": pid, "path": os.path.join(raw_images_dir, meta["file"])}
    for pid, meta in PATIENT_METADATA.items()
]

base_output_dir = r"D:\GitHub\Pancreas-Spatial-Senescence\dataset\tiles"
os.makedirs(base_output_dir, exist_ok=True)
TILE_SIZE = 2048

for ds in datasets:
    age = ds["age"]
    pid = ds["id"]
    file_path = ds["path"]
    
    out_dir = os.path.join(base_output_dir, f"age_{age}_{pid}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*50}\nPROCESSING PATIENT: {pid} (Age: {age})\n{'='*50}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found! -> {file_path}")
        continue
        
    try:
        with tifffile.TiffFile(file_path) as tif:
            store = tif.series[0].aszarr()
            z_group = zarr.open(store, mode='r')
            
            if isinstance(z_group, zarr.hierarchy.Group):
                z = z_group['0']
            else:
                z = z_group
                
            if len(z.shape) == 3:
                num_channels, height, width = z.shape
                if num_channels > 100:
                    z_to_use = np.transpose(z, (2, 0, 1))
                    num_channels, height, width = z_to_use.shape
                else:
                    z_to_use = z
            elif len(z.shape) == 4:
                z_to_use = z[:, 0, :, :]
                num_channels, height, width = z_to_use.shape
            else:
                print(f"Unsupported shape: {z.shape}")
                continue
                
            print(f"Original Image Size: {width}x{height}, Channels: {num_channels}")
            
            x_coords = range(0, width, TILE_SIZE)
            y_coords = range(0, height, TILE_SIZE)
            total_tiles = len(x_coords) * len(y_coords)
            
            print(f"Extracting {total_tiles} tiles of size {TILE_SIZE}x{TILE_SIZE}...")
            
            count = 0
            for y in y_coords:
                for x in x_coords:
                    count += 1
                    y_end = min(y + TILE_SIZE, height)
                    x_end = min(x + TILE_SIZE, width)
                    
                    tile_name = f"tile_y{y}_x{x}.tiff"
                    tile_path = os.path.join(out_dir, tile_name)
                    
                    if os.path.exists(tile_path):
                        continue
                        
                    tile_data = z_to_use[:, y:y_end, x:x_end]
                    
                    if np.max(tile_data[CH_DAPI]) <= 0:
                        continue
                        
                    tifffile.imwrite(tile_path, tile_data, photometric='minisblack')
                    
                    if count % 10 == 0:
                        print(f"[{count}/{total_tiles}] Saved: {tile_name}")
                        
    except Exception as e:
        print(f"ERROR: {str(e)}")
