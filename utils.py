# utils.py
import os
import warnings
warnings.filterwarnings('ignore')

# scikit-image导入
try:
    from skimage import measure
    if hasattr(measure, 'marching_cubes'):
        marching_cubes = measure.marching_cubes
    elif hasattr(measure, 'marching_cubes_lewiner'):
        marching_cubes = measure.marching_cubes_lewiner
    else:
        marching_cubes = None
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    marching_cubes = None

def create_output_dir(output_dir):
    """创建输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")
    return output_dir