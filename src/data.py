from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from l5kit.data import PERCEPTION_LABELS

from tqdm import tqdm
from collections import Counter
from prettytable import PrettyTable
import os

os.environ['L5KIT_DATA_FOLDER']='../.'
cfg = load_config_data('../visualization_config.yaml')

dm = LocalDataManager()
dataset_path=dm.require('../scenes/sample.zarr')
zarr_dataset=ChunkedDataset(dataset_path)
zarr_dataset.open()

'''
SCENE_DTYPE = [
    ("frame_index_interval", np.int64, (2,)),
    ("host", "<U16"),  # Unicode string up to 16 chars
    ("start_time", np.int64),
    ("end_time", np.int64),
]

FRAME_DTYPE = [
    ("timestamp", np.int64),
    ("agent_index_interval", np.int64, (2,)),
    ("traffic_light_faces_index_interval", np.int64, (2,)),
    ("ego_translation", np.float64, (3,)),
    ("ego_rotation", np.float64, (3, 3)),
]

AGENT_DTYPE = [
    ("centroid", np.float64, (2,)),
    ("extent", np.float32, (3,)),
    ("yaw", np.float32),
    ("velocity", np.float32, (2,)),
    ("track_id", np.uint64),
    ("label_probabilities", np.float32, (len(LABELS),)),
]

TL_FACE_DTYPE = [
    ("face_id", "<U16"),
    ("traffic_light_id", "<U16"),
    ("traffic_light_face_status", np.float32, (len(TL_FACE_LABELS,))),
]
'''

def visualize_trajectory(dataset, index):
    data = dataset[index]
    im = data['image'].transpose(1,2,0)
    im = dataset.rasterizer.to_rgb(im)
    target_position_pixels = transform_points(data['target_positions']+data['centroid'][:2],data['world_to_image'])
    draw_trajectory(im, target_position_pixels, data['target_yaws'], TARGET_POINTS_COLOR)

    plt.imshow(im[::-1])
    plt.show()
