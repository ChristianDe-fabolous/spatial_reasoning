import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

# ----------------------------
# CONFIG
# ----------------------------
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

T_PAST = 10
T_FUTURE = 10
STRIDE = 5

IMG_SIZE = (224, 224)
CAMERA_NAME = dataset_pb2.CameraName.FRONT

# ----------------------------
# HELPERS
# ----------------------------
def save_image(img_bytes, save_path):
    img = Image.open(tf.io.gfile.GFile(img_bytes, "rb"))
    img = img.resize(IMG_SIZE)
    img.save(save_path)

def ego_normalize(xy, ref_xy, ref_yaw):
    xy = xy - ref_xy
    c, s = np.cos(-ref_yaw), np.sin(-ref_yaw)
    R = np.array([[c, -s], [s, c]])
    return xy @ R.T

# ----------------------------
# MAIN
# ----------------------------
def process_tfrecord(tfrecord_path: Path):
    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

    images = []
    positions = []
    yaws = []

    for record in tqdm(dataset, desc=tfrecord_path.name):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(record.numpy())

        cam = next(im for im in frame.images if im.name == CAMERA_NAME)
        images.append(cam.image)

        pos = np.array([
            frame.pose.transform.translation.x,
            frame.pose.transform.translation.y
        ])
        yaw = frame.pose.transform.rotation.yaw

        positions.append(pos)
        yaws.append(yaw)

    images = np.array(images)
    positions = np.array(positions)
    yaws = np.array(yaws)

    num_frames = len(images)

    for t in range(T_PAST, num_frames - T_FUTURE, STRIDE):
        ref_xy = positions[t]
        ref_yaw = yaws[t]

        ego_past = positions[t - T_PAST : t]
        ego_future = positions[t : t + T_FUTURE]

        ego_past = ego_normalize(ego_past, ref_xy, ref_yaw)
        ego_future = ego_normalize(ego_future, ref_xy, ref_yaw)

        sample_id = f"{tfrecord_path.stem}_{t}"
        sample_dir = PROCESSED_DIR / sample_id
        img_dir = sample_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        for i in range(T_PAST):
            save_image(
                images[t - T_PAST + i],
                img_dir / f"frame_{i}.jpg"
            )

        np.save(sample_dir / "ego_past.npy", ego_past.astype(np.float32))
        np.save(sample_dir / "ego_future.npy", ego_future.astype(np.float32))

# ----------------------------
# ENTRY
# ----------------------------
if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tfrecords = sorted(RAW_DATA_DIR.glob("*.tfrecord"))
    for tfrec in tfrecords:
        process_tfrecord(tfrec)

    print("Preprocessing finished.")

