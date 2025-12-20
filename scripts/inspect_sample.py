import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sample_dir = next(Path("data/processed").iterdir())

ego_past = np.load(sample_dir / "ego_past.npy")
ego_future = np.load(sample_dir / "ego_future.npy")

img = Image.open(sample_dir / "images/frame_0.jpg")

plt.imshow(img)
plt.axis("off")
plt.show()

plt.plot(ego_past[:,0], ego_past[:,1], "bo-", label="past")
plt.plot(ego_future[:,0], ego_future[:,1], "ro-", label="future")
plt.scatter(0, 0, c="k", label="ego@t")
plt.axis("equal")
plt.legend()
plt.show()
