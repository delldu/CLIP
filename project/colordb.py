import argparse
import os
from tqdm import tqdm
from PIL import Image

import torch
import todos
from torchvision.transforms import ToTensor
import CLIP
import faiss # pip install faiss-cpu
import pdb


DATABASE_ROOT_DIR="."
image_filenames = todos.data.load_files(f"{DATABASE_ROOT_DIR}/images/*.png")

def get_vector(model, device, image_filename):
    image = Image.open(image_filename).convert("L").convert("RGB")
    image = ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model.encode_image(image)

    return feature


def create_database(model, device):
    print("Create database ...")

    # load files
    index = faiss.IndexFlatIP(512)
    quantizer = faiss.IndexFlatIP(512)

    progress_bar = tqdm(total=len(image_filenames))
    data = []
    for filename in image_filenames:
        progress_bar.update(1)
        vector = get_vector(model, device, filename)
        data.append(vector)

    data = torch.cat(data, dim = 0).cpu()
    progress_bar.close()

    print("Index ...")
    index.train(data.numpy())
    index.add(data.numpy())

    print(f"Saving index to {DATABASE_ROOT_DIR}/images.index ...")
    faiss.write_index(index, f"{DATABASE_ROOT_DIR}/images.index")


def search_image(model, device, filename):
    if not os.path.exists(f"{DATABASE_ROOT_DIR}/images.index"):
        print("Database not exist, please create in advance !!!")
        sys.exit(-1)

    print(f"Search image {filename} ...")
    index = faiss.read_index(f"{DATABASE_ROOT_DIR}/images.index")
    index.nprobe = 1
    qvector = get_vector(model, device, filename).cpu().numpy()
    D, I = index.search(qvector, 2)
    # print(D, I)
    qresult = [image_filenames[i] for i in I.reshape(-1).tolist()]
    print(qresult)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'Color Database {DATABASE_ROOT_DIR}')
    parser.add_argument('--create',
                        action='store_true',
                        help='Create database')

    parser.add_argument('--image',
                        type=str,
                        default='images/example2.png',
                        help='Search image from database')

    args = parser.parse_args()

    model_version = "ViT-B-16"
    model, device = CLIP.get_model(model_version)

    if args.create:
        create_database(model, device)
    else:
        search_image(model, device, args.image)