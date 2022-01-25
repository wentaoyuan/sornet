from datasets import CLEVRDataset
from networks import EmbeddingNet, ReadoutNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--split', default='valB')
    parser.add_argument('--max_nobj', type=int, default=10)
    parser.add_argument('--img_h', type=int, default=320)
    parser.add_argument('--img_w', type=int, default=480)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    parser.add_argument('--n_relation', type=int, default=4)
    # Evaluation
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_worker', type=int, default=2)
    args = parser.parse_args()

    data = CLEVRDataset(
        f'{args.data_dir}/{args.split}.h5',
        f'{args.data_dir}/objects.h5',
        args.max_nobj, rand_patch=False
    )
    loader = DataLoader(data, args.batch_size, num_workers=args.n_worker)

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, args.max_nobj,
        args.width, args.layers, args.heads
    )
    head = ReadoutNet(args.width, args.d_hidden, 0, args.n_relation)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model = model.cuda().eval()
    head = head.cuda().eval()

    correct = 0
    total = 0
    for img, obj_patches, target, mask in tqdm(loader):
        img = img.cuda()
        obj_patches = obj_patches.cuda()
        with torch.no_grad():
            emb, attn = model(img, obj_patches)
            logits = head(emb)
            pred = (logits > 0).int().cpu()
        target = target.int()
        mask = mask.bool()
        correct += (pred[mask] == target[mask]).sum().item()
        total += mask.sum().item()

    print('Total', total)
    print('Accuracy', correct / total * 100)
