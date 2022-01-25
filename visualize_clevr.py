'''
MIT License

Copyright (c) 2022 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import h5py
import numpy as np
import torch
from PIL import Image
from datasets import normalize_rgb
from io import BytesIO
from matplotlib import pyplot as plt
from networks import EmbeddingNet, ReadoutNet

relations = {'left': 0, 'right': 1, 'front': 2, 'behind': 3}
relation_phrases = {
    'left': 'to the left of',
    'right': 'to the right of',
    'front': 'in front of',
    'behind': 'behind'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('scene_id', type=int)
    parser.add_argument('relation')
    parser.add_argument('obj1')
    parser.add_argument('obj2')
    parser.add_argument('--data_dir')
    parser.add_argument('--split')
    parser.add_argument('--img_h', type=int, default=320)
    parser.add_argument('--img_w', type=int, default=480)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()

    if args.relation not in relations:
        print(
            f'{args.relation} not supported. Use one of the following:',
            ', '.join(list(relations.keys()))
        )
    relation = relations[args.relation]

    obj_h5 = h5py.File(f'{args.data_dir}/objects.h5')
    if args.obj1 not in obj_h5:
        print(f'No canonical view of {args.obj1} available in object database')
        exit(1)
    if args.obj2 not in obj_h5:
        print(f'No canonical view of {args.obj2} available in object database')
        exit(1)
    obj1_patch = Image.open(BytesIO(obj_h5[args.obj1][0]))
    obj2_patch = Image.open(BytesIO(obj_h5[args.obj2][0]))
    patch_tensors = torch.stack(
        [normalize_rgb(obj1_patch), normalize_rgb(obj2_patch)]
    ).unsqueeze(0)

    scene_h5 = h5py.File(f'{args.data_dir}/{args.split}.h5')
    scene = scene_h5[f'{args.scene_id:06d}']
    objects = {
        obj: i for i, obj in enumerate(scene['objects'][()].decode().split(','))
    }
    if args.obj1 not in objects:
        print(f'{args.obj1} not in the scene. Prediction may be arbitrary.')
    else:
        obj1 = objects[args.obj1]
    if args.obj2 not in objects:
        print(f'{args.obj2} not in the scene. Prediction may be arbitrary.')
    else:
        obj2 = objects[args.obj2]

    gt = scene['relations'][relation][obj1][obj2]
    if gt == -1:
        print(
            'Relation is ambiguous due to repeated objects. '
            'Prediction may be arbitrary.'
        )
        gt_text = ''
    else:
        gt_text = 'Yes' if gt else 'No'

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, 2,
        args.width, args.layers, args.heads
    )
    head = ReadoutNet(args.width, args.d_hidden, 0, len(relations))
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model = model.eval()
    head = head.eval()

    img = Image.open(BytesIO(scene['image'][()])).convert('RGB')
    img_tensor = normalize_rgb(img).unsqueeze(0)
    with torch.no_grad():
        emb, attn = model(img_tensor, patch_tensors)
        logits = head(emb).cpu()
    pred = logits.reshape(len(relations), -1)[relation][0] > 0
    pred_text = 'Yes' if pred else 'No'

    rel_phrase = relation_phrases[args.relation]
    q_text = f"Is the {args.obj1.replace('_', ' ')} {rel_phrase}" \
             f" the {args.obj2.replace('_', ' ')}?"
    q_text = q_text.split()
    q1 = ' '.join(q_text[:len(q_text) // 2])
    q2 = ' '.join(q_text[len(q_text) // 2:])

    fig, (a0, a1, a2) = plt.subplots(
        1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [7, 2, 4]}
    )
    a0.imshow(img)
    a0.set_title('Input image', fontsize=18)
    a0.axis('off')
    obj_img = np.ones((224, 96, 3)).astype('uint8') * 255
    obj_img[:96] = np.array(obj1_patch.resize((96, 96)))
    obj_img[128:] = np.array(obj2_patch.resize((96, 96)))
    a1.imshow(obj_img)
    a1.set_title('Query Object', fontsize=18)
    a1.axis('off')
    a2.set_title('Question', fontsize=18)
    a2.text(0.5, 0.85, q1, fontsize=16, ha='center', va='center')
    a2.text(0.5, 0.75, q2, fontsize=16, ha='center', va='center')
    a2.text(0.5, 0.5, 'Answer', fontsize=18, ha='center', va='center')
    a2.text(0.5, 0.2, f'SORNet: {pred_text}', fontsize=16, ha='center', va='center')
    a2.text(0.5, 0.1, f'Ground truth: {gt_text}', fontsize=16, ha='center', va='center')
    a2.axis('off')
    plt.tight_layout()
    plt.show()
