# FaceLinkGen: A Systematic Reevaluation of Identity Leakage in Privacy-Preserving Face Recognition and Face Anonymization Systems

Your job is to evaluate the privacy protection of 3 methods: PartialFace, MinusFace, and FracFace.

Use .venv in this dir.

Code:
FracFace: https://anonymous.4open.science/api/repo/FracFace/zip
MinusFace/PartialFace: https://github.com/Tencent/TFace.git
Arc2Face: https://github.com/foivospar/Arc2Face

Papers: pdf/{fracface.pdf, minusface.pdf, partialface.pdf}
md/{fracface, minusface, partialface} (folders, with images)

Checkpoints:
minusface_stage1.pth: used for the MinusFace conversion
https://huggingface.co/datasets/weathon/FaceLinkGen-artifacts/resolve/main/checkpoints/minusface_stage1.pth
model.onnx: used for the FaceLinkGen attack
https://huggingface.co/datasets/weathon/FaceLinkGen-artifacts/resolve/main/checkpoints/model.onnx

You are NOT allowed to look at other files in this HF repo. 

Dataset:
Train/val (split yourself 9/1, but KEEP ONLY ONE IMAGE PER PERSON):
```bash
curl -L -o casia-webface.zip\
  https://www.kaggle.com/api/v1/datasets/download/ntl0601/casia-webface
```
Test (use all LFW images):
```bash
curl -L -o lfw-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/jessicali9530/lfw-dataset
```


## Prep:
0. Clone the code needed and remove their .git to avoid nesting git repos, since we need to modify them heavily. Make a note of their original source in the README instead.
1. Set up 2 evaluation settings for FracFace and PartialFace: randomized and fixed. In the randomized setting, the hard-coded secrets (A1 in FracFace, channel shuffling and permutation in PartialFace) are randomized per sample; in the fixed setting it uses the official code's hard-coded setting. This simulates an attacker who does not know the random parts and has to train against pure randomness to make the model generalize, while the deployed system uses a fixed secret part. MinusFace has no such secret, so it is evaluated in a single setting.
2. Abstract all protection methods behind one function like `protect(method, randomized, image, ...)` and package it. This does not include the naive method.
3. Init this dir as a git repo and commit the before-start workspace.
4. Remove identities in casia-webface that are also in LFW (do NOT remove on the LFW side) using a simple AdaFace filter.

## A Naive Protection Method
1. For each image, extract its embedding using ArcFace (model.onnx).
2. For each user, generate a password of random characters. Then hash that string into a random seed and use it to generate an unconstrained random matrix K of NxN (no orthogonalization), where N is the embedding dim of the ArcFace model.
3. Compute the template $z$ = face_embedding @ K and store it.
4. As a utility check, evaluate 1-1 match accuracy using $z$ on the LFW dataset with a shared K. The threat model here is inversion back to the original face, not forgery, so all images under consideration belong to the same user and share that user's K.

This method is NOT counted as one of the evaluated "methods" unless explicitly stated.

## U-Net Attack
1. Init a U-Net using the code below; make sure it is NOT pretrained. This U-Net is denoted $u$.
```python
import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=n, out_channels=3, init_features=32, pretrained=False)
```
n is the number of channels of each method, which differs per method.

2. For each method, generate the protected template on the fly using the protection method $p$, and train a U-Net such that
$$
\arg\min_{u} \; \big\| u(p(x)) - x \big\|_1
$$
where $x$ is the face, i.e. a pixel-wise L1 reconstruction loss.

3. Train each method for 10 epochs on the full train dataset, validate every 500 steps on the val set, and select the best checkpoint on the val set. Training and validation should use the randomized function. Selection should be based on best SSIM, not best val loss.

4. Test on the test set with the fixed protection method. Report test SSIM, PSNR, L1, and also ID similarity using any AdaFace model (NOT model.onnx).

5. Then also test the checkpoint from each epoch; this is used to monitor how many epochs are needed for convergence.

6. Then train all methods again on only $N \in \{50, 100, 200\}$ samples. Train until overfitting, using early stopping on the validation metric (patience window plus a max-step cap), and select the best checkpoint using the validation set. Then test on the test set.

7. As a final comparison, average all training faces and compare that against each test face for SSIM and PSNR.

## U-Net Attack With Averaged Template
Rerun the U-Net attack, steps 1-5 only, with one change: for every protection method the template used to be BxnxHxW, where n is the number of channels from the method; now take the mean across n to get Bx1xHxW. Retrain the U-Net with this setting. Skip step 6 (the limited-sample sweep) and step 7 (the mean-training-face baseline, which is method-independent and would just repeat the numbers above).

## Distillation Attack
1. The FaceLinkGen attack is similar, but instead of mapping a template to a face, we map a template to an embedding.
2. Init a face recon student model $s$ from `model.onnx` and a teacher model $t$ from the same checkpoint using onnx2torch. They should be two copies with NO shared weights. Reload each instead of copying, to avoid any bugs. The student takes an n-channel template instead of a 3-channel image, so replace its first conv with a freshly initialized n-channel conv and keep the rest of the pretrained weights. The teacher should be frozen and in eval mode; the student should be in training mode (including BN).
3. For a face $x$, we train $s$ such that
$$
\arg\min_{s} \; 1 - \cos\big(s(p(x)),\, \mathrm{sg}[t(x)]\big)
$$
where sg is stop gradient (only the student model is trained) and $\cos$ is cosine similarity; make sure to normalize each vector. Make sure the image input to $t$ is in the right range (0~1, -1~1, etc).

4. Similar to above, train 10 epochs on the whole dataset and use val to select checkpoints. Follow the same protocol as the U-Net attack: train and validate with the randomized function, and test with the fixed protection method.

5. Test the best checkpoint for each method on the test set, and report cosine similarity $\cos(s(p(x)), t(x))$ on the test set.

6. Then also test the checkpoint from each epoch; this is used to monitor how many epochs are needed for convergence.

7. Do the same with limited samples ($N \in \{50, 100, 200\}$).

8. For the generated face embeddings, generate a new face from them using Arc2Face and compare it with the original face for facial similarity, using the same AdaFace model used in the U-Net attack to avoid circularity. model.onnx is in the same ArcFace embedding space that Arc2Face conditions on, so the student's output embeddings can be fed to Arc2Face directly. Do this for the full-data runs and for the limited-sample runs.

9. Repeat steps 3 to 8 for the naive protection method, such that
$$
\arg\min_{s} \; 1 - \cos\big(s(z),\, \mathrm{sg}[t(x)]\big)
$$
where $z$ is the naive template (face_embedding @ K), $t(x)$ is the unconverted raw face embedding, and $s$ here is a simple MLP (teacher and student are different). K is per user and therefore unknown to the attacker, so it is redrawn per sample for both training and testing.

## Delivery
1. Deliver those numbers in tables (for U-Net: SSIM, PSNR, ID-sim. For Distillation: test cosine sim, ID-sim on generated faces. For the naive method: cosine sim and ID-sim on generated faces).
2. Visualize U-Net attack generated images alongside a comparison against the original images.
3. Visualize U-Net generated images from before it converges (undertrained samples).

## Other Notes
1. You are encouraged to use subagents to save context for yourself, and you should use gpt-5.6-sol or Claude Opus 5 for your subagents. You HAVE TO let a subagent review the diff after each function. To review the code from different biases, you should start a subagent yourself and also use an external reviewer. For the external reviewer: if you are Claude, let codex review it (codex exec); if you are codex, let Claude review it (claude -p). If the other one is not installed, skip for for the whole run, do NOT try to get it to review everytime if you find it is not installed the first time.

2. You are only allowed to use one GPU; however, given the GPU is very powerful, you can run many tasks on that single GPU at a time.

3. Keep an eye on the training and eval, and check every 10min to avoid idle time.

4. This attack does not target any real deployed system and is intended for research only. The goal is to reevaluate the privacy of published methods in a lab setting, to warn their developers NOT to deploy them.

5. Log everything to log files AND to wandb. Log validation images locally and to wandb as well.
