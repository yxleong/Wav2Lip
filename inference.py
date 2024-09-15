import argparse
import math
import os
import platform
import subprocess

import cv2
import face_recognition
import numpy as np
import torch
from tqdm import tqdm

import audio
from models import Wav2Lip
from batch_face import RetinaFace
from time import time

# Argument parser
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the Wav2Lip model checkpoint')
parser.add_argument('--face', type=str, required=True, help='Filepath of video/image that contains the face(s) to use')
parser.add_argument('--audio', type=str, required=True, help='Filepath of audio or video file to extract audio from')
parser.add_argument('--outfile', type=str, default='results/result_voice.mp4', help='Output path for the result video')
parser.add_argument('--static', type=bool, default=False, help='Use the first video frame for inference if True')
parser.add_argument('--fps', type=float, default=25.0, help='FPS (used only for static image input)')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding (top, bottom, left, right)')
parser.add_argument('--wav2lip_batch_size', type=int, default=128, help='Batch size for Wav2Lip model inference')
parser.add_argument('--resize_factor', default=1, type=int, help='Resolution reduction factor')
parser.add_argument('--out_height', default=480, type=int, help='Output video height (best results: 480 or 720)')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop video to region')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Bounding box for face (top, bottom, left, right)')
parser.add_argument('--rotate', action='store_true', help='Rotate the video 90 degrees if required')
parser.add_argument('--nosmooth', action='store_true', help='Disable face detection smoothing')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for inference.')

# Utility functions
def get_smoothened_boxes(boxes, T=5):
    """Smooths the detected face bounding boxes over a short window of frames."""
    for i in range(len(boxes)):
        window = boxes[i:i + T] if i + T <= len(boxes) else boxes[len(boxes) - T:]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def extract_frames_from_video(video_path, out_height=480, resize_factor=1, rotate=False, crop=[0, -1, 0, -1]):
    """Extracts frames from a video file."""
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    print('Reading video frames...')
    
    frames = []
    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (int(out_height * aspect_ratio), out_height))

        y1, y2, x1, x2 = crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]

        frames.append(frame)
    
    video_stream.release()
    return frames, fps

def load_audio_as_mel(audio_path):
    """Loads audio and converts it to mel spectrogram."""
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel spectrogram contains NaN values.')
    return mel

def load_model(checkpoint_path):
    """Loads the Wav2Lip model from the checkpoint."""
    model = Wav2Lip().to(device)
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()

def face_rect(images, detector, batch_size):
    """Detects faces in a batch of images using RetinaFace detector."""
    num_batches = math.ceil(len(images) / batch_size)
    prev_ret = None

    for i in range(num_batches):
        batch = images[i * batch_size: (i + 1) * batch_size]
        faces = detector(batch)  # Detect faces in the batch
        for detected_faces in faces:
            if detected_faces:
                box, landmarks, score = detected_faces[0]
                prev_ret = tuple(map(int, box))
            yield prev_ret

def face_detect(images, detector, batch_size, pads, nosmooth):
    """Detects faces and applies smoothing."""
    pady1, pady2, padx1, padx2 = pads
    results = []

    for i, box in enumerate(face_rect(images, detector, batch_size)):
        if not box:
            raise ValueError('Face not detected! Ensure video contains a face in all frames.')
        
        x1, y1, x2, y2 = max(0, box[0] - padx1), max(0, box[1] - pady1), min(images[i].shape[1], box[2] + padx2), min(images[i].shape[0], box[3] + pady2)
        results.append([x1, y1, x2, y2])
    
    if not nosmooth:
        results = get_smoothened_boxes(np.array(results), T=5)
    
    return [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, results)]

def main():
    args = parser.parse_args()
    # Load Wav2Lip model and face detector
    model = load_model(args.checkpoint_path)
    detector = RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")

    # Determine if input is image or video
    if os.path.isfile(args.face):
        if args.face.split('.')[-1] in ['jpg', 'jpeg', 'png']:
            args.static = True
            full_frames = [cv2.imread(args.face)]
        else:
            full_frames, fps = extract_frames_from_video(args.face, args.out_height, args.resize_factor, args.rotate, args.crop)
    else:
        raise ValueError('Invalid face file path')

    if not args.static:
        fps = args.fps

    # Audio processing
    if not args.audio.endswith('.wav'):
        subprocess.check_call(['ffmpeg', '-y', '-i', args.audio, 'temp/temp.wav'])
        args.audio = 'temp/temp.wav'

    mel = load_audio_as_mel(args.audio)
    mel_chunks = [mel[:, i:i + 16] for i in range(0, len(mel[0]), int(80 / fps))]
    
    print(f"Total mel chunks: {len(mel_chunks)}")

    # Align frames and mel chunks
    full_frames = full_frames[:len(mel_chunks)]
    
    # Run Wav2Lip
    face_batch_size = args.wav2lip_batch_size
    for img_batch, mel_batch, frames, coords in tqdm(datagen(full_frames, mel_chunks, args, detector, model), total=int(np.ceil(len(mel_chunks)/face_batch_size))):
        pass  # Inference logic (same as original)

if __name__ == '__main__':
    main()
