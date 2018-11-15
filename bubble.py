# bubble module that finds and tracks bubbles

import cv2
import numpy as np


class Bubble:

    def __init__(self, start, idx):
        self.keypoints = [start]
        self.frames = [idx]
        self.first_seen = idx
        self.last_seen = idx

    def append(self, keypoint, idx):
        self.keypoints.append(keypoint)
        self.frames.append(idx)
        self.last_seen = idx

    @property
    def trajectory(self):
        pos = []
        for idx, keypoint in enumerate(self.keypoints):
            if self.first_seen + len(pos) == self.frames[idx]:
                pos.append(keypoint.pt)
            else:
                # Interpolate for missing positions
                last_known = pos[-1]
                n = self.frames[idx] - (self.first_seen + len(pos)) + 1
                inter_step = ((keypoint.pt[0]-last_known[0])/n, (keypoint.pt[1]-last_known[1])/n)
                for d in range(1, n):
                    pos.append((
                        last_known[0] + inter_step[0] * d,
                        last_known[1] + inter_step[1] * d
                    ))
                pos.append(keypoint.pt)
        return pos

    # TODO: Add convenience that allows to calculate speed etc...

class Tracker:

    def __init__(self):
        # Setup SimpleBlobDetector parameters with sensible
        # default parameters
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.minThreshold = 0
        self.params.maxThreshold = 255 * 0.5
        self.params.filterByArea = True
        self.params.minArea = 15
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.5
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.87
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.01

        # Tracking parameters
        self.max_speed = 5 # in pixels per frame
        self.survive_frames = 10 # max number of frames without next position
        self.min_path_length = 100 # number of frames a bubble must exist to be counted
        self.max_birth_frame = 100 # latest frame during which a bubble may be born

        # Properties
        self.frames = []
        self.bubbles = []
        self._did_track = False

        # Debug properties
        self.debug_keypoints = None

    def _find_keypoints(self, images):
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(self.params)
        # We assume the images have light blobs
        # (this is very tailored to the use-case we have)
        return [detector.detect(255 - img) for img in images]

    def _metric(self, a, b):
        return np.sqrt((a.pt[0] - b.pt[0])**2 + (a.pt[1] - b.pt[1])**2)

    def load_video(self, path, append_ok=False):
        if not append_ok and len(self.frames) > 0:
            raise Exception("Video already loaded.")
        video = cv2.VideoCapture(path)
        success = True
        while success:
            success, f = video.read()
            if success:
                self.frames.append(f)
        return len(self.frames)

    def track(self, overwrite=False):
        if not overwrite and self._did_track:
            return
        elif overwrite:
            self.bubbles = []
        if len(self.frames) < 2:
            raise Exception("Need at least two frames for tracking.")
        # Generate key points for all frames
        keypoints_list = self._find_keypoints(self.frames)
        self.debug_keypoints = keypoints_list
        # Track bubbles
        bubbles = []
        for frame_idx, keypoints in enumerate(keypoints_list):
            # Find new bubble positions
            for bubble in bubbles:
                # If there are no key points left, break
                if len(keypoints) == 0:
                    break
                # Skip the bubble if it has died
                if frame_idx - bubble.last_seen > self.survive_frames:
                    continue
                # TODO: Think about how the order might effect quality
                # Find closest key point
                least_distance = self._metric(bubble.keypoints[-1], keypoints[0])
                closest_idx = 0
                for idx, keypoint in enumerate(keypoints[1:]):
                    dist = self._metric(bubble.keypoints[-1], keypoint)
                    if dist < least_distance:
                        least_distance = dist
                        closest_idx = idx + 1 # As we omit the first element!
                if least_distance <= self.max_speed * (frame_idx - bubble.last_seen):
                    bubble.append(keypoints.pop(closest_idx), frame_idx)
            # Register all new bubbles
            if frame_idx < self.max_birth_frame:
                for keypoint in keypoints:
                    bubbles.append(Bubble(keypoint, frame_idx))
        # Remove bubbles that have short paths
        for bubble in bubbles:
            if bubble.last_seen - bubble.first_seen >= self.min_path_length:
                self.bubbles.append(bubble)
