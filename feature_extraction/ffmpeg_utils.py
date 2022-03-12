import ffmpeg
import numpy as np


def extract_frames(video_path, fps, size=None, crop=None, start=None, duration=None):
    if start is not None:
        cmd = ffmpeg.input(video_path, ss=start, t=duration)
    else:
        cmd = ffmpeg.input(video_path)

    if size is None:
        info = [s for s in ffmpeg.probe(video_path)["streams"] if s["codec_type"] == "video"][0]
        size = (info["width"], info["height"])
    elif isinstance(size, int):
        size = (size, size)

    if fps is not None:
        cmd = cmd.filter('fps', fps=fps)
    cmd = cmd.filter('scale', size[0], size[1])

    if crop is not None:
        cmd = cmd.filter('crop', f'in_w-{crop[0]}', f'in_h-{crop[1]}')
        size = (size[0] - crop[0], size[1] - crop[1])

    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
           .run(capture_stdout=True, quiet=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, size[1], size[0], 3])
    return video
