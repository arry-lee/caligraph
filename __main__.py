import argparse
import cv2
import imageio
from animation import write_to_file, generate_frames

parser = argparse.ArgumentParser(description="Write frames to files")
parser.add_argument(
    "char", type=str, help="The character to generate frames for"
)
parser.add_argument(
    "-fp",
    "--filepath",
    type=str,
    default=None,
    help="The output filepath for the video file. Default is <char>.mp4",
)
parser.add_argument(
    "-s",
    "--size",
    nargs=2,
    type=int,
    default=[150, 150],
    help="The size (width and height) of the frames. Default is 150x150",
)
args = parser.parse_args()
if not args.filepath:
    args.filepath = f"{args.char}.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(args.filepath, fourcc, 30.0, args.size)
gif_writer = imageio.get_writer(f"{args.char}.gif", fps=30)
for key_frame in generate_frames(args.char, size=args.size):
    video_writer.write(cv2.cvtColor(key_frame, cv2.COLOR_GRAY2BGR))
    gif_writer.append_data(cv2.cvtColor(key_frame, cv2.COLOR_BGR2RGB))
video_writer.release()
gif_writer.close()
write_to_file(args.char, args.filepath, tuple(args.size))
