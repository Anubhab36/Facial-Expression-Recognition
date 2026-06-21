import argparse

from video.video_processor import VideoProcessor


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True
)

parser.add_argument(
    "--output",
    required=True
)

args = parser.parse_args()

processor = VideoProcessor()

processor.process(
    args.input,
    args.output
)