import glob
import os

from PIL import Image


def split_stitched_gifs(input_folder: str, num_angles: int, output_folder: str | None = None):
    """
    Finds all GIFs in a folder and de-interleaves their frames to create
    separate GIFs for each camera angle.

    It assumes that the frames for different angles at a single timestep are
    placed sequentially in the source GIF. For example, if a GIF has 3 angles,
    the frames are expected to be in the order:
    [t0_angle0, t0_angle1, t0_angle2, t1_angle0, t1_angle1, t1_angle2, ...]

    Originally implemented by alex to fix vicors messy mesh assembly v4 gifs.

    Args:
        input_folder (str): Path to the folder containing the source GIFs.
        num_angles (int): The number of camera angles interleaved in the GIF.
        output_folder (str, optional): Path to where the split GIFs will be saved.
                                     Defaults to the input_folder.
    """
    if output_folder is None:
        output_folder = input_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gif_files = glob.glob(os.path.join(input_folder, "*.gif"))

    for gif_path in gif_files:
        # Skip already processed gifs
        if "_angle_" in gif_path:
            continue
        try:
            with Image.open(gif_path) as img:
                if not img.is_animated:
                    print(f"Skipping non-animated GIF: {gif_path}")
                    continue

                if img.n_frames % num_angles != 0:
                    print(
                        f"Warning: Total frames ({img.n_frames}) in {gif_path} "
                        f"is not perfectly divisible by the number of angles ({num_angles})."
                    )

                # Collect all frames from the original GIF
                all_frames = []
                for i in range(img.n_frames):
                    img.seek(i)
                    # Use copy() to avoid issues with frame disposal methods
                    all_frames.append(img.copy())

                # De-interleave frames into separate lists for each angle
                angle_frames: list = [[] for _ in range(num_angles)]
                for i, frame in enumerate(all_frames):
                    angle_index = i % num_angles
                    angle_frames[angle_index].append(frame)

                base_name = os.path.splitext(os.path.basename(gif_path))[0]

                for i, frames in enumerate(angle_frames):
                    if frames:
                        output_filename = f"{base_name}_angle_{i + 1}.gif"
                        output_path = os.path.join(output_folder, output_filename)

                        duration = img.info.get("duration", 100)

                        frames[0].save(
                            output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=False
                        )
                        print(f"Saved split GIF: {output_path}")

        except Exception as e:
            print(f"Could not process {gif_path}. Error: {e}")


import argparse  # noqa: E402


def main():
    """
    Main function to parse command-line arguments and run the GIF splitting process.
    """
    parser = argparse.ArgumentParser(
        description="De-interleave frames of stitched GIFs for each camera angle.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example usage:\n"
            "python gif_hack.py path/to/your/gifs 3\n"
            "python gif_hack.py path/to/your/gifs 6 -o path/to/output\n"
            "python gif_hack.py --run-test"
        ),
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=None,
        type=str,
        help="Path to the folder containing the source GIFs. Required unless --run-test is used.",
    )
    parser.add_argument(
        "num_angles",
        nargs="?",
        default=None,
        type=int,
        help="The number of camera angles interleaved in the GIF. Required unless --run-test is used.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default=None,
        help="Path to save split GIFs. Defaults to the input folder if not provided.",
    )
    parser.add_argument(
        "--run-test", action="store_true", help="Run a self-contained test with dummy data. Ignores other arguments."
    )

    args = parser.parse_args()

    if args.run_test:
        run_dummy_test()
    elif args.input_folder is None or args.num_angles is None:
        parser.error("the following arguments are required: input_folder, num_angles (unless --run-test is used)")
    else:
        output_folder = args.output_folder or args.input_folder

        print(f"Input folder: {args.input_folder}")
        print(f"Number of angles: {args.num_angles}")
        print(f"Output folder: {output_folder}")

        split_stitched_gifs(input_folder=args.input_folder, num_angles=args.num_angles, output_folder=output_folder)
        print("\nProcessing complete.")


def run_dummy_test():
    """
    Sets up and runs a dummy test case to demonstrate the functionality.
    """
    print("Setting up and running a dummy test case...")

    dummy_input_folder = "dummy_input_gifs"
    dummy_output_folder = "dummy_output_gifs"
    os.makedirs(dummy_input_folder, exist_ok=True)
    os.makedirs(dummy_output_folder, exist_ok=True)

    num_test_angles = 3
    frame_width = 100
    frame_height = 100
    num_test_frames = 10

    stitched_width = frame_width * num_test_angles

    dummy_gif_path = os.path.join(dummy_input_folder, "stitched_test.gif")

    test_frames = []
    for i in range(num_test_frames):
        stitched_frame = Image.new("RGB", (stitched_width, frame_height))
        for j in range(num_test_angles):
            angle_img = Image.new("RGB", (frame_width, frame_height), color=(50 * (j + 1), 50 * (i % 5), 100))
            stitched_frame.paste(angle_img, (j * frame_width, 0))
        test_frames.append(stitched_frame)

    if test_frames:
        test_frames[0].save(dummy_gif_path, save_all=True, append_images=test_frames[1:], duration=150, loop=0)
        print(f"Created dummy GIF: {dummy_gif_path}")

        print("\nRunning the split function on the dummy data...")
        split_stitched_gifs(input_folder=dummy_input_folder, num_angles=num_test_angles, output_folder=dummy_output_folder)
        print("\nDummy test complete. Check the 'dummy_output_gifs' folder.")
    else:
        print("Could not create dummy test frames.")


if __name__ == "__main__":
    main()
