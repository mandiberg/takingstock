import argparse
import math
import time
import os
import ffmpeg
import pandas as pd

VIDEO_FOLDER = '../TheGymVideos/videos'
CSV_PATH = '../TheGymVideos/videos/installation.csv'

# rounding to 2 to avoid floating math errors on ratios
def floor_ratio(r):
    return math.floor(float(r) * 100) / 100


def parse_args():
    parser = argparse.ArgumentParser(description='Scale videos to a target height.')
    parser.add_argument('--limit', type=int, default=None, help='Only encode this many videos.')
    parser.add_argument('--dry-run', action='store_true', help='Run without encoding videos.')
    return parser.parse_args()


def input_resolution():
    while True:
        res = input('Enter the maximum height of your display in pixels: \n')
        if res.isnumeric():
            print(f'Selected resolution: {res}')
            return int(res)
        print('Please enter a valid number.')


def load_csv(file_path):
    df = pd.read_csv(file_path)
    before = len(df)
    df = df[df['file_name'] != 'file_name']
    removed = before - len(df)
    print(f'{removed} header rows removed. {len(df)} rows remaining.')
    return df


def verify_matches(df):
    '''verifies that all file names present in the CSV are also present in the videos folder.
    prints number of hits, misses, and names of missing files'''

    miss_list = []

    for file_name in df['file_name']:
        if not os.path.exists(os.path.join(VIDEO_FOLDER, file_name)):
            miss_list.append(file_name)

    hit_count = len(df) - len(miss_list)
    miss_count = len(miss_list)

    print(f'{hit_count} file names matched. {miss_count} files missing matches.')
    for miss in miss_list:
        print(f'File Missing Match: {miss}')
    return miss_list


def find_new_dimensions(df, target_height):
    '''takes a dataframe and target height in pixels.
    returns dict of file_name -> (new_width, target_height), using ratio column to compute width.
    width is forced even for h264 compatibility.'''
    new_resolutions = {}
    missing_ratio = []

    for _, row in df.iterrows():
        if pd.isna(row['ratio']):
            missing_ratio.append(row['file_name'])
            continue
        new_width = round(float(row['ratio']) * target_height)
        if new_width % 2 != 0:
            new_width += 1
        new_resolutions[row['file_name']] = (new_width, target_height)

    print(f'{len(new_resolutions)} dimensions calculated. {len(missing_ratio)} rows missing ratio.')
    for f in missing_ratio:
        print(f'Missing ratio: {f}')
    return new_resolutions


def update_df(df, new_resolutions):
    '''updates width, height, ratio, and file_name in df for all entries in new_resolutions.
    returns (updated_df, before_res, after_res) for resolution comparison.'''
    before_res = set(floor_ratio(r) for r in df['ratio'])
    for file_name, (frame_width, frame_height) in new_resolutions.items():
        out_name = f'{os.path.splitext(file_name)[0]}_{frame_height}.mp4'
        idx = df.index[df['file_name'] == file_name]
        df.loc[idx, 'width'] = frame_width
        df.loc[idx, 'height'] = frame_height
        df.loc[idx, 'ratio'] = floor_ratio(frame_width / frame_height)
        df.loc[idx, 'file_name'] = out_name
    after_res = set(floor_ratio(w / h) for w, h in new_resolutions.values())
    return df, before_res, after_res


def report_resolution_changes(before_res, after_res):
    print(f'\nRatio check: {len(before_res)} unique before → {len(after_res)} unique after.')
    only_before = before_res - after_res
    only_after = after_res - before_res
    if only_before:
        print(f'Only in original ({len(only_before)}): {sorted(only_before)}')
    if only_after:
        print(f'Only in output ({len(only_after)}): {sorted(only_after)}')
    if not only_before and not only_after:
        print('Ratios match exactly.')


def output_videos(new_resolutions, output_dir, df):
    '''encodes each video to its new dimensions.
    appends one CSV row per successful encode. returns (processed, skipped, failed).
    failed contains original input filenames for use in failed.csv.'''
    skipped = []
    processed = []
    failed = []
    csv_path = os.path.join(output_dir, 'installation.csv')
    total = len(new_resolutions)

    for i, (file_name, (frame_width, frame_height)) in enumerate(new_resolutions.items(), 1):
        out_name = f'{os.path.splitext(file_name)[0]}_{frame_height}.mp4'
        out_path = os.path.join(output_dir, out_name)

        #skip videos that already exist in output directory
        if os.path.exists(out_path):
            skipped.append(file_name)
            print(f'{i}/{total} {out_name} already exists, skipping.', flush=True)
            continue

        print(f'starting {i}/{total}: {file_name}', flush=True)
        start_timer = time.perf_counter()

        try:
            (
                ffmpeg
                .input(os.path.join(VIDEO_FOLDER, file_name))
                .filter('scale', frame_width, frame_height)
                .output(out_path, vcodec='h264_videotoolbox', loglevel='error')
                .run()
            )
            processed.append(file_name)
            header = not os.path.exists(csv_path)
            df[df['file_name'] == out_name].to_csv(csv_path, mode='a', index=False, header=header)
            print(f'{out_name} done. Took {round(time.perf_counter() - start_timer, 2)}s.', flush=True)
        except ffmpeg.Error as e:
            failed.append(file_name)
            stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
            print(f'FAILED {file_name}: {stderr}', flush=True)

    return processed, skipped, failed


if __name__ == "__main__":
    args = parse_args()
    target_height = input_resolution()
    df = load_csv(CSV_PATH)
    missing_files = verify_matches(df)
    output_dir = f'scaled_videos_{target_height}'
    os.makedirs(output_dir, exist_ok=True)
    new_resolutions = find_new_dimensions(df, target_height)
    original_df = df.copy()
    df, before_res, after_res = update_df(df, new_resolutions)
    report_resolution_changes(before_res, after_res)

    if args.dry_run:
        print('Dry run: skipping encoding.')
    else:
        encode_set = dict(list(new_resolutions.items())[:args.limit]) if args.limit else new_resolutions
        if args.limit:
            print(f'Limit set: encoding {args.limit} videos.')
        processed, skipped, failed = output_videos(encode_set, output_dir, df)
        all_failed = failed + missing_files
        total = len(processed) + len(skipped) + len(all_failed)
        print(f'Done. {len(processed)}/{total} encoded, {len(skipped)} skipped, {len(all_failed)} failed.')
        if all_failed:
            failed_csv_path = os.path.join(output_dir, 'failed.csv')
            original_df[original_df['file_name'].isin(all_failed)].to_csv(failed_csv_path, index=False)
            print(f'{len(all_failed)} failed files written to {failed_csv_path}')
