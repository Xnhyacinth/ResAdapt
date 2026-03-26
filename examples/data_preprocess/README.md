# Video Data Preprocessing

This example demonstrates how to preprocess a local video dataset into a standardized Parquet format compatible with the ResAdapt training pipeline.

## Data Format

The preprocessed dataset will have the following schema:

- `data_source`: The name of the data source (e.g., "LLaVA-Video-178K").
- `prompt`: A list of conversation turns, where each turn is a dictionary with `role` ("user") and `content` (the prompt text).
- `images`: A list of images (empty for video-only tasks).
- `videos`: A list of video metadata dictionaries:
    - `type`: "video"
    - `video`: Absolute path to the video file.
    - `min_frames`: Minimum frames to extract (default: 1).
    - `max_frames`: Maximum frames to extract (default: 32).
- `ability`: Task category (e.g., "video_reasoning").
- `reward_model`: Reward configuration for RL:
    - `style`: Reward type (e.g., "rule").
    - `ground_truth`: The expected answer content.
- `extra_info`: Additional metadata including split, index, original question, answer, and paths.

## Usage

1. **Prepare your raw data**: Ensure your dataset is organized in subfolders (e.g., named by numbers) containing JSON/JSONL shards and video files.
2. **Install dependencies**:
   ```bash
   pip install datasets
   ```
3. **Run the script**:
   ```bash
   python preprocess_videodata.py \
       --local_dataset_root /path/to/your/raw_data \
       --local_save_dir /path/to/save/processed_data \
       --val_size 500
   ```

The script will generate `train.parquet` and `val.parquet` in the specified `local_save_dir`.
