# ACE-Step User-friendly Gradio Interface

This is a simplified and intuitive Gradio interface for ACE-Step, making AI music generation more accessible through a user-friendly UI.

## Features

The interface offers three main capabilities:

1. **Text to Music**: Generate music from text prompts and lyrics
2. **Audio Transformation**: Transform existing audio files with new tags/lyrics or modify their style using denoising controls. Supports uploading files.
3. **Song Editor**: Edit sections of songs, add intros or outros

## Getting Started

### Installation

1. Make sure you have ACE-Step installed (https://github.com/ace-step/ACE-Step/)
2. Install additional dependencies:
   ```
   pip install gradio==4.26.0 soundfile
   ```

### Running the App

```
Copy utils.py, gradio_app.py and param_maps.py into the original ACE-Step folder.
python gradio_app.py
```

### Options

The app supports various command-line options:

```
Options:
  --checkpoint_path TEXT  Path to the checkpoint directory (default: auto-download)
  --bf16 BOOLEAN          Whether to use bfloat16 (default: True)
  --torch_compile BOOLEAN Whether to use torch compile (default: False)
  --device_id INTEGER     Device ID to use (default: 0)
  --share BOOLEAN         Whether to create a public share link (default: False)
  --server_name TEXT      Server name (default: 127.0.0.1)
  --server_port INTEGER   Server port (default: 7860)
  --help                  Show this message and exit.
```

## Interface Guide

### Text to Music

This tab focuses on straightforward generation from text prompts and lyrics.

1. Optionally select a **Genre Preset** to quickly populate the tags.
2. Enter **Music Tags & Description**: Use comma-separated tags describing the music you want (genre, instruments, mood, tempo, etc.)
3. Enter **Lyrics**: Structure your lyrics with tags like `[verse]`, `[chorus]`, and `[bridge]`. Use `[instrumental]` for instrumental music.
4. Adjust **Song Duration**: Set the length of the generated music in seconds.
5. Set **Creativity / Variation**: Lower values result in stricter adherence to the prompt; higher values allow more creative freedom. (This adjusts the `guidance_scale`).
6. Choose **Generation Quality/Speed**: Select between faster generation or higher quality output. (This adjusts `infer_step` and `scheduler_type`).
7. (Advanced) Manually set **Guidance Scale** values for fine-grained control.
8. (Advanced) Set **Manual Seeds** for reproducibility.
9. Click **Generate Music** to create your song.

### Audio Transformation

This tab allows you to modify existing audio files using new descriptive tags and lyrics, controlling the amount of change.

1. **Upload Input Audio**: Upload an audio file to modify.
2. Enter **New Tags**: Describe the desired style/elements for the transformed audio.
3. Enter **New Lyrics**: Provide new lyrics or use `[instrumental]`.
4. Adjust **Denoise Amount**: Control how much the original audio influences the output. Higher values lead to more significant transformation based on the new tags/lyrics.
5. (Advanced) Enable **Advanced Settings** to directly control `n_min`, `n_max`, `infer_step`, `guidance_scale`, `cfg_type`, and `omega_scale` for precise transformation control.
6. Click **Transform Audio** to create your modified version.

> **Note**: The app automatically converts mono audio to stereo format if needed, as required by ACE-Step.

### Song Editor

This tab lets you edit sections of songs, add intros, or outros.

1. **Upload Song to Edit** or click **Load from Other Tab** to edit a song generated in another tab.
2. Choose **Edit Operation**:
   - **Inpaint Section**: Replace a section of the song
   - **Add Intro (Prefix)**: Generate music that leads into the song
   - **Add Outro (Suffix)**: Generate music that extends the ending
3. Specify the timing parameters for your edit.
4. Enter **Tags** and **Lyrics** for the new section.
5. Adjust **Variation Amount** to control how different the new section is.
6. Click **Apply Edit to Song** to create your edited version.

## Advanced Usage

Each tab contains an **Advanced Options** accordion offering more granular control:
- **Text to Music**: Manual guidance scales and seeds.
- **Audio Transformation**: Direct control over diffusion parameters (`n_min`, `n_max`, `infer_step`, etc.) and seeds.
- **Song Editor**: Manual seeds and retake variance.

## Output

Generated audio appears in the right column of each tab. You can:

1. Play the audio directly in the browser
2. Download the audio file
3. View the generation parameters by expanding the **Output Parameters** accordion
4. Send the generated audio to the Song Editor tab for further editing

## Project Structure

- `gradio_app.py`: Main application file
- `param_maps.py`: Module for mapping simplified UI controls to ACE-Step parameters
- `utils.py`: Utility functions for the app

## How It Works

This Gradio interface simplifies ACE-Step's numerous parameters by:

1. **Abstraction**: Exposing intuitive controls (like Creativity, Denoise Amount) while managing complex underlying parameters.
2. **Sensible Defaults**: Using well-tested default values for many ACE-Step settings.
3. **Parameter Mapping**: Translating simple UI inputs (sliders, radio buttons) into appropriate combinations of ACE-Step parameters via `param_maps.py`.
4. **Audio Processing**: Automatically handling tasks like ensuring stereo audio input using functions in `utils.py`. 
