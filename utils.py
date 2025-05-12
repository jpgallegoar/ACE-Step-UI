"""
Utility functions for ACE-Step Gradio UI.
"""

import os
import json
import time
import datetime
import librosa
import numpy as np
import torch
import soundfile as sf
from pathlib import Path


def ensure_directory_exists(directory):
    """Make sure a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_output_directory():
    """Get the output directory for saving generated audio."""
    output_dir = os.path.join(os.getcwd(), "outputs")
    ensure_directory_exists(output_dir)
    return output_dir


def generate_output_filename(prefix="output"):
    """Generate a unique filename for output audio."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}.wav"


def save_params_json(params, audio_path):
    """Save parameters as a companion JSON file."""
    json_path = audio_path.replace(".wav", "_input_params.json").replace(".flac", "_input_params.json")
    
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    return json_path


def load_audio_duration(audio_path):
    """Get the duration of an audio file."""
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        return librosa.get_duration(y=audio_data, sr=sr)
    except Exception as e:
        print(f"Error loading audio duration: {e}")
        return None


def ensure_stereo_audio(input_audio_path):
    """
    Ensure the audio file is in stereo format.
    If the input is mono, convert it to stereo by duplicating the channel.
    
    Args:
        input_audio_path (str): Path to the input audio file
        
    Returns:
        str: Path to stereo audio file (either the original if already stereo,
             or path to a new stereo version if the original was mono)
    """
    try:
        # Load audio
        audio_data, sr = librosa.load(input_audio_path, sr=None, mono=False)
        
        # Check if mono (single channel)
        if audio_data.ndim == 1:
            print(f"Converting mono audio to stereo: {input_audio_path}")
            
            # Convert to stereo by duplicating the mono channel
            stereo_data = np.vstack((audio_data, audio_data))
            
            # Create a new filename for the stereo version
            output_dir = get_output_directory()
            filename = os.path.basename(input_audio_path)
            base, ext = os.path.splitext(filename)
            stereo_path = os.path.join(output_dir, f"{base}_stereo.wav")
            
            # Save the stereo version
            sf.write(stereo_path, stereo_data.T, sr, format="WAV")
            
            return stereo_path
        
        # Already stereo, return original path
        return input_audio_path
        
    except Exception as e:
        print(f"Error converting audio to stereo: {e}")
        # If there's an error, return the original path and let the pipeline handle any errors
        return input_audio_path


def parse_manual_seeds(seeds_str):
    """Parse a comma-separated string of seeds into a list of integers."""
    if not seeds_str or seeds_str.lower() == "none":
        return None
        
    try:
        seeds = [int(s.strip()) for s in seeds_str.split(',') if s.strip()]
        return seeds if seeds else None
    except ValueError:
        return None


def get_device_info():
    """Get information about available devices for the UI."""
    device_info = {
        "has_cuda": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "has_mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    return device_info


def format_time_mmss(seconds):
    """Format time in seconds to MM:SS format."""
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def check_for_existing_checkpoint(checkpoint_dir=None):
    """Check if ACE-Step checkpoint exists in the specified directory."""
    if not checkpoint_dir:
        # Use default path
        checkpoint_dir = os.path.join(os.path.expanduser("~"), ".cache/ace-step/checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        return False
    
    # Check for required files in the checkpoint directory
    required_paths = [
        os.path.join(checkpoint_dir, "music_dcae_f8c8", "config.json"),
        os.path.join(checkpoint_dir, "music_dcae_f8c8", "diffusion_pytorch_model.safetensors"),
        os.path.join(checkpoint_dir, "music_vocoder", "config.json"),
        os.path.join(checkpoint_dir, "music_vocoder", "diffusion_pytorch_model.safetensors"),
        os.path.join(checkpoint_dir, "ace_step_transformer", "config.json"),
        os.path.join(checkpoint_dir, "ace_step_transformer", "diffusion_pytorch_model.safetensors"),
        os.path.join(checkpoint_dir, "umt5-base", "config.json"),
        os.path.join(checkpoint_dir, "umt5-base", "model.safetensors"),
        os.path.join(checkpoint_dir, "umt5-base", "tokenizer.json"),
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            return False
    
    return True


def concatenate_audio_files(audio_paths, output_path):
    """
    Concatenate multiple audio files into a single file.
    
    Args:
        audio_paths (list): List of paths to audio files to concatenate
        output_path (str): Path to save the concatenated audio
        
    Returns:
        str: Path to the concatenated audio file if successful, None otherwise
    """
    try:
        import numpy as np
        import soundfile as sf
        
        # Load all audio files
        audio_data = []
        sampling_rates = []
        
        for path in audio_paths:
            data, sr = librosa.load(path, sr=None, mono=False)
            audio_data.append(data)
            sampling_rates.append(sr)
        
        # Check if all sampling rates are the same
        if len(set(sampling_rates)) > 1:
            # Resample all to the first sampling rate
            target_sr = sampling_rates[0]
            for i in range(1, len(audio_data)):
                if sampling_rates[i] != target_sr:
                    # If mono, convert to stereo first
                    if audio_data[i].ndim == 1:
                        audio_data[i] = np.vstack((audio_data[i], audio_data[i]))
                    
                    # Resample
                    audio_data[i] = librosa.resample(
                        audio_data[i], orig_sr=sampling_rates[i], target_sr=target_sr
                    )
            sampling_rates = [target_sr] * len(audio_data)
        
        # Ensure all audio is stereo
        for i in range(len(audio_data)):
            if audio_data[i].ndim == 1:
                audio_data[i] = np.vstack((audio_data[i], audio_data[i]))
        
        # Concatenate audio data
        concatenated = np.concatenate(audio_data, axis=1)
        
        # Save the concatenated audio
        sf.write(output_path, concatenated.T, sampling_rates[0], format="WAV")
        
        return output_path
    
    except Exception as e:
        print(f"Error concatenating audio files: {e}")
        import traceback
        traceback.print_exc()
        return None 