"""
Parameter mapping module for ACE-Step Gradio UI.
This module contains functions that map simplified UI controls to ACE-Step parameters.
"""

import json
import os
import librosa
import numpy as np


def map_creativity_to_guidance_scale(creativity_value):
    """
    Maps the creativity slider value to guidance_scale.
    Lower creativity = higher guidance_scale (more prompt adherence)
    Higher creativity = lower guidance_scale (more freedom)
    """
    return 20.0 - (creativity_value * 17.0)


def map_quality_speed_choice(quality_choice):
    """
    Maps quality/speed radio button selection to infer_steps and scheduler_type.
    """
    params = {}
    if quality_choice == "Fastest":
        params["infer_step"] = 27
        params["scheduler_type"] = "euler"
    elif quality_choice == "Balanced":
        params["infer_step"] = 50
        params["scheduler_type"] = "heun"
    else:  # "Highest Quality"
        params["infer_step"] = 100
        params["scheduler_type"] = "heun"
    return params


def map_change_intensity_for_edit(intensity_value, n_min, n_max):
    """
    Maps change intensity to edit_n_min and edit_n_max for 'edit' task.
    """
    params = {}
    params["edit_n_min"] = n_min 
    params["edit_n_max"] = n_max    
    
    return params


def get_text2music_params(tags, lyrics, duration, creativity, quality_choice, 
                         use_advanced_settings=False, guidance_scale_direct=None, 
                         guidance_scale_text_direct=None, guidance_scale_lyric_direct=None):
    """
    Maps simplified Text-to-Music UI controls to ACE-Step parameters.
    Supports both simple creativity slider and advanced guidance scale controls.
    
    Args:
        use_advanced_settings: If True, uses direct guidance scale values instead of creativity slider
        guidance_scale_direct: Direct value for main guidance scale when advanced settings are enabled
        guidance_scale_text_direct: Direct value for text guidance scale when advanced settings are enabled
        guidance_scale_lyric_direct: Direct value for lyric guidance scale when advanced settings are enabled
    """
    params = {
        "task": "text2music",
        "prompt": tags if tags else "instrumental music",
        "lyrics": lyrics if lyrics else "[instrumental]",
        "audio_duration": duration,
        "manual_seeds": None,
        "cfg_type": "apg",
        "omega_scale": 10.0,
        "guidance_interval": 0.5,
        "guidance_interval_decay": 0.0,
        "min_guidance_scale": 3.0,
        "use_erg_tag": True,
        "use_erg_lyric": True,
        "use_erg_diffusion": True,
        "oss_steps": []
    }

    if use_advanced_settings:
        params["guidance_scale"] = guidance_scale_direct
        params["guidance_scale_text"] = guidance_scale_text_direct
        params["guidance_scale_lyric"] = guidance_scale_lyric_direct
    else:
        params["guidance_scale"] = map_creativity_to_guidance_scale(creativity)
        params["guidance_scale_text"] = 0.0
        params["guidance_scale_lyric"] = 0.0

    quality_params = map_quality_speed_choice(quality_choice)
    params.update(quality_params)
    return params


def get_audio_transform_params(input_song_path, new_tags, new_lyrics, denoise_amount, n_min, n_max,
                            infer_step=10, guidance_scale=3.0, cfg_type="apg", omega_scale=10.0):
    """
    Parameters for transforming an existing audio with new tags and lyrics.
    This will use task="edit".
    
    Args:
        input_song_path: Path to the input audio file
        new_tags: New tags for the transformed audio
        new_lyrics: New lyrics for the transformed audio
        denoise_amount: Amount of denoising to apply (used when not in advanced mode)
        n_min: When to start applying changes in the diffusion process
        n_max: When to stop applying changes in the diffusion process
        infer_step: Number of inference steps (default: 10)
        guidance_scale: How closely to follow the prompt (default: 3.0)
        cfg_type: Type of classifier-free guidance to use (default: "apg")
        omega_scale: Strength of the guidance (default: 10.0)
    """
    params = {
        "task": "edit",
        "src_audio_path": input_song_path,
        "manual_seeds": None,
        "infer_step": infer_step,
        "guidance_scale": guidance_scale,
        "scheduler_type": "heun",
        "cfg_type": cfg_type,
        "omega_scale": omega_scale,
        "guidance_interval": 0.5,
        "guidance_interval_decay": 0.0,
        "min_guidance_scale": 3.0,
        "use_erg_tag": True,
        "use_erg_lyric": True,
        "use_erg_diffusion": True,
        "oss_steps": [],
        "guidance_scale_text": 0.0,
        "guidance_scale_lyric": 0.0
    }

    original_params_data = {}
    json_path = input_song_path.replace(".wav", "_input_params.json").replace(".flac", "_input_params.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            original_params_data = json.load(f)
    
    # Source prompt/lyrics (from original if available, else generic)
    params["prompt"] = original_params_data.get("prompt", "instrumental music")
    params["lyrics"] = original_params_data.get("lyrics", "[instrumental]")

    # Target prompt/lyrics for the edit
    params["edit_target_prompt"] = new_tags if new_tags else params["prompt"]
    params["edit_target_lyrics"] = new_lyrics if new_lyrics else "[instrumental]"

    # Map denoise_amount to edit_n_min, edit_n_max for "remix" type
    intensity_params = map_change_intensity_for_edit(denoise_amount, n_min, n_max)
    params.update(intensity_params)

    try:
        audio_data, sr = librosa.load(input_song_path, sr=None)
        params["audio_duration"] = librosa.get_duration(y=audio_data, sr=sr)
    except Exception as e:
        print(f"Error loading audio duration for {input_song_path}: {e}")
        params["audio_duration"] = 60.0

    return params


def get_song_editor_params(loaded_song_path, operation_type, start_time, 
                           end_time_or_added_duration, 
                           new_tags, new_lyrics, variation_slider):
    """
    Maps Song Editor UI controls to ACE-Step parameters.
    """
    params = {
        "src_audio_path": loaded_song_path,
        "manual_seeds": None,
        "infer_step": 50, 
        "guidance_scale": 15.0,
        "scheduler_type": "euler",
        "cfg_type": "apg",
        "omega_scale": 10.0,
        "guidance_interval": 0.5,
        "guidance_interval_decay": 0.0,
        "min_guidance_scale": 3.0,
        "use_erg_tag": True,
        "use_erg_lyric": True,
        "use_erg_diffusion": True,
        "oss_steps": [],
        "guidance_scale_text": 0.0,
        "guidance_scale_lyric": 0.0,
        "retake_variance": variation_slider
    }

    original_params_data = {}
    original_duration = None
    json_path = loaded_song_path.replace(".wav", "_input_params.json").replace(".flac", "_input_params.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            original_params_data = json.load(f)
        if "audio_duration" in original_params_data:
            original_duration = original_params_data["audio_duration"]

    if original_duration is None:
        try:
            audio_data, sr = librosa.load(loaded_song_path, sr=None)
            original_duration = librosa.get_duration(y=audio_data, sr=sr)
        except Exception as e:
            print(f"Error loading audio duration for {loaded_song_path}: {e}")
            original_duration = 60.0
    
    # Prompt and lyrics for guiding the new/edited section
    params["prompt"] = new_tags if new_tags else original_params_data.get("prompt", "instrumental music")
    params["lyrics"] = new_lyrics if new_lyrics else original_params_data.get("lyrics", "[instrumental]")

    if operation_type == "Inpaint Section":
        params["task"] = "repaint"
        params["repaint_start"] = float(start_time)
        params["repaint_end"] = float(end_time_or_added_duration)
        params["audio_duration"] = original_duration
    
    elif operation_type == "Add Intro (Prefix)":
        params["task"] = "extend"
        added_duration_val = float(end_time_or_added_duration)
        params["repaint_start"] = -added_duration_val
        params["repaint_end"] = original_duration
        params["audio_duration"] = original_duration + added_duration_val
        params["retake_variance"] = 1.0
    
    elif operation_type == "Add Outro (Suffix)":
        params["task"] = "extend"
        added_duration_val = float(end_time_or_added_duration)
        params["repaint_start"] = 0
        params["repaint_end"] = original_duration + added_duration_val
        params["audio_duration"] = original_duration + added_duration_val
        params["retake_variance"] = 1.0
    
    return params