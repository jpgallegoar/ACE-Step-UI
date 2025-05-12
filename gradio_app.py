"""
ACE-Step Gradio UI

A user-friendly interface for ACE-Step music generation.
"""

import os
import sys
import random
import json
import time
import click
import gradio as gr
import numpy as np
import torch
from acestep.pipeline_ace_step import ACEStepPipeline
import param_maps
import utils
from acestep.ui.components import TAG_DEFAULT, LYRIC_DEFAULT

# Constants
DEFAULT_CHECKPOINT_PATH = './ACE-Step-v1-3.5B'  # Will use auto-download if None
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_BF16 = True
DEFAULT_DEVICE_ID = 0
DEFAULT_TORCH_COMPILE = False

# Initialize the pipeline
pipeline_instance = None

# First, let's define the presets at the top of the file, after the imports
GENRE_PRESETS = {
    "Modern Pop": "pop, synth, drums, guitar, 120 bpm, upbeat, catchy, vibrant, female vocals, polished vocals",
    "Rock": "rock, electric guitar, drums, bass, 130 bpm, energetic, rebellious, gritty, male vocals, raw vocals",
    "Hip Hop": "hip hop, 808 bass, hi-hats, synth, 90 bpm, bold, urban, intense, male vocals, rhythmic vocals",
    "Country": "country, acoustic guitar, steel guitar, fiddle, 100 bpm, heartfelt, rustic, warm, male vocals, twangy vocals",
    "EDM": "edm, synth, bass, kick drum, 128 bpm, euphoric, pulsating, energetic, instrumental",
    "Reggae": "reggae, guitar, bass, drums, 80 bpm, chill, soulful, positive, male vocals, smooth vocals",
    "Classical": "classical, orchestral, strings, piano, 60 bpm, elegant, emotive, timeless, instrumental",
    "Jazz": "jazz, saxophone, piano, double bass, 110 bpm, smooth, improvisational, soulful, male vocals, crooning vocals",
    "Metal": "metal, electric guitar, double kick drum, bass, 160 bpm, aggressive, intense, heavy, male vocals, screamed vocals",
    "R&B": "r&b, synth, bass, drums, 85 bpm, sultry, groovy, romantic, female vocals, silky vocals"
}

# Add this function to handle preset selection
def update_tags_from_preset(preset_name):
    if preset_name == "Custom":
        return ""
    return GENRE_PRESETS.get(preset_name, "")

def initialize_pipeline(checkpoint_path, bf16, torch_compile, device_id):
    global pipeline_instance
    
    try:
        pipeline_instance = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=torch_compile,
            device_id=device_id
        )
        pipeline_instance.load_checkpoint(checkpoint_path)
        return True
    except Exception as e:
        print(f"Error initializing ACE-Step pipeline: {e}")
        return False


def text_to_music_callback(tags, lyrics, duration, creativity_slider, quality_speed_radio, 
                        use_advanced_settings=False, guidance_scale=None, guidance_scale_text=None, 
                        guidance_scale_lyric=None, manual_seeds=None):
    """Generate music from text and lyrics."""
    if pipeline_instance is None:
        return None, {"error": "Pipeline not initialized"}
    
    # Prepare parameters
    params = param_maps.get_text2music_params(
        tags=tags,
        lyrics=lyrics,
        duration=duration,
        creativity=creativity_slider,
        quality_choice=quality_speed_radio,
        use_advanced_settings=use_advanced_settings,
        guidance_scale_direct=guidance_scale,
        guidance_scale_text_direct=guidance_scale_text,
        guidance_scale_lyric_direct=guidance_scale_lyric
    )
    
    # Handle manual seeds if provided
    if manual_seeds:
        params["manual_seeds"] = utils.parse_manual_seeds(manual_seeds)
    
    # Generate output path
    output_dir = utils.get_output_directory()
    output_filename = utils.generate_output_filename("text2music")
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # Call the pipeline
        result = pipeline_instance(
            save_path=output_path,
            **params
        )
        
        # Save parameters to JSON for potential future editing
        if result and isinstance(result, list) and len(result) > 0:
            audio_path = result[0]
            utils.save_params_json(params, audio_path)
            return audio_path, params
        
        return None, {"error": "Generation failed"}
    
    except Exception as e:
        print(f"Error in text to music generation: {e}")
        return None, {"error": str(e)}


def audio_to_music_callback(audio_input, tags, lyrics, denoise_amount, use_advanced_transform, 
                            n_min_slider, n_max_slider, infer_step, guidance_scale, cfg_type, omega_scale):
    """Transform audio input to music with simplified controls."""
    if pipeline_instance is None:
        return None, {"error": "Pipeline not initialized"}
    
    if audio_input is None:
        return None, {"error": "No audio provided"}
    
    print(f"===== Processing Audio to Music =====")
    print(f"Audio input path: {audio_input}")
    print(f"Tags: {tags}")
    print(f"Lyrics: {lyrics}")
    
    if use_advanced_transform:
        print(f"Using advanced settings:")
        print(f"- n_min: {n_min_slider}, n_max: {n_max_slider}")
        print(f"- infer_step: {infer_step}")
        print(f"- guidance_scale: {guidance_scale}")
        print(f"- cfg_type: {cfg_type}")
        print(f"- omega_scale: {omega_scale}")
        effective_n_min = n_min_slider
        effective_n_max = n_max_slider
    else:
        print(f"Using simple denoise amount: {denoise_amount}")
        effective_n_min = 0.0
        effective_n_max = 1.0
        infer_step = 10  # Default value
        guidance_scale = 3.0  # Default value
        cfg_type = "apg"  # Default value
        omega_scale = 10.0  # Default value
    
    stereo_audio_path = utils.ensure_stereo_audio(audio_input)
    print(f"Stereo audio path: {stereo_audio_path}")
    
    # Use the new param_maps function
    params = param_maps.get_audio_transform_params(
        input_song_path=stereo_audio_path,
        new_tags=tags,
        new_lyrics=lyrics,
        denoise_amount=denoise_amount if not use_advanced_transform else 0.5,
        n_min=effective_n_min,
        n_max=effective_n_max,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        cfg_type=cfg_type,
        omega_scale=omega_scale
    )
    
    print(f"Parameters: {params}")
    
    output_dir = utils.get_output_directory()
    output_filename = utils.generate_output_filename("audio_transform")
    output_path = os.path.join(output_dir, output_filename)
    print(f"Output path: {output_path}")
    
    try:
        print(f"Calling pipeline with params...")
        result = pipeline_instance(
            save_path=output_path,
            **params
        )
        
        if result and isinstance(result, list) and len(result) > 0:
            audio_path = result[0]
            utils.save_params_json(params, audio_path)
            return audio_path, params
        
        return None, {"error": "Transformation failed"}
    
    except Exception as e:
        print(f"Error in audio transformation: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None, {"error": str(e)}


def song_editor_callback(loaded_song, operation_type, start_time, end_time_or_added_duration, 
                         new_tags, new_lyrics, variation_slider, manual_seeds=None):
    """Edit an existing song with various operations."""
    if pipeline_instance is None:
        return None, {"error": "Pipeline not initialized"}
    
    if loaded_song is None:
        return None, {"error": "No song loaded for editing"}
    
    print(f"===== Processing Song Edit =====")
    print(f"Loaded song: {loaded_song}")
    print(f"Operation type: {operation_type}")
    
    stereo_audio_path = utils.ensure_stereo_audio(loaded_song)
    print(f"Stereo audio path: {stereo_audio_path}")
    
    # Get parameters using the updated param_maps function
    params = param_maps.get_song_editor_params(
        loaded_song_path=stereo_audio_path,
        operation_type=operation_type,
        start_time=start_time,
        end_time_or_added_duration=end_time_or_added_duration,
        new_tags=new_tags,
        new_lyrics=new_lyrics,
        variation_slider=variation_slider
    )
    
    if manual_seeds:
        params["manual_seeds"] = utils.parse_manual_seeds(manual_seeds)
        
    print(f"Parameters for {operation_type}: {params}")
    
    output_dir = utils.get_output_directory()
    op_name_slug = operation_type.lower().replace(' ', '_').replace('(', '').replace(')', '')
    output_filename = utils.generate_output_filename(f"editor_{op_name_slug}")
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        result = pipeline_instance(
            save_path=output_path,
            **params
        )
        
        if result and isinstance(result, list) and len(result) > 0:
            audio_path = result[0]
            utils.save_params_json(params, audio_path)
            return audio_path, params
        
        return None, {"error": f"{operation_type} failed"}
        
    except Exception as e:
        print(f"Error in song editor ({operation_type}): {e}")
        import traceback
        traceback.print_exc()
        return None, {"error": str(e)}


def update_song_editor_ui(operation_type):
    """Updates the Song Editor UI elements based on the selected operation."""
    
    # Default visibility states
    start_time_visible = True
    end_time_visible = True
    
    # Default labels
    start_time_label = "Section Start Time (seconds)"
    end_time_label = "Section End Time (seconds)"
    
    if operation_type == "Inpaint Section":
        # Both start and end times are needed for inpainting
        start_time_visible = True
        end_time_visible = True
        start_time_label = "Section Start Time (seconds)"
        end_time_label = "Section End Time (seconds)"
    
    elif operation_type == "Add Intro (Prefix)":
        # Only need duration for prefix
        start_time_visible = False
        end_time_visible = True
        end_time_label = "Intro Duration (seconds)"
    
    elif operation_type == "Add Outro (Suffix)":
        # Only need duration for suffix
        start_time_visible = False
        end_time_visible = True
        end_time_label = "Outro Duration (seconds)"
    
    # Return the updates in a list, in the same order as the outputs in the change function
    return [
        gr.update(visible=start_time_visible, label=start_time_label),
        gr.update(visible=end_time_visible, label=end_time_label)
    ]


def create_ui():
    """Create the Gradio UI."""
    
    with gr.Blocks(
        title="ACE-Step Music Generator",
        theme=gr.themes.Soft(),
        css="""
        .footer {margin-top: 20px; text-align: center; color: #666;}
        """
    ) as demo:
        
        # State for storing the currently loaded song that can be accessed across tabs
        loaded_song_state = gr.State(value=None)
        
        gr.Markdown("# ACE-Step Music Generator")
        gr.Markdown("### A powerful AI music generation tool with a simple interface")
        
        with gr.Tabs() as tabs:
            # Tab 1: Text to Music
            with gr.Tab("🎼 Text to Music"):
                gr.Markdown("### Generate music from text and lyrics")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            genre_preset = gr.Dropdown(
                                choices=["Custom"] + list(GENRE_PRESETS.keys()),
                                value="Custom",
                                label="Genre Preset",
                                info="Select a preset or customize your own tags"
                            )
                        
                        tags = gr.Textbox(
                            lines=2,
                            label="Music Tags & Description",
                            info="Use descriptive tags separated by commas (genre, instruments, mood, tempo, etc.)",
                            placeholder="e.g., epic cinematic, female vocal, violin solo, 80 BPM, melancholic",
                            value=""
                        )
                        
                        # Add the change event for the preset dropdown
                        genre_preset.change(
                            fn=update_tags_from_preset,
                            inputs=[genre_preset],
                            outputs=[tags]
                        )
                        
                        lyrics = gr.Textbox(
                            lines=6,
                            label="Lyrics",
                            info="Use [verse], [chorus], [bridge] to structure your lyrics. Use [instrumental] for no vocals.",
                            placeholder="[verse]\nLyrics for your song...\n[chorus]\nMore lyrics...",
                            value=LYRIC_DEFAULT
                        )
                        
                        with gr.Row():
                            duration = gr.Slider(
                                minimum=10,
                                maximum=240,
                                step=1,
                                value=60,
                                label="Song Duration (seconds)"
                            )
                            
                            creativity = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                step=0.05,
                                value=0.7,
                                label="Creativity / Variation",
                                info="Lower = strict prompt adherence, Higher = more creative freedom"
                            )
                        
                        quality_speed = gr.Radio(
                            choices=["Fastest", "Balanced", "Highest Quality"],
                            label="Generation Quality/Speed",
                            value="Balanced",
                            info="Higher quality takes longer to generate"
                        )
                        
                        with gr.Accordion("Advanced Options", open=False):
                            use_advanced_settings = gr.Checkbox(
                                label="Use Advanced Guidance Settings",
                                value=False,
                                info="Enable direct control over guidance scale parameters"
                            )
                            
                            with gr.Column(visible=False) as advanced_settings_col:
                                guidance_scale = gr.Slider(
                                    minimum=3.0,
                                    maximum=20.0,
                                    step=0.1,
                                    value=7.0,  # Balanced default, user can increase for more adherence
                                    label="Main Guidance Scale",
                                    info="Overall prompt adherence. Less effective if Text/Lyric specific scales are > 1.0 and CFG type is 'cfg_double_condition'."
                                )
                                guidance_scale_text = gr.Slider(
                                    minimum=0.0,
                                    maximum=10.0,
                                    step=0.1,
                                    value=0.0,  # Default off; set > 1.0 to enable text-specific guidance
                                    label="Text Guidance Scale (CFG Double Condition)",
                                    info="Set > 1.0 (e.g., 5.0) along with Lyric Scale > 1.0 to enable separate text guidance."
                                )
                                guidance_scale_lyric = gr.Slider(
                                    minimum=0.0,
                                    maximum=10.0,
                                    step=0.1,
                                    value=0.0,  # Default off; set > 1.0 to enable lyric-specific guidance
                                    label="Lyric Guidance Scale (CFG Double Condition)",
                                    info="Set > 1.0 (e.g., 1.5 or 5.0) along with Text Scale > 1.0 to enable separate lyric guidance."
                                )
                            
                            manual_seeds = gr.Textbox(
                                label="Manual Seeds (optional)",
                                placeholder="e.g., 42, 100, 1234",
                                info="For reproducible results. Multiple seeds separated by commas."
                            )
                        
                        generate_button = gr.Button("Generate Music", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_audio = gr.Audio(
                            label="Generated Music",
                            type="filepath",
                            show_download_button=True
                        )
                        
                        with gr.Accordion("Output Parameters", open=False):
                            output_params = gr.JSON(label="Generation Parameters")
                        
                        save_to_editor_button = gr.Button("Send to Song Editor")
                
                # Event handlers for Text to Music
                use_advanced_settings.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[use_advanced_settings],
                    outputs=[advanced_settings_col]
                )
                
                generate_button.click(
                    fn=text_to_music_callback,
                    inputs=[
                        tags, lyrics, duration, creativity, quality_speed,
                        use_advanced_settings, guidance_scale, guidance_scale_text, guidance_scale_lyric,
                        manual_seeds
                    ],
                    outputs=[output_audio, output_params]
                )
                
                save_to_editor_button.click(
                    fn=lambda x: x,
                    inputs=[output_audio],
                    outputs=[loaded_song_state]
                )
            
            # Tab 2: Audio to Music (Consolidated from Music to Music and Mic to Music)
            with gr.Tab("🔄 Audio to Music"):
                gr.Markdown("### Transform your audio into music")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Allow both microphone recording and file upload
                        audio_input = gr.Audio(
                            label="Record or Upload Audio",
                            sources=["microphone", "upload"],
                            type="filepath"
                        )
                        
                        genre_preset_audio = gr.Dropdown(
                            choices=["Custom"] + list(GENRE_PRESETS.keys()),
                            value="Custom",
                            label="Genre Preset",
                            info="Select a preset or customize your own tags"
                        )
                        
                        tags_audio_transform = gr.Textbox(
                            lines=2,
                            label="Music Style (Optional)",
                            placeholder="e.g., jazz piano, mellow chords, relaxed",
                            info="The style of music you want to transform your audio into",
                            value=""
                        )
                        
                        lyrics_audio_transform = gr.Textbox(
                            lines=6,
                            label="Lyrics (Optional)",
                            placeholder="[verse]\nLyrics for your song...\n[chorus]\nMore lyrics...",
                            info="Use [verse], [chorus], [bridge] to structure your lyrics. Use [instrumental] for no vocals.",
                            value="[instrumental]"
                        )
                        
                        denoise_amount = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            value=0.5,
                            label="Denoise Amount",
                            info="Lower = keep more of your original audio, Higher = create more new content",
                            interactive=True
                        )
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            use_advanced_transform = gr.Checkbox(
                                label="Use Advanced Transform Settings",
                                value=False,
                                info="Enable direct control over transformation parameters. This will disable the simple Denoise Amount slider."
                            )
                            
                            with gr.Column(visible=False) as advanced_transform_col:
                                n_min_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.05,
                                    value=0.0,
                                    label="Minimum Noise Level (n_min)",
                                    info="When to start applying changes in the diffusion process. Lower values start changes earlier."
                                )
                                n_max_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.05,
                                    value=1.0,
                                    label="Maximum Noise Level (n_max)",
                                    info="When to stop applying changes in the diffusion process. Higher values apply changes longer."
                                )
                                
                                gr.Markdown("### Generation Parameters")
                                
                                infer_step = gr.Slider(
                                    minimum=10,
                                    maximum=100,
                                    step=1,
                                    value=10,
                                    label="Inference Steps",
                                    info="Number of denoising steps. Higher values = better quality but slower generation."
                                )
                                
                                guidance_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    step=0.5,
                                    value=3.0,
                                    label="Guidance Scale",
                                    info="How closely to follow the prompt. Higher values = more prompt adherence but potentially less natural results."
                                )
                                
                                cfg_type = gr.Radio(
                                    choices=["apg", "cfg", "cfg_star"],
                                    value="apg",
                                    label="Guidance Type",
                                    info="APG: Adaptive with momentum (smoother). CFG: Standard (can be sharper). CFG_STAR: CFG variant."
                                )
                                
                                omega_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    step=0.5,
                                    value=10.0,
                                    label="Omega Scale",
                                    info="Controls the strength of the guidance. Higher values = stronger effect."
                                )
                        
                        transform_button = gr.Button("Transform Audio", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_transformed_audio = gr.Audio(
                            label="Transformed Music",
                            type="filepath",
                            show_download_button=True
                        )
                        
                        with gr.Accordion("Output Parameters", open=False):
                            output_transformed_params = gr.JSON(label="Transformation Parameters")
                        
                        save_transformed_to_editor_button = gr.Button("Send to Song Editor")
                
                # Event handlers for Audio to Music
                def update_denoise_slider(use_advanced):
                    return gr.update(interactive=not use_advanced)
                
                use_advanced_transform.change(
                    fn=lambda x: [gr.update(visible=x), update_denoise_slider(x)],
                    inputs=[use_advanced_transform],
                    outputs=[advanced_transform_col, denoise_amount]
                )
                
                transform_button.click(
                    fn=audio_to_music_callback,
                    inputs=[
                        audio_input, tags_audio_transform, lyrics_audio_transform, denoise_amount,
                        use_advanced_transform, n_min_slider, n_max_slider,
                        infer_step, guidance_scale, cfg_type, omega_scale
                    ],
                    outputs=[output_transformed_audio, output_transformed_params]
                )
                
                save_transformed_to_editor_button.click(
                    fn=lambda x: x,
                    inputs=[output_transformed_audio],
                    outputs=[loaded_song_state]
                )
                
                genre_preset_audio.change(
                    fn=update_tags_from_preset,
                    inputs=[genre_preset_audio],
                    outputs=[tags_audio_transform]
                )
            
            # Tab 3: Song Editor
            with gr.Tab("✂️ Song Editor"):
                gr.Markdown("### Edit sections of a song, add intros or outros")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # The song being edited can come from the state or be uploaded here
                        with gr.Row():
                            editor_song_upload = gr.Audio(
                                label="Upload or Record Song to Edit",
                                type="filepath",
                                sources=["upload", "microphone"],
                                scale=3
                            )
                            
                            load_from_state_button = gr.Button("Load from Other Tab", scale=1)
                        
                        editor_song_display = gr.Audio(
                            label="Current Song",
                            type="filepath",
                            show_download_button=True
                        )
                        
                        operation_type = gr.Radio(
                            choices=["Inpaint Section", "Add Intro (Prefix)", "Add Outro (Suffix)"],
                            label="Edit Operation",
                            value="Inpaint Section"
                        )
                        
                        with gr.Row():
                            start_time = gr.Number(
                                label="Section Start Time (seconds)",
                                value=0
                            )
                            
                            end_time = gr.Number(
                                label="Section End Time (seconds)",
                                value=10
                            )
                        
                        genre_preset_editor = gr.Dropdown(
                            choices=["Custom"] + list(GENRE_PRESETS.keys()),
                            value="Custom",
                            label="Genre Preset",
                            info="Select a preset or customize your own tags"
                        )
                        
                        editor_tags = gr.Textbox(
                            lines=2,
                            label="New Section Tags",
                            placeholder="Tags for the new section",
                            info="Use descriptive tags for the new section",
                            value=""
                        )
                        
                        editor_lyrics = gr.Textbox(
                            lines=6,
                            label="Lyrics for New/Edited Section (or leave blank for instrumental)",
                            placeholder="[verse]\nNew lyrics for this section...",
                        )
                        
                        variation_amount = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            value=0.5,
                            label="Variation Amount",
                            info="How much the new section differs from the surrounding context"
                        )
                        
                        with gr.Accordion("Advanced Options", open=False):
                            editor_manual_seeds = gr.Textbox(
                                label="Manual Seeds (optional)",
                                placeholder="e.g., 42, 100, 1234",
                                info="For reproducible results"
                            )
                        
                        apply_edit_button = gr.Button("Apply Edit to Song", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_edited_audio = gr.Audio(
                            label="Edited Song",
                            type="filepath",
                            show_download_button=True
                        )
                        
                        with gr.Accordion("Output Parameters", open=False):
                            output_edited_params = gr.JSON(label="Edit Parameters")
                        
                        save_edited_as_current_button = gr.Button("Make This the Current Song")
                
                # Event handlers for Song Editor
                load_from_state_button.click(
                    fn=lambda x: x,
                    inputs=[loaded_song_state],
                    outputs=[editor_song_display]
                )
                
                editor_song_upload.upload(
                    fn=lambda x: x,
                    inputs=[editor_song_upload],
                    outputs=[editor_song_display]
                )
                
                apply_edit_button.click(
                    fn=song_editor_callback,
                    inputs=[editor_song_display, operation_type, start_time, end_time, 
                            editor_tags, editor_lyrics, variation_amount, editor_manual_seeds],
                    outputs=[output_edited_audio, output_edited_params]
                )
                
                save_edited_as_current_button.click(
                    fn=lambda x: x,
                    inputs=[output_edited_audio],
                    outputs=[editor_song_display, loaded_song_state]
                )
                
                # Update UI based on operation type
                operation_type.change(
                    fn=update_song_editor_ui,
                    inputs=[operation_type],
                    outputs=[start_time, end_time]
                )
                
                genre_preset_editor.change(
                    fn=update_tags_from_preset,
                    inputs=[genre_preset_editor],
                    outputs=[editor_tags]
                )
        
        # Footer
        gr.Markdown(
            """
            <div class="footer">
                <p>ACE-Step: A Step Towards Music Generation Foundation Model
                <br>
                <a href="https://github.com/ace-step/ACE-Step" target="_blank">GitHub Repository</a>
                </p>
            </div>
            """,
            elem_classes=["footer"]
        )
    
    return demo


@click.command()
@click.option(
    "--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH, 
    help="Path to the checkpoint directory (default: auto-download)"
)
@click.option(
    "--bf16", type=bool, default=DEFAULT_BF16, 
    help="Whether to use bfloat16 (default: True)"
)
@click.option(
    "--torch_compile", type=bool, default=DEFAULT_TORCH_COMPILE, 
    help="Whether to use torch compile (default: False)"
)
@click.option(
    "--device_id", type=int, default=DEFAULT_DEVICE_ID, 
    help="Device ID to use (default: 0)"
)
@click.option(
    "--share", type=bool, default=False, 
    help="Whether to create a public share link (default: False)"
)
@click.option(
    "--server_name", type=str, default="127.0.0.1", 
    help="Server name (default: 127.0.0.1)"
)
@click.option(
    "--server_port", type=int, default=7860, 
    help="Server port (default: 7860)"
)
def main(checkpoint_path, bf16, torch_compile, device_id, share, server_name, server_port):
    """Run the ACE-Step Gradio interface."""
    
    print("Initializing ACE-Step pipeline...")
    success = initialize_pipeline(checkpoint_path, bf16, torch_compile, device_id)
    
    if not success:
        print("Error initializing pipeline. Please check your checkpoint path and try again.")
        sys.exit(1)
    
    print("Creating Gradio UI...")
    demo = create_ui()
    
    print(f"Starting Gradio server on {server_name}:{server_port}")
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port
    )


if __name__ == "__main__":
    main() 