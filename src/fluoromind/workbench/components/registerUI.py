import gradio as gr
import tempfile
from ..config import get_workdir
from ...core.registration import apply_transform
from ...io import read_stack_image, read_image_array, read_single_image
import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import files


atlas_path = files("fluoromind.workbench").joinpath("ccf.npy")


def read_file(filenames):
    if filenames is None:
        return None
    if len(filenames) == 1:
        return read_stack_image(filenames[0])
    elif len(filenames) > 1:
        return read_image_array(filenames)
    else:
        return None


def read_atlas(filename):
    if filename is None:
        return np.load(atlas_path)
    return read_single_image(filename)


def extract_frame(data, frame_number):
    if not data:
        return None
    return data[frame_number]


def register_frame(
    frame,
    mask,
    translation_x,
    translation_y,
    rotation,
    scale_x,
    scale_y,
    shear,
    color_map="RdYlBu_r",
):
    if frame is None or mask is None:
        return None

    # Apply transformation to the selected frame
    transformed_frame = apply_transform(
        image=frame,
        translation=(translation_x, translation_y),
        rotation=rotation,
        scale=(scale_x, scale_y),
        shear=shear,
        shape=mask.shape,
    )

    normalized = np.interp(transformed_frame, (transformed_frame.min(), transformed_frame.max()), (0, 1))
    masked_frame = np.ma.masked_array(normalized, mask=mask)
    return plt.cm.get_cmap(color_map)(masked_frame)


def process_entire_file(
    data,
    mask,
    translation_x,
    translation_y,
    rotation,
    scale_x,
    scale_y,
    shear,
):
    if data is None or mask is None:
        raise gr.Error("No data or mask provided")
    # Apply transformation to all frames
    transformed_data = []
    for t in range(len(data)):
        transformed_data.append(
            apply_transform(
                image=data[t].astype(np.float32),
                translation=(translation_x, translation_y),
                rotation=rotation,
                scale=(scale_x, scale_y),
                shear=shear,
                shape=mask.shape,
            )
        )

    output_params = {
        "translation": {"x": translation_x, "y": translation_y},
        "rotation": rotation,
        "scale": {"x": scale_x, "y": scale_y},
        "shear": shear,
        "shape": mask.shape,
    }

    # Save the transformed data to a file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npy", dir=get_workdir())
    np.save(tmp.name, transformed_data)

    # Return the path to the saved file
    return output_params, tmp.name


def update_frame_slider(file_content):
    if file_content is None:
        return gr.update(maximum=0)
    num_frames = len(file_content)
    return gr.update(maximum=num_frames - 1)


# Update the Gradio interface
with gr.Blocks() as ui:
    file_content = gr.State(None)
    atlas_content = gr.State(np.load(atlas_path))
    frame_result = gr.State(None)
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Inputs")
            with gr.Row():
                file_input = gr.File(label="Input Image", file_count="multiple", interactive=True)
                atlas_input = gr.File(label="Atlas File (default: ccf.npy)", interactive=True)

            frame_slider = gr.Slider(label="Preview Frame", step=1, minimum=0)

            gr.Markdown("## Registration Parameters")
            with gr.Row():
                translation_x = gr.Slider(label="Translation X", step=1, minimum=-100, maximum=100, value=0)
                translation_y = gr.Slider(label="Translation Y", step=1, minimum=-100, maximum=100, value=0)

            with gr.Row():
                rotation = gr.Slider(label="Rotation", step=0.1, minimum=-180, maximum=180, value=0)
                shear = gr.Slider(label="Shear", step=0.1, minimum=-180, maximum=180, value=0)

            with gr.Row():
                scale_x = gr.Slider(label="Scale X", step=0.01, minimum=0.1, maximum=10, value=1)
                scale_y = gr.Slider(label="Scale Y", step=0.01, minimum=0.1, maximum=10, value=1)

        with gr.Column():
            gr.Markdown("## Results")
            preview_output = gr.Image(label="Preview Frame")
            process_button = gr.Button("Process File")

            output_file = gr.File(label="Transformed Data File")
            output_params = gr.JSON(label="Parameters")

    # Set up event listeners for live preview
    file_input.change(
        fn=read_file,
        inputs=[file_input],
        outputs=file_content,
    )
    # Update frame slider maximum when file input changes
    file_content.change(fn=update_frame_slider, inputs=file_content, outputs=frame_slider)
    file_content.change(
        fn=extract_frame,
        inputs=[file_content, frame_slider],
        outputs=frame_result,
    )
    frame_slider.change(
        fn=extract_frame,
        inputs=[file_content, frame_slider],
        outputs=frame_result,
    )
    atlas_input.change(
        fn=read_atlas,
        inputs=atlas_input,
        outputs=atlas_content,
    )
    frame_result.change(
        fn=register_frame,
        inputs=[frame_result, atlas_content, translation_x, translation_y, rotation, scale_x, scale_y, shear],
        outputs=preview_output,
    )
    atlas_content.change(
        fn=register_frame,
        inputs=[frame_result, atlas_content, translation_x, translation_y, rotation, scale_x, scale_y, shear],
        outputs=preview_output,
    )
    translation_x.change(
        fn=register_frame,
        inputs=[frame_result, atlas_content, translation_x, translation_y, rotation, scale_x, scale_y, shear],
        outputs=preview_output,
    )
    translation_y.change(
        fn=register_frame,
        inputs=[frame_result, atlas_content, translation_x, translation_y, rotation, scale_x, scale_y, shear],
        outputs=preview_output,
    )
    rotation.change(
        fn=register_frame,
        inputs=[frame_result, atlas_content, translation_x, translation_y, rotation, scale_x, scale_y, shear],
        outputs=preview_output,
    )
    scale_x.change(
        fn=register_frame,
        inputs=[frame_result, atlas_content, translation_x, translation_y, rotation, scale_x, scale_y, shear],
        outputs=preview_output,
    )
    scale_y.change(
        fn=register_frame,
        inputs=[frame_result, atlas_content, translation_x, translation_y, rotation, scale_x, scale_y, shear],
        outputs=preview_output,
    )
    shear.change(
        fn=register_frame,
        inputs=[frame_result, atlas_content, translation_x, translation_y, rotation, scale_x, scale_y, shear],
        outputs=preview_output,
    )

    # Process entire file on button click
    process_button.click(
        fn=process_entire_file,
        inputs=[file_content, atlas_content, translation_x, translation_y, rotation, scale_x, scale_y, shear],
        outputs=[output_params, output_file],
    )
