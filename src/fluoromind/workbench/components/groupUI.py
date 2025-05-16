import gradio as gr
import numpy as np
from ...io import read_stack_image, read_image_array, read_single_image
from importlib.resources import files
from scipy.stats import zscore
from ...io import MaskedData
from ...core.preprocessing import gsr, bandpass, debleaching
from ...core.analysis import group_fc, group_swc, group_pca, group_cpca, group_caps
from ..config import get_workdir

default_mask_path = files("fluoromind.workbench").joinpath("mask.npy")
default_seed_point_path = files("fluoromind.workbench").joinpath("seed_points.npy")


def read_file(filenames):
    if filenames is None:
        return None
    if len(filenames) == 1:
        return read_stack_image(filenames[0])
    elif len(filenames) > 1:
        return read_image_array(filenames)
    else:
        return None


def read_mask(filename):
    if filename is None:
        return np.load(default_mask_path)
    return read_single_image(filename)


def run_analysis(
    file_input,
    mask_input,
    gsr_option,
    apply_filter,
    sampling_rate,
    low_pass_freq,
    high_pass_freq,
    normalize,
    deblaching_option,
    fc_option,
    seed_point_file,
    swc_option,
    swc_window_size,
    swc_step_size,
    swc_num_clusters,
    pca_option,
    num_pca_components,
    cpca_option,
    num_cpca_components,
    caps_option,
    num_caps_patterns,
    caps_threshold,
):
    if file_input is None:
        raise gr.Error("No data provided")
    if mask_input is None:
        mask_input = default_mask_path

    pipeline = Pipeline("fluoromind.workbench.group_analysis", workdir=get_workdir())

    @pipeline.node
    def read_data():
        return MaskedData(read_file(file_input), read_mask(mask_input))

    x = read_data()

    if apply_filter:

        @pipeline.node
        def apply_filter(x):
            return bandpass(x, low_pass_freq, high_pass_freq, sampling_rate)

        x = apply_filter(x)

    if normalize:

        @pipeline.node
        def normalize(x):
            return zscore(x, axis=0)

        x = normalize(x)

    if deblaching_option:

        @pipeline.node
        def _debleaching(x):
            return debleaching(x)

        x = _debleaching(x)

    def analysis(name, dependencies):
        if fc_option or swc_option:
            if seed_point_file is None:
                seed_points = np.load(default_seed_point_path)
            else:
                seed_points = read_single_image(seed_point_file)

            if fc_option:
                pipeline.add_node(
                    name + "_fc",
                    lambda x: group_fc(x, seed_indices=seed_points),
                    dependencies=dependencies,
                )
                pipeline.output_nodes.append(name + "_fc")
            if swc_option:
                pipeline.add_node(
                    name + "_swc",
                    lambda x: group_swc(
                        x,
                        window_size=swc_window_size,
                        stride=swc_step_size,
                        n_clusters=swc_num_clusters,
                        seed_indices=seed_points,
                    ),
                    dependencies=dependencies,
                )
                pipeline.output_nodes.append(name + "_swc")

        # These analyses can potentially run in parallel
        if pca_option:
            pipeline.add_node(
                name + "_pca",
                lambda x: group_pca(x, n_components=num_pca_components if num_pca_components > 0 else None),
                dependencies=dependencies,
            )
            pipeline.output_nodes.append(name + "_pca")

        if cpca_option:
            pipeline.add_node(
                name + "_cpca",
                lambda x: group_cpca(
                    x,
                    n_components=num_cpca_components if num_cpca_components > 0 else None,
                    whiten=False,
                ),
                dependencies=dependencies,
            )
            pipeline.output_nodes.append(name + "_cpca")

        if caps_option:
            pipeline.add_node(
                name + "_caps",
                lambda x: group_caps(
                    x,
                    n_patterns=num_caps_patterns if num_caps_patterns > 0 else None,
                    threshold=caps_threshold,
                ),
                dependencies=dependencies,
            )
            pipeline.output_nodes.append(name + "_caps")

    @pipeline.node(name="identity")
    def identity(x):
        return x

    x = identity(x)

    if gsr_option == "With GSR":
        pipeline.add_node(
            "gsr",
            lambda x: gsr(x),
            dependencies=["identity"],
        )
        analysis("gsr", ["gsr"])
    elif gsr_option == "Without GSR":
        analysis("no_gsr", ["identity"])
        pipeline.output_nodes = ["no_gsr"]
    elif gsr_option == "Both":
        pipeline.add_node(
            "gsr",
            lambda x: gsr(x),
            dependencies=["identity"],
        )
        analysis("gsr", ["gsr"])
        analysis("no_gsr", ["identity"])
    # Run the analyses

    return pipeline.run()


with gr.Blocks() as ui:
    processing_result = gr.State(None)

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Inputs")
                with gr.Row():
                    file_input = gr.File(
                        label="Registered Images",
                        file_count="multiple",
                        interactive=True,
                    )
                    mask_input = gr.File(
                        label="Mask File",
                        file_count="single",
                        interactive=True,
                    )

                gr.Markdown("## Preprocessing Options")

                with gr.Column():
                    filter_checkbox = gr.Checkbox(
                        label="Apply Signal Filtering",
                        value=True,
                        interactive=True,
                    )

                    with gr.Row(visible=True) as filter_params_group:
                        sampling_rate = gr.Number(
                            label="Sampling Rate (Hz)",
                            value=1,
                            scale=1,
                            visible=True,
                        )
                        low_pass_freq = gr.Number(
                            label="Low-pass Frequency (Hz)",
                            value=1,
                            scale=1,
                            visible=True,
                        )
                        high_pass_freq = gr.Number(
                            label="High-pass Frequency (Hz)",
                            value=1,
                            scale=1,
                            visible=True,
                        )

                    filter_checkbox.change(
                        fn=lambda checked: gr.update(visible=checked),
                        inputs=filter_checkbox,
                        outputs=filter_params_group,
                    )

                    with gr.Row():
                        normalize_checkbox = gr.Checkbox(
                            label="Normalize Data",
                            value=True,
                            interactive=True,
                        )
                        deblaching_checkbox = gr.Checkbox(
                            label="Remove Bleaching Artifacts",
                            value=True,
                            interactive=True,
                        )
                    gsr_checkbox = gr.Radio(
                        label="Global Signal Regression (GSR)",
                        choices=["Without GSR", "With GSR", "Both"],
                        value="Both",
                        interactive=True,
                    )

            with gr.Column(scale=1):
                gr.Markdown("## Analysis Methods")

                with gr.Row():
                    fc_checkbox = gr.Checkbox(
                        label="Functional Connectivity Analysis",
                        value=True,
                        interactive=True,
                    )
                    swc_checkbox = gr.Checkbox(
                        label="Sliding Window Correlation Analysis",
                        value=True,
                        interactive=True,
                    )

                with gr.Row(visible=True) as fc_params_group:
                    seed_point_file = gr.File(
                        label="Seed Point File",
                        file_count="single",
                        interactive=True,
                        visible=True,
                    )

                    fc_checkbox.change(
                        fn=lambda checked, swc_checked: gr.update(visible=checked or swc_checked),
                        inputs=[fc_checkbox, swc_checkbox],
                        outputs=fc_params_group,
                    )

                    with gr.Column(visible=True) as swc_params_group:
                        swc_window_size = gr.Number(
                            label="Window Size",
                            value=100,
                            scale=1,
                            visible=True,
                        )
                        swc_step_size = gr.Number(
                            label="Step Size",
                            value=100,
                            scale=1,
                            visible=True,
                        )
                        swc_num_clusters = gr.Number(
                            label="Number of Clusters",
                            value=10,
                            minimum=1,
                            maximum=100,
                            step=1,
                            visible=True,
                        )

                swc_checkbox.change(
                    fn=lambda checked, fc_checked: [
                        gr.update(visible=checked),
                        gr.update(visible=checked or fc_checked),
                    ],
                    inputs=[swc_checkbox, fc_checkbox],
                    outputs=[swc_params_group, fc_params_group],
                )

                pca_checkbox = gr.Checkbox(
                    label="Principal Component Analysis",
                    value=True,
                    interactive=True,
                )
                with gr.Row(visible=True) as pca_params_group:
                    num_pca_components = gr.Number(
                        label="Number of Principal Components (0 = auto)",
                        value=0,
                        scale=1,
                        visible=True,
                    )

                pca_checkbox.change(
                    fn=lambda checked: gr.update(visible=checked),
                    inputs=pca_checkbox,
                    outputs=pca_params_group,
                )

                cpca_checkbox = gr.Checkbox(
                    label="Complex Principal Component Analysis",
                    value=True,
                    interactive=True,
                )

                with gr.Row(visible=True) as cpca_params_group:
                    num_cpca_components = gr.Number(
                        label="Number of Complex Principal Components (0 = auto)",
                        value=0,
                        scale=1,
                        visible=True,
                    )

                cpca_checkbox.change(
                    fn=lambda checked: gr.update(visible=checked),
                    inputs=cpca_checkbox,
                    outputs=cpca_params_group,
                )

                caps_checkbox = gr.Checkbox(
                    label="Co-activation Patterns Analysis",
                    value=True,
                    interactive=True,
                )

                with gr.Row(visible=True) as caps_params_group:
                    num_caps_patterns = gr.Number(
                        label="Number of Patterns (0 = auto)",
                        value=0,
                        scale=1,
                        visible=True,
                    )
                    caps_threshold = gr.Number(
                        label="Threshold",
                        value=0,
                        scale=1,
                        visible=True,
                    )

                caps_checkbox.change(
                    fn=lambda checked: gr.update(visible=checked),
                    inputs=caps_checkbox,
                    outputs=caps_params_group,
                )

        run_button = gr.Button("Run Analysis")
        run_button.click(
            fn=run_analysis,
            inputs=[
                file_input,
                mask_input,
                gsr_checkbox,
                filter_checkbox,
                sampling_rate,
                low_pass_freq,
                high_pass_freq,
                normalize_checkbox,
                deblaching_checkbox,
                fc_checkbox,
                seed_point_file,
                swc_checkbox,
                swc_window_size,
                swc_step_size,
                swc_num_clusters,
                pca_checkbox,
                num_pca_components,
                cpca_checkbox,
                num_cpca_components,
                caps_checkbox,
                num_caps_patterns,
                caps_threshold,
            ],
            outputs=processing_result,
        )

        with gr.Column():
            gr.Markdown("## Results")

            @gr.render(inputs=processing_result)
            def show_results(processing_result):
                # Implement the logic to display results
                return processing_result
