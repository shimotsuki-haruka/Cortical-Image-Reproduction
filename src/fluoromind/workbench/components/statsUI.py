import gradio as gr

with gr.Blocks() as ui:
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Group Results", file_count="multiple", interactive=True)
            analysis_type = gr.Dropdown(
                label="Analysis Type",
                choices=["Group Comparison", "Semi-Hierarchical Analysis"],
                value="Group Comparison",
            )
            gr.Button("Run")

        with gr.Column():
            gr.Markdown("## Results")
