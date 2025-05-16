import gradio as gr
from .components.registerUI import ui as register_ui
from .components.groupUI import ui as group_ui
from .components.statsUI import ui as stats_ui

simple_ui = gr.TabbedInterface(
    [register_ui, group_ui, stats_ui], ["Register Image", "Group Analysis", "Statistics"], title="FluoroMind Workbench"
)
