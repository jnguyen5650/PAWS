"""
Compose the main Streamlit settings and process columns.
"""

import streamlit as st

from constants import ICON_ROCKET
from ui_jobs import handle_process_click
from ui_jobs import render_job_panel
from ui_models import render_model_settings
from ui_output import render_output_settings
from ui_sources import render_upload_section


def render_settings_column():
    """
    Render the left-side source and settings panels.

    Returns:
        tuple: Source values, model settings, and output settings.
    """
    with st.container(border=True, key="upload-container"):
        use_local, local_paths, uploads = render_upload_section()

    with st.container(border=True, key="config-container"):
        model_settings = render_model_settings()

    with st.container(border=True, key="output-container"):
        output_settings = render_output_settings()

    return use_local, local_paths, uploads, model_settings, output_settings


def render_process_column(
    use_local, local_paths, uploads, model_settings, output_settings
):
    """
    Render the right-side process controls and job panel.

    Args:
        use_local (bool): True when local-file mode is enabled.
        local_paths (list): Selected local file paths.
        uploads (list): Uploaded file objects.
        model_settings (dict): Model inference settings dictionary.
        output_settings (dict): Output settings dictionary.

    Returns:
        None: Renders the Run Inference button and job panel.
    """
    with st.container(border=True, key="processing-container"):
        if st.button(
            "Run Inference",
            type="primary",
            icon=ICON_ROCKET,
            use_container_width=True,
        ):
            handle_process_click(
                use_local, local_paths, uploads, model_settings, output_settings
            )
        render_job_panel()
