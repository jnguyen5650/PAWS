"""
Launch the Streamlit entrypoint for the PAWS demo.
"""

import streamlit as st


def main():
    """
    Set up and render the Streamlit app.

    The UI imports stay local so spawned worker processes on Windows do not
    import Streamlit UI modules outside an active app runtime.

    Returns:
        None: Renders the complete Streamlit application.
    """
    from page import ensure_session_defaults
    from page import header
    from page import inject_network_background
    from page import render_footer
    from page import set_page
    from ui import render_process_column
    from ui import render_settings_column

    set_page()
    inject_network_background()
    ensure_session_defaults()
    header()

    col_settings, col_process = st.columns([4 / 3, 1])

    with col_settings:
        use_local, local_paths, uploads, model_settings, output_settings = (
            render_settings_column()
        )

    with col_process:
        render_process_column(
            use_local, local_paths, uploads, model_settings, output_settings
        )

    render_footer()


if __name__ == "__main__":
    main()
