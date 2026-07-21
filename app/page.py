"""
Set up the Streamlit page and shared visual elements.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from constants import APP_SUBTITLE
from constants import APP_TITLE
from constants import DEFAULT_MODEL_DIR
from constants import DEFAULT_OUT_DIR

_NETWORK_BACKGROUND_HTML = """
<div id="netbg-host"><canvas id="netbg"></canvas></div>
<script>
  (() => {
    const canvas = document.getElementById("netbg");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const cfg = {
      n: 75,
      maxDist: 200,
      speed: 0.15,
      dotColor: "rgba(47,246,255, 0.10)",
      lineColorPrefix: "rgba(65,167,255, ",
      lineOpacity: 0.15,
    };

    let w = 0, h = 0, dpr = 1;
    function resize() {
      dpr = Math.max(1, window.devicePixelRatio || 1);
      w = canvas.width = Math.floor(window.innerWidth * dpr);
      h = canvas.height = Math.floor(window.innerHeight * dpr);
      canvas.style.width = window.innerWidth + "px";
      canvas.style.height = window.innerHeight + "px";
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    const rand = (a, b) => a + Math.random() * (b - a);
    const pts = [];

    function resetPoints() {
      pts.length = 0;
      for (let i = 0; i < cfg.n; i += 1) {
        pts.push({
          x: rand(0, w),
          y: rand(0, h),
          vx: rand(-cfg.speed, cfg.speed) * dpr,
          vy: rand(-cfg.speed, cfg.speed) * dpr,
          r: rand(1.1, 2.2) * dpr
        });
      }
    }

    function step() {
      ctx.clearRect(0, 0, w, h);

      for (const p of pts) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = w;
        else if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h;
        else if (p.y > h) p.y = 0;
      }

      for (let i = 0; i < pts.length; i += 1) {
        for (let j = i + 1; j < pts.length; j += 1) {
          const a = pts[i], b = pts[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist >= cfg.maxDist * dpr) continue;

          const t = 1 - (dist / (cfg.maxDist * dpr));
          const alpha = cfg.lineOpacity * (t * t);
          ctx.strokeStyle = cfg.lineColorPrefix + alpha.toFixed(4) + ")";
          ctx.lineWidth = (1 * dpr) * (0.6 + 0.8 * t);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      ctx.fillStyle = cfg.dotColor;
      ctx.shadowColor = "rgba(47,246,255, 0.30)";
      ctx.shadowBlur = 10 * dpr;
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
      requestAnimationFrame(step);
    }

    window.addEventListener("resize", () => {
      resize();
      resetPoints();
    }, { passive: true });

    resize();
    resetPoints();
    requestAnimationFrame(step);
  })();
</script>
"""

_FOOTER_HTML = """
<div class="stFooter">
  <div class="footer-row">
    <div class="footer-content">
      <p class="footer-title">&#128062; PAWS: Practical Video Super-Resolution</p>
      <p class="footer-sub">
        Developed by Justin Nguyen. Licensed under Apache 2.0.
      </p>
    </div>
    <a class="footer-link" href="#top">
      Back to Top
    </a>
  </div>
</div>
"""


def inject_network_background():
    """
    Render the animated network background.

    Returns:
        None: Renders the animated network background HTML.
    """
    width = 0
    height = 0
    if hasattr(st, "iframe"):
        if isinstance(width, (int, float)) and width <= 0:
            width = "stretch"
        if isinstance(height, (int, float)) and height <= 0:
            height = 1
        st.iframe(_NETWORK_BACKGROUND_HTML, height=height, width=width)
        return
    components.html(_NETWORK_BACKGROUND_HTML, height=height, width=width)


def set_page():
    """
    Configure the Streamlit page and load the app CSS.

    Returns:
        None: Sets page config and injects CSS.
    """
    st.set_page_config(
        page_title="PAWS Video Super-Resolution",
        page_icon="PAWS",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    css_path = Path(__file__).with_name("app_theme.css")
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True
    )


def header():
    """
    Render the page header.

    Returns:
        None: Renders the title and subtitle.
    """
    st.html('<div id="top"></div>')
    st.title(APP_TITLE, text_alignment="center")
    st.html(f'<p class="header-subtitle">{APP_SUBTITLE}</p>')


def ensure_session_defaults():
    """
    Initialize required Streamlit session state.

    Returns:
        None: Initializes output_dir and model_dir in session state.
    """
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = DEFAULT_OUT_DIR
    if "model_dir" not in st.session_state:
        st.session_state.model_dir = DEFAULT_MODEL_DIR


def render_footer():
    """
    Render the fixed page footer.

    Returns:
        None: Renders the fixed page footer HTML.
    """
    st.html(_FOOTER_HTML)
