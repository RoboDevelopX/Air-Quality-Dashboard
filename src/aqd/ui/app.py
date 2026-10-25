"""Top-level Streamlit page. Placeholder until the dashboard milestone."""

from __future__ import annotations

import streamlit as st

from aqd.config import Settings, load_dotenv
from aqd.log import configure_logging


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    configure_logging(settings.log_level, settings.log_format)
    st.set_page_config(page_title="Air Quality Dashboard", page_icon="🌫️", layout="wide")
    st.title("Air Quality Dashboard")
    st.info("Scaffold only. The dashboard arrives in a later milestone.")
