import os
import sqlite3

import pandas as pd
import streamlit as st

from py_challenge_frontend.models import (
    AnimalCharacteristics,
    ModelPrediction,
)

SQLITE_DB = os.environ.get("PY_CHALLENGE_SQLITE_DB")


def get_db_connection():
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    return conn


def ensure_db_schema(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS animal_characteristics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            walks_on_n_legs INTEGER NOT NULL,
            height REAL NOT NULL,
            weight REAL NOT NULL,
            has_wings BOOLEAN NOT NULL,
            has_tail BOOLEAN NOT NULL,
            species TEXT NOT NULL
        );
        """)


def insert_into_db(characteristics: AnimalCharacteristics, prediction: ModelPrediction):
    conn = get_db_connection()
    cursor = conn.cursor()
    ensure_db_schema(cursor)
    cursor.execute(
        "INSERT INTO animal_characteristics (walks_on_n_legs, height, weight, has_wings, has_tail, species) VALUES (?, ?, ? ,?, ?, ?)",
        (
            characteristics.walks_on_n_legs,
            characteristics.height,
            characteristics.weight,
            characteristics.has_wings,
            characteristics.has_tail,
            prediction.species,
        ),
    )
    conn.commit()
    conn.close()


def render_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    ensure_db_schema(cursor)
    cursor.execute("""
    SELECT * FROM animal_characteristics;
    """)
    data = cursor.fetchall()
    conn.commit()
    conn.close()
    st.markdown("### Historical data")
    if not data:
        st.write("No data in the database")
    else:
        df = pd.DataFrame(data, columns=data[0].keys())
        st.write(df)
