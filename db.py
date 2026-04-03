"""
db.py — MySQL Database Helper for Smart Queue Management System
--------------------------------------------------------------
Tables:
  patients        → Patient Intake records (tokens, triage, wait times)
  queue_history   → Served patients from Live Queue


"""

import mysql.connector
from mysql.connector import Error
from datetime import datetime
import streamlit as st


# ── DB CONFIG — Railway MySQL (public endpoint) ──
# Default fallback (local dev)
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "M@noj777",
    "database": "smartqueue_db"
}


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
def get_connection():
    """Returns a MySQL connection. Returns None if fails."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        return None


def test_connection():
    """Returns True if DB is reachable."""
    conn = get_connection()
    if conn and conn.is_connected():
        conn.close()
        return True
    return False


# ─────────────────────────────────────────────
# SETUP — Create Tables if not exist
# NOTE: Railway already provides the 'railway' database.
#       We do NOT create a new database — just create tables.
# ─────────────────────────────────────────────
def setup_database():
    """
    Creates required tables inside the existing Railway database.
    Call this once when app starts.
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # ── patients table ──
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            token_id        VARCHAR(20)   NOT NULL,
            patient_id      VARCHAR(20),
            triage          VARCHAR(30)   NOT NULL,
            age_group       VARCHAR(40),
            department      VARCHAR(60),
            appt_type       VARCHAR(60),
            insurance       VARCHAR(40),
            arrival_method  VARCHAR(40),
            reason          VARCHAR(60),
            tests           VARCHAR(40),
            consultation    VARCHAR(10),
            booking_type    VARCHAR(30),
            arrival_hour    INT,
            arrival_month   INT,
            is_weekend      TINYINT(1),
            providers       INT,
            nurses          INT,
            occupancy       FLOAT,
            composite_score INT,
            age_vuln        INT,
            pred_wait_min   FLOAT,
            staff_shortage  TINYINT(1),
            hi_occupancy    TINYINT(1),
            registered_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # ── queue_history table ──
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS queue_history (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            token_id        VARCHAR(20)   NOT NULL,
            patient_id      VARCHAR(20),
            triage          VARCHAR(30),
            age_group       VARCHAR(40),
            department      VARCHAR(60),
            ml_wait_min     FLOAT,
            dynamic_wait    FLOAT,
            composite_score INT,
            age_vuln        INT,
            status          VARCHAR(20)   DEFAULT 'SERVED',
            served_at       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        return True

    except Error as e:
        return False


# ─────────────────────────────────────────────
# PATIENTS TABLE — Insert & Fetch
# ─────────────────────────────────────────────
def insert_patient(data: dict) -> bool:
    """
    Insert a patient intake record.
    data keys match patients table columns.
    """
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        sql = """
        INSERT INTO patients
        (token_id, patient_id, triage, age_group, department, appt_type,
         insurance, arrival_method, reason, tests, consultation, booking_type,
         arrival_hour, arrival_month, is_weekend, providers, nurses, occupancy,
         composite_score, age_vuln, pred_wait_min, staff_shortage, hi_occupancy)
        VALUES
        (%(token_id)s, %(patient_id)s, %(triage)s, %(age_group)s, %(department)s,
         %(appt_type)s, %(insurance)s, %(arrival_method)s, %(reason)s, %(tests)s,
         %(consultation)s, %(booking_type)s, %(arrival_hour)s, %(arrival_month)s,
         %(is_weekend)s, %(providers)s, %(nurses)s, %(occupancy)s,
         %(composite_score)s, %(age_vuln)s, %(pred_wait_min)s,
         %(staff_shortage)s, %(hi_occupancy)s)
        """
        cursor.execute(sql, data)
        conn.commit()
        return True
    except Error:
        return False
    finally:
        cursor.close()
        conn.close()


def fetch_all_patients(limit=200):
    """Fetch recent patients from DB."""
    conn = get_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM patients ORDER BY registered_at DESC LIMIT %s",
            (limit,)
        )
        rows = cursor.fetchall()
        return rows
    except Error:
        return []
    finally:
        cursor.close()
        conn.close()


def count_patients_today():
    """Count patients registered today."""
    conn = get_connection()
    if not conn:
        return 0
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM patients WHERE DATE(registered_at) = CURDATE()"
        )
        result = cursor.fetchone()
        return result[0] if result else 0
    except Error:
        return 0
    finally:
        cursor.close()
        conn.close()


# ─────────────────────────────────────────────
# QUEUE HISTORY TABLE — Insert & Fetch
# ─────────────────────────────────────────────
def insert_served_patient(token: dict) -> bool:
    """
    Insert a served patient record from Live Queue.
    token is the queue dict from session_state.
    """
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        sql = """
        INSERT INTO queue_history
        (token_id, patient_id, triage, age_group, department,
         ml_wait_min, dynamic_wait, composite_score, age_vuln, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (
            token.get('token'),
            token.get('patient'),
            token.get('triage'),
            token.get('age_group'),
            token.get('dept'),
            token.get('ml_wait'),
            token.get('dynamic_wait'),
            token.get('composite'),
            token.get('age_vuln'),
            'SERVED'
        ))
        conn.commit()
        return True
    except Error:
        return False
    finally:
        cursor.close()
        conn.close()


def fetch_queue_history(limit=200):
    """Fetch recent served patients."""
    conn = get_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM queue_history ORDER BY served_at DESC LIMIT %s",
            (limit,)
        )
        return cursor.fetchall()
    except Error:
        return []
    finally:
        cursor.close()
        conn.close()


def count_served_today():
    """Count patients served today."""
    conn = get_connection()
    if not conn:
        return 0
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM queue_history WHERE DATE(served_at) = CURDATE()"
        )
        result = cursor.fetchone()
        return result[0] if result else 0
    except Error:
        return 0
    finally:
        cursor.close()
        conn.close()


def avg_wait_today():
    """Average predicted wait time for today's patients."""
    conn = get_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT AVG(pred_wait_min) FROM patients WHERE DATE(registered_at) = CURDATE()"
        )
        result = cursor.fetchone()
        return round(result[0], 1) if result and result[0] else None
    except Error:
        return None
    finally:
        cursor.close()
        conn.close()
