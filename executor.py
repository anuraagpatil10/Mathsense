import streamlit as st

def execute_script(script: str):
    try:
        # Safe built-in functions only
        safe_globals = {
            "__builtins__": __builtins__,
            "st": st,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
        }
        exec(script, safe_globals)
    except Exception as e:
        st.error(f"Execution failed: {e}")
