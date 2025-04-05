import io
import matplotlib.pyplot as plt
import contextlib

def execute_script(code: str):

    local_env = {}
    plot_buffer = io.StringIO()

    try:
        plt.close("all")  # ← clear any previous figures to avoid duplicate/ghost plots
        with contextlib.redirect_stdout(plot_buffer):
            exec(code, {"plt": plt}, local_env)

        fig = plt.gcf()
        return {"plot": fig}

    except Exception as e:
        return {"error": f"Execution failed: {str(e)}"}
