from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from sudoku_service import sudoko_service

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            # Process image
            output_filename = "solved_" + filename
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            sudoko_service(input_path, output_path)

            return render_template(
                "result.html",
                input_image=filename,
                output_image=output_filename
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)