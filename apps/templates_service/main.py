from flask import Flask
from routes.template import routes

app = Flask(__name__, template_folder="templates", static_folder="static")
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
