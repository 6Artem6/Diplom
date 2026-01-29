from flask import Blueprint, render_template

routes = Blueprint("routes", __name__)


@routes.route("/login")
def login():
    return render_template("login.html.j2")


@routes.route("/dashboard")
def dashboard():
    return render_template("dashboard.html.j2")


@routes.route("/form")
def form():
    return render_template("form.html.j2")


@routes.route("/")
def index():
    return render_template("index.html")


@routes.route("/about")
def about():
    return render_template("about.html")


@routes.route("/index.php")
def legacy():
    # эмуляция старого URL
    return render_template("contact.html")


@routes.route("/search")
def search():
    return render_template("search.html", q="demo")
