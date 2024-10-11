from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, this is a test app."

if __name__ == "__main__":
    app.run()
