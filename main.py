#main.py
from bokeh.embed import server_document
from flask import Flask, request, render_template
from flask_sockets import Sockets
import subprocess

app = Flask(__name__)
sockets = Sockets(app)

bokeh_process = subprocess.Popen(
    ['bokeh', 'serve','--allow-websocket-origin="*"','dashboard.py'], stdout=subprocess.PIPE)


@sockets.route('/chat')
def chat_socket(ws):
    while not ws.closed:
        message = ws.receive()
        if message is None:  # message is "None" if the client has closed.
            continue
        # Send the message to all clients connected to this webserver
        # process. (To support multiple processes or instances, an
        # extra-instance storage or messaging system would be required.)
        clients = ws.handler.server.clients.values()
        for client in clients:
            client.ws.send(message)


@app.route('/')
def index():
    script = server_document("http://localhost:5006/dashboard")
#     script = server_document("https://5006-dot-7920283-dot-devshell.appspot.com/dashboard")
#     print(f"Script: {script}")
    return render_template('index.html', plot_script=script)

if __name__ == '__main__':
    app.run(port='8080')
