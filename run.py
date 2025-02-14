"""
This is the launcher.
"""
from app import create_app
from flask_cors import CORS

app = create_app()
CORS(app, supports_credentials=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9527, debug=True, ssl_context=("SSL_certification/sslconfigure.pem", "SSL_certification/sslconfigure.key"))
    # app.run(host='0.0.0.0', port=9527, debug=True)
