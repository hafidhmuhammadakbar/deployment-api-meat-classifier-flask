from app.api import app

PORT = 8000
HOST = '0.0.0.0'

if __name__ == '__main__':
    app.run(port=PORT, host=HOST)