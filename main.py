import azure.functions as func
from function_app import app
from azure.functions import WsgiMiddleware

wsgi_app = WsgiMiddleware(app)

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return wsgi_app.handle(req, context)
