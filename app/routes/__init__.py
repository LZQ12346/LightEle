# import all routes from every module
from .test_routes import *
from .layer_routes import *
from .device_routes import *
from .light_routes import *
from .nk_routes import *
from .cal_routes import *
from .auth_routes import *

from flask import Blueprint


main_bp = Blueprint('main', __name__, url_prefix="/api")

main_bp.register_blueprint(test_bp, url_prefix='/test')
main_bp.register_blueprint(layer_bp, url_prefix='/layer')
main_bp.register_blueprint(device_bp, url_prefix='/device')
main_bp.register_blueprint(light_bp, url_prefix='/light')
main_bp.register_blueprint(nk_bp, url_prefix='/nk')
main_bp.register_blueprint(cal_bp, url_prefix='/cal')
main_bp.register_blueprint(auth_bp, url_prefix='/auth')
