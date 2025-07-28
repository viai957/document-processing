from flask import Flask
from trustt_gpt_service.views import trustt_gpt_service
from trustt_gpt_service.db import Database
import logging
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Initialize the database
    Database.init_db(app)

    # Register Blueprints
    app.register_blueprint(trustt_gpt_service, url_prefix='/trustt/api/trustt-gpt-service')

    return app 


def show_motd():
    motd = """
  _______  _______   __    __   ______   ________  ________           ______   _______   ________       
/        |/       \ /  |  /  | /      \ /        |/        |         /      \ /       \ /        |      
$$$$$$$$/ $$$$$$$  |$$ |  $$ |/$$$$$$  |$$$$$$$$/ $$$$$$$$/         /$$$$$$  |$$$$$$$  |$$$$$$$$/       
   $$ |   $$ |__$$ |$$ |  $$ |$$ \__$$/    $$ |      $$ |    ______ $$ | _$$/ $$ |__$$ |   $$ |         
   $$ |   $$    $$< $$ |  $$ |$$      \    $$ |      $$ |   /      |$$ |/    |$$    $$/    $$ |         
   $$ |   $$$$$$$  |$$ |  $$ | $$$$$$  |   $$ |      $$ |   $$$$$$/ $$ |$$$$ |$$$$$$$/     $$ |         
   $$ |   $$ |  $$ |$$ \__$$ |/  \__$$ |   $$ |      $$ |           $$ \__$$ |$$ |         $$ |         
   $$ |   $$ |  $$ |$$    $$/ $$    $$/    $$ |      $$ |           $$    $$/ $$ |         $$ |         
   $$/    $$/   $$/  $$$$$$/   $$$$$$/     $$/       $$/             $$$$$$/  $$/          $$/          
                                                                                                """
    print(motd)

if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    show_motd()
    
    app = create_app()
    app.run(host='0.0.0.0', port=os.getenv("SERVER_PORT"),debug=True)

