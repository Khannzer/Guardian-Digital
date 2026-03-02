import mysql.connector
import os
from dotenv import load_dotenv

#cargar las variables de entorno desde el archivo .env
load_dotenv()

class ConexionDb:
    
    @staticmethod
    def conexionBaseDeDatos():
        try:
            conexion = mysql.connector.connect(
                user= os.getenv("DB_USER"),
                password= os.getenv("DB_PASSWORD"),
                host= os.getenv("DB_HOST"),
                database= os.getenv("DB_NAME"),
                port= os.getenv("DB_PORT")
            )
            return conexion

        except mysql.connector.Error as error:
            print("Error al conectar con la base de datos {}".format(error))
            return None
