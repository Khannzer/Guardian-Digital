import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

class ConexionDb:
    
    @staticmethod
    def conexionBaseDeDatos():
        try:
            conexion = mysql.connector.connect(
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                port=os.getenv("DB_PORT"),
                buffered=True   # ← soluciona "Unread result found" en todas las rutas
            )
            return conexion

        except mysql.connector.Error as error:
            print(f"Error al conectar con la base de datos: {error}")
            return None

