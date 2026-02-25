import mysql.connector

class ConexionDb:
    
    @staticmethod
    def conexionBaseDeDatos():
        try:
            conexion = mysql.connector.connect(
                user='root', 
                password='root',
                host='localhost',
                database='proyectoContraSuicidioV1',
                port='3306'
            )
            return conexion

        except mysql.connector.Error as error:
            print("Error al conectar con la base de datos {}".format(error))
            return None
        
