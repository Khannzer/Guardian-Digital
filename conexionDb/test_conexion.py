from conexionDb.conexionDb import ConexionDb

conexion = ConexionDb.conexionBaseDeDatos()

if conexion:
    print("Conexión exitosa a la base de datos")
    conexion.close()
else:
    print("No se pudo conectar")