/**
 * 
 */

   /* Validación de todos los campos, si todo esta OK se registra, si no, muestra mensaje de error */
   $("#id_btn_iniciosesion").click(function(){
	    var validator = $('#id_form').data('bootstrapValidator');
        validator.validate();
	
        if (validator.isValid()) {
            $.ajax({
              type: "POST",
              url: "logueo", 
              data: $('#id_form').serialize(),
              success: function(data){
        	      mostrarMensaje(data.mensaje);
        	      limpiarFormulario();
        	      validator.resetForm();
              },
              error: function(){
        	      mostrarMensaje(MSG_ERROR);
              }
            });
        
        }
    });
    
    $("#id_btn_limpiar").click(function(){
		 limpiarFormulario();
    });
    
    /* función para limpiar el formulario dps de un registro */
     function limpiarFormulario(){	
		$('#id_usuario').val('');
		$('#id_contraseña').val('');
	} 