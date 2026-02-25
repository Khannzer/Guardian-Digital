const temaOscuro = () => {
	document.querySelector("body").setAttribute("data-bs-theme", "dark");
	document.querySelector("body").setAttribute("style", "background-color:#282928; color:black;");
	document.getElementById("idfiltro").setAttribute("style", "color:white;");
	document.querySelector("h1").setAttribute("style", "color:white; margin:auto; margin-top:30px;")
	document.querySelector("#dl-icon").setAttribute("class", "bi bi-sun-fill");
	
	
	var elementos = document.querySelectorAll("#id_table td");
    elementos.forEach(elemento => {
    elemento.style.color = 'white';
  });
};

const temaClaro = () => {
	document.querySelector("body").setAttribute("data-bs-theme", "light");
	document.querySelector("body").setAttribute("style", "background-color:white; color:black;");
	document.getElementById("idfiltro").setAttribute("style", "color:black;");
	document.querySelector("h1").setAttribute("style", "color:black; margin:auto; margin-top:30px;")
	document.querySelector("#dl-icon").setAttribute("class", "bi bi-moon-fill");
	
	var elementos = document.querySelectorAll("#id_table td");
    elementos.forEach(elemento => {
    elemento.style.color = 'black';
  });
};

const cambiarTema = () => {
	document.querySelector("body").getAttribute("data-bs-theme") === "light"?
	temaOscuro() : temaClaro();
}
