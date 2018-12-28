/*jshint esversion: 6 */

$(() => {
  $( document ).ready(() => {
    var x = document.getElementById("features");
    x.style.display = "none";
  });

  $("#featuresToggle").click(e=>{
    e.preventDefault()
    var x = document.getElementById("features");
    if (x.style.display === "none") {
      x.style.display = "block";
    } else {
      x.style.display = "none";
    }
  })
})
