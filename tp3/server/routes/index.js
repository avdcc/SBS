/*jshint esversion: 6 */

var express = require('express');
var router = express.Router();
var axios = require('axios')


var websiteTitle = 'title'

var fs = require('fs')
var parseCSV = require('papaparse')

var file = fs.readFileSync('../filmes.csv', 'utf8')

var filmes 

parseCSV.parse(file,{
  delimiter: ";",
  complete: (results) => {
    filmes = results.data
  }
})


function aux(string){
	var res;
  for (var i = 0; i< string.length ; i++){
  	if(string[i] != "\""){
    	res.append(string[i]);
    }
  }
  return res;
}

function stripall(list){
	return list.map(x => aux(x));
}

function column(table,ind){
	var res = [];
  for(var i =0; i< table.length ; i++){
  	res += (table[i][ind]);
  }
  return res;
}

function createDic(table){
	dic = [];
  for(var i=0; table[0].length; i++){
  	dic.push({ key: table[0][i],
               value: column(table,i)})
  }
  return dic;
}

function data_imdbid(id,database){
	for(var i=0; i<database.length; i++){
    
  	if(database[i][5] == id){
    	return database[i];
    }
  }	
}

function List_toSet(List){
	var res = new Set();
  for(var i=0; i<List.length; i++){
  	res.add(List[i]);
  }
  return res;
}

function setup() {
  noCanvas();
  filmes = filmes.map(x => x.replace(/\"/g,"").split(";"));

  print(filmes[0]);
  
  //print(data_imdbid("tt0113326",filmes));
  //print(createDic(filmes));
  //print(Array.from(List_toSet(column(filmes,0))));
  //print(List_toSet(["ola","ole","ola"]));
  var atores = column(filmes,0);
   for(var i=0; i<atores.length; i++){
   	print(atores[i]);
   }
  
}













//auxiliar functions
function idListToMovies(listItems){
  var res = []
  var i=0
  var elemAux = listItems[i]
  while(elemAux){

    var elem = data_imdbid(elemAux,filmes)

    res.push(elem)
    
    i++
    elemAux = listItems[i]
  }
  return res
}


function csvDataToDict(elem){
  // "actors";"awards";"country";"director";"genre";"imdbId";
  //"imdb_rating";"imdb_votes";"language";"metascore";"plot";
  //"poster";"production";"ratings";"title";"writer";"year";
  //"dvdYear";"releasedMonth";"releasedYear";"duration"
  var res 

  if(elem){
    res = {
      actors: elem[0],awards: elem[1],country: elem[2],director: elem[3],
      genre: elem[4],imdbId: elem[5],imdb_rating: elem[6],imdb_votes: elem[7],
      language: elem[8],metascore: elem[9],plot: elem[10],poster: elem[11],
      production: elem[12],ratings: elem[13],title: elem[14],writer: elem[15],
      year: elem[16],dvdYear: elem[17],releasedMonth: elem[18],releasedYear: elem[19],
      duration: elem[20]
    }  
  }
  

  return res
}




/* GET home page. */
router.get('/', function(req, res, next) {
  //res.render(nome_do_pug_a_carregar,argumentos_a_passar_ao_pug)
  res.render('index', { title: websiteTitle });
});



//content-based filtering

router.get('/contFilt',(req,res)=>{
  res.render('content',{ title: websiteTitle })
})

router.post('/contFilt',(req,res)=>{
  //test for the python server
  axios.get('http://localhost:5000/test')
       .then(dataRec => res.render('message', {
        message:"data received: " + JSON.stringify(dataRec.data) , title: websiteTitle
       }))
       .catch(erro =>{
          console.log('Erro na listagem de utilizadores: ' + erro)
          res.render('index')
       })
  res.render('message',{message:"you wrote: " + req.body.data, title: websiteTitle})
})



//collaborative filtering

router.get('/collFilt',(req,res)=>{
  res.render('collaborative',{ title: websiteTitle })
})

router.post('/collFilt',(req,res)=>{
  //here we will handle the actual data passed from the user
  res.render('message',{message:"you wrote: " + req.body.data, title: websiteTitle})
})



//userBestRated

router.get('/userBestRated',(req,res)=>{
  res.render('filterMethods/userBestRatedGET',{title: websiteTitle})
})

router.post('/userBestRated',(req,res)=>{
  axios.get('http://localhost:5000/userBestRated')
       .then(dataRec =>{
        var listString = JSON.stringify(dataRec.data)
        var listData = JSON.parse(listString).result.slice(0,9)
        var listRec = idListToMovies(listData)
        //21 campos por cada entrada de listRec
        //estamos a limitar a 10 entradas do array(caso contrário demora muito tempo)
        var dataProcessed = listRec.map(elem => csvDataToDict(elem))
        res.render('listDataFromFilms',{
          films: dataProcessed ,
          title: websiteTitle
        })
       }
       )
       .catch(erro =>{
          console.log('Erro na listagem de utilizadores: ' + erro)
          res.render('index')
       })
})



//userMostPopular

router.get('/userMostPopular',(req,res)=>{
  res.render('filterMethods/userMostPopularGET',{title: websiteTitle})
})


router.post('/userMostPopular',(req,res)=>{
  axios.get('http://localhost:5000/userMostPopular')
       .then(dataRec => res.render('message', {
        message:"data received: " + JSON.stringify(dataRec.data) , title: websiteTitle
       }))
       .catch(erro =>{
          console.log('Erro na listagem de utilizadores: ' + erro)
          res.render('index')
       })
  //res.render('filterMethods/userMostPopularPOST',{title: websiteTitle})
})



//wsBestRated

router.get('/wsBestRated',(req,res)=>{
  res.render('filterMethods/wsBestRatedGET',{title: websiteTitle})
})

router.post('/wsBestRated',(req,res)=>{
  axios.get('http://localhost:5000/wsBestRated/' + req.body.data)
      .then(dataRec => res.render('message', {
      message:"data received: " + JSON.stringify(dataRec.data) , title: websiteTitle
      }))
      .catch(erro =>{
        console.log('Erro na listagem de utilizadores: ' + erro)
        res.render('index')
      })
  //res.render('filterMethods/wsBestRatedPOST',{title: websiteTitle})
})



//wsMostPopular

router.get('/wsMostPopular',(req,res)=>{
  res.render('filterMethods/wsMostPopularGET',{title: websiteTitle})
})

router.post('/wsMostPopular',(req,res)=>{
  res.render('filterMethods/wsMostPopularPOST',{title: websiteTitle})
})




module.exports = router;
