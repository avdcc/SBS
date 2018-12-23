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
  header: true,
  encoding: "utf8",
  newline: "\n",
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
    if(database[i]['imdb_id']==id){
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


//páginas para os filmes
router.get('/movies/:id',(req,res)=>{
  var imdbid = req.params.id
  var movieInfo = data_imdbid(imdbid,filmes)
  var transformedInfo = csvDataToDict(movieInfo)
  res.render('filmTemplate',{film: transformedInfo,title:websiteTitle})
})





//content-based filtering

router.get('/contentBased',(req,res)=>{
  res.render('filterMethods/contentBased',{ title: websiteTitle })
})

router.post('/contentBased',(req,res)=>{
  var mainTitle = req.body.mainTitle

  var features = {
    title : req.body.title,
    actors : req.body.actors,
    country : req.body.country,
    genre : req.body.genre,
    language : req.body.language,
    writer : req.body.writer,
    plot : req.body.plot,
    director : req.body.director,
    production : req.body.production
  }

  Object.keys(features).map((key,index)=>{
    if(features[key]){
      features[key] = (features[key],1)
    }else{
      features[key] = (features[key],0)
    }
  })


  var headers = {"Content-Type" : "application/json"}

  axios.post('http://localhost:5000/contentBased/' + mainTitle, features,{headers:headers})
       .then(dataRec =>{
         console.log(dataRec)
         var listString = JSON.stringify(dataRec.data)
         var listData = JSON.parse(listString).result.slice(0,9)
         var dataProcessed = idListToMovies(listData)
         //21 campos por cada entrada de listRec
         //estamos a limitar a 10 entradas do array(caso contrário demora muito tempo)
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



//collaborative filtering

router.get('/collaborativeBased',(req,res)=>{
  res.render('filterMethods/collaborativeBased',{ title: websiteTitle })
})


router.post('/collaborativeBased',(req,res)=>{
  var user = req.body.user

  var features = {
    title : req.body.title,
    actors : req.body.actors,
    country : req.body.country,
    genre : req.body.genre,
    language : req.body.language,
    writer : req.body.writer,
    plot : req.body.plot,
    director : req.body.director,
    production : req.body.production
  }

  Object.keys(features).map((key,index)=>{
    if(features[key]){
      features[key] = (features[key],1)
    }else{
      features[key] = (features[key],0)
    }
  })


  var headers = {"Content-Type" : "application/json"}

  axios.post('http://localhost:5000/collaborativeBased/' + user, features,{headers:headers})
       .then(dataRec =>{
         console.log(dataRec)
         var listString = JSON.stringify(dataRec.data)
         var listData = JSON.parse(listString).result.slice(0,9)
         var dataProcessed = idListToMovies(listData)
         //21 campos por cada entrada de listRec
         //estamos a limitar a 10 entradas do array(caso contrário demora muito tempo)
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



//hibrido
router.get('/hybrid',(req,res)=>{
  res.render('filterMethods/hybrid',{ title: websiteTitle })
})

router.post('/hybrid',(req,res)=>{
  var user = req.body.user

  axios.get('http://localhost:5000/hybrid/' + user)
       .then(dataRec =>{
         var listString = JSON.stringify(dataRec.data)
         var listData = JSON.parse(listString).result.slice(0,9)
         var dataProcessed = idListToMovies(listData)
         //21 campos por cada entrada de listRec
         //estamos a limitar a 10 entradas do array(caso contrário demora muito tempo)
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






//popular
router.get('/popular',(req,res)=>{
  res.render('popular',{title:websiteTitle})
})

//userBestRated

router.get('/userBestRated',(req,res)=>{
  res.render('filterMethods/userBestRated',{title: websiteTitle})
})

router.post('/userBestRated',(req,res)=>{
  axios.get('http://localhost:5000/userBestRated')
       .then(dataRec =>{
        var listString = JSON.stringify(dataRec.data)
        
        var listData = JSON.parse(listString).result.slice(0,9)
        
        var dataProcessed = idListToMovies(listData)
        
        //21 campos por cada entrada de listRec
        //estamos a limitar a 10 entradas do array(caso contrário demora muito tempo)
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
  res.render('filterMethods/userMostPopular',{title: websiteTitle})
})


router.post('/userMostPopular',(req,res)=>{
  axios.get('http://localhost:5000/userMostPopular')
       .then(dataRec =>{
        var listString = JSON.stringify(dataRec.data)
        
        var listData = JSON.parse(listString).result.slice(0,9)
        
        var dataProcessed = idListToMovies(listData)
        
        //21 campos por cada entrada de listRec
        //estamos a limitar a 10 entradas do array(caso contrário demora muito tempo)
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



//wsBestRated

router.get('/wsBestRated',(req,res)=>{
  res.render('filterMethods/wsBestRated',{title: websiteTitle})
})

router.post('/wsBestRated',(req,res)=>{
  axios.get('http://localhost:5000/wsBestRated/' + req.body.data)
       .then(dataRec =>{
        var listString = JSON.stringify(dataRec.data)
        
        var listData = JSON.parse(listString).result.slice(0,9)
        
        var dataProcessed = idListToMovies(listData)
        
        //21 campos por cada entrada de listRec
        //estamos a limitar a 10 entradas do array(caso contrário demora muito tempo)
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






module.exports = router;
