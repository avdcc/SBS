/*jshint esversion: 6 */

var express = require('express');
var router = express.Router();
var axios = require('axios')


var websiteTitle = 'tp2'

var fs = require('fs')
var parseCSV = require('papaparse')

var file = fs.readFileSync('../filmes.csv', 'utf8')

var filmes 

parseCSV.parse(file,{
  delimiter: "§",
  encoding: "utf8",
  newline: "\n",
  complete: (results) => {
    filmes = results.data
  }
})


function parseLineFilmes(line){
  res = line.split(';')
  return res
}

function parseFilmesDatabase(filmes){
  var result = []
  var headers = parseLineFilmes(filmes[0][0])

  for(var i=1; i<filmes.length; i++){
    var line = parseLineFilmes(filmes[i][0])
    var len = headers.length
    var aux = {}  
    for(var k=0;k<len;k++){
      aux[headers[k]] = line[k]
    }
    result.push(aux)
  }	

  return result

}

filmes = parseFilmesDatabase(filmes)



//auxiliar functions

function data_imdbid(id,database){
	for(var i=0; i<database.length; i++){
    if(database[i]['imdb_id']==id){
      return database[i];
    }
  }	
}


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

function adjustKeys(movieInfo){
  newinfo = {}

  for(var key in movieInfo){
    if(key=='Internet Movie Database' ){
      newinfo['IMD'] = movieInfo['Internet Movie Database']
    }else if(key=='Rotten tomatoes'){
      newinfo['rotten'] = movieInfo['Rotten tomatoes']
    }else{
      newinfo[key] = movieInfo[key]
    }
  }

  return newinfo
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
  movieInfo = adjustKeys(movieInfo)
  res.render('filmTemplate',{film: movieInfo,title:websiteTitle})
})





//content-based filtering

router.get('/contentBased',(req,res)=>{
  res.render('filterMethods/contentBased',{ title: websiteTitle })
})

router.post('/contentBased',(req,res)=>{
  var user = req.body.user
  var selectedTitle = req.body.selectedTitle

  var usage 
  var titleOrId

  if(user){ 
    usage = 'user' 
    titleOrId = user
  }else if(selectedTitle){ 
    usage = 'title' 
    titleOrId = selectedTitle
  }


  axios.get('http://localhost:5000/contentBased/' + titleOrId + '/' + usage)
       .then(dataRec =>{
         var listString = JSON.stringify(dataRec.data)
         var listData = JSON.parse(listString).result.slice(0,9)
         var dataProcessed = idListToMovies(listData)
         //21 campos por cada entrada de listRec
         //estamos a limitar a 10 entradas do array(caso contrário demora muito tempo)
         res.render('listDataFromFilms',{
           films: dataProcessed ,
           title: websiteTitle,
           arrayOptions: []
         })
       }
       )
       .catch(erro =>{
          console.log('Erro na listagem de utilizadores: ' + erro)
          res.redirect('/')
       })
})



router.get('/contentBased/title',(req,res)=>{
  res.render('filterMethods/contentBasedTitle',{ title: websiteTitle})
})

router.get('/contentBased/user',(req,res)=>{
  res.render('filterMethods/contentBasedUser',{ title: websiteTitle})
})




//collaborative filtering

router.get('/collaborativeBased',(req,res)=>{
  res.render('filterMethods/collaborativeBased',{ title: websiteTitle})
})


router.post('/collaborativeBased',(req,res)=>{
  var user = req.body.user

  axios.get('http://localhost:5000/collaborativeBased/' + user)
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
          res.redirect('/')
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
          res.redirect('/')
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
          res.redirect('/')
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
          res.redirect('/')
       })
})



//wsBestRated

router.get('/wsBestRated',(req,res)=>{
  res.render('filterMethods/wsBestRated',{title: websiteTitle})
})

router.post('/wsBestRated',(req,res)=>{
  var site = req.body.selected

  axios.get('http://localhost:5000/wsBestRated/' + site)
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
          res.redirect('/')
       })
})






module.exports = router;
