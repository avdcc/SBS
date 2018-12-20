/*jshint esversion: 6 */

var express = require('express');
var router = express.Router();
var axios = require('axios')
var stringy = require('stringy')

var websiteTitle = 'title'

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
  //here we will handle the actual data passed from the user
  axios.get('http://localhost:5000/test')
       .then(dataRec => res.render('message', {
        message:"data received: " + stringy.stringify(dataRec.data) , title: websiteTitle
       }))
       .catch(erro =>{
          console.log('Erro na listagem de utilizadores: ' + erro)
          res.render('index')
       })
})



//collaborative filtering

router.get('/collFilt',(req,res)=>{
  res.render('collaborative',{ title: websiteTitle })
})

router.post('/collFilt',(req,res)=>{
  res.render('message',{message:"you wrote: " + req.body.data, title: websiteTitle})
})



//userBestRated

router.get('/userBestRated',(req,res)=>{
  res.render('filterMethods/userBestRatedGET')
})

router.post('/userBestRated',(req,res)=>{
  res.render('filterMethods/userBestRatedPOST')
})



//userMostPopular

router.get('/userMostPopular',(req,res)=>{
  res.render('filterMethods/userMostPopularGET')
})


router.post('/userMostPopular',(req,res)=>{
  res.render('filterMethods/userMostPopularPOST')
})



//wsBestRated

router.get('/wsBestRated',(req,res)=>{
  res.render('filterMethods/wsBestRatedGET')
})

router.post('/wsBestRated',(req,res)=>{
  res.render('filterMethods/wsBestRatedPOST')
})



//wsMostPopular

router.get('/wsMostPopular',(req,res)=>{
  res.render('filterMethods/wsMostPopularGET')
})

router.post('/wsMostPopular',(req,res)=>{
  res.render('filterMethods/wsMostPopularPOST')
})




module.exports = router;
