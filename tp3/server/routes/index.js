/*jshint esversion: 6 */

var express = require('express');
var router = express.Router();

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
})



//collaborative filtering

router.get('/collFilt',(req,res)=>{
  res.render('collaborative',{ title: websiteTitle })
})

router.post('/collFilt',(req,res)=>{
  //here we will handle the actual data passed from the user
})

module.exports = router;
