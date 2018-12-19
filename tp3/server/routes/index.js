/*jshint esversion: 6 */

var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  //res.render(nome_do_pug_a_carregar,argumentos_a_passar_ao_pug)
  res.render('index', { title: 'Express' });
});



//content-based filtering

router.get('/contFilt',()=>{
  res.render('content',{})
})

router.post('/contFilt',()=>{
  //here we will handle the actual data passed from the user
})



//collaborative filtering

router.get('/collFilt',()=>{
  res.render('collaborative',{})
})

router.post('/collFilt',()=>{
  //here we will handle the actual data passed from the user
})

module.exports = router;
