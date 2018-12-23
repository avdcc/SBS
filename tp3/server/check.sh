#!/bin/bash


function callNpm(){
  pgrep -f npm
}

function callFlask(){
  pgrep -f flask
}

npmVal="$(callNpm)"
flaskVal="$(callFlask)"

npmExists=false
flaskExists=false


if [ ! -z "$npmVal" ]
then
  npmExists=true
fi

if [ ! -z "$flaskVal" ]
then
  flaskExists=true
fi


if($npmExists)
then 
  echo "A npm server is up in pid" $npmVal
else
  echo "No npm server detected"
fi

if($flaskExists)
then 
  echo "A flask server is up in pid" $flaskVal
else
  echo "No flask server detected"
fi
