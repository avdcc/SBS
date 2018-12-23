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
  killall node
fi

if($flaskExists)
then 
  kill -9 $flaskVal
fi
