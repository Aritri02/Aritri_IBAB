#!/bin/bash
current_time=$(date +%H)

if [[ $current_time -ge 05 && $current_time -lt 12 ]]
then 
	echo "Good morning!"
elif [[ $current_time -ge 12 && $current_time -lt 18 ]]
then 
	echo "Good afternoon!"
elif [[ $current_time -ge 19 && $current_time -lt 05 ]]
then 
	echo "Good night!"
fi
