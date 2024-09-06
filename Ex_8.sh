#!/bin/bash
read -p "Enter the number" n
read -p "Enter the threshold value" threshold
for (( i=0 ; i<=n ; i++ ))
do
	if (( "$i" > "$threshold" ))
	then
		echo "The number $i is greater than threshold $threshold" >> ex8.txt
	else
		echo "$i" >> ex8.txt
	fi
done