#!/bin/bash
root_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
home_usage=$(df /home | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$root_usage" -eq 0 ]
	 then
    echo "Root directory usage is 0%, cannot compute the relative percentage."
    exit 1
fi
relative_percentage=$( echo "scale=2; $home_usage / $root_usage * 100" | bc )
echo "Home directory usage as a percentage of root directory usage: $relative_percentage%"
#echo "$home_usage"
#echo "$root_usage"
#echo "(($home_usage / $root_usage * 100))" | bc