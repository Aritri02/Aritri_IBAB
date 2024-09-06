#!/bin/bash
#set -x
threshold=70
used=$(df "$HOME" | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $used -ge $threshold ]
then
	echo "WARNING: Disk usage of $HOME directory is ${used}% which exceeds ${threshold}% "
else
	echo "Disk usage of $HOME directory is ${used}% which is less than ${threshold}%"
fi
#set +x

#!/bin/bash

# # Set the threshold percentage
# threshold=70

# # Get the disk usage percentage for the file system containing the $HOME directory
# usage=$(df "$HOME" | awk 'NR==2 {print $5}' | sed 's/%//')

# # Check if the usage exceeds the threshold
# if [ "$usage" -ge "$threshold" ]
#  then
#     echo "Warning: Disk usage of $HOME directory is ${usage}% which exceeds the threshold of ${threshold}%."
# else
#     echo "Disk usage of $HOME directory is ${usage}% which is below the threshold of ${threshold}%."
# fi